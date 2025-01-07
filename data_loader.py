import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as T
import math
import numpy as np
from kornia import create_meshgrid
from tqdm import tqdm
import json

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    rays = torch.cat([rays_o, rays_d], dim=-1)
    return rays

def get_ray_directions(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.FloatTensor([0, 1, 0]).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

class ViSNeRFDataset(Dataset):
    def __init__(self, args, split='train'):
        self.resolution = args.input_res if split == 'train' else args.output_res
        self.N_vis = -1
        self.root_dir = args.datadir
        self.split = split
        self.is_stack = False if split == 'train' else True
        self.img_wh = self.resolution
        print('{} resolution: {}'.format(split, self.img_wh))
        self.transform = T.ToTensor()
        self.scene_bbox = torch.tensor(args.bbox)
        self.min_params = np.array(args.min_params)
        self.max_params = np.array(args.max_params)
        self.n_params = args.nParams
        print(self.scene_bbox)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        self.white_bg = True
        # self.near_far = [2.0,6.0]
        self.near_far = [0.1,2.0]
        # self.near_far = [0.5,4.0]
        # self.near_far = [0.01, 6.0]
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.load_all()

    
    def load_all(self):
        print(f"Loading {self.split}...")

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 / np.tan(0.5 * 30/180*math.pi) * self.img_wh[0]  # original focal length


        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_params = []
        self.downsample = 1.0

        for i, frame in enumerate(tqdm(self.meta['frames'])):
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = f"{self.root_dir}/{frame['file_path']}.png"
            self.image_paths += [image_path]
            img = Image.open(image_path)
            if any(img.size[i] != self.resolution[i] for i in range(len(img.size))):
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)

            rays = get_rays(self.directions, c2w)

            self.all_rgbs.append(img)
            self.all_rays.append(rays)

            params = np.array(frame['params'])
            # print(params, self.min_params)
            params = (params - self.min_params) / (self.max_params - self.min_params) * 2.0 - 1.0
            params = torch.ones(rays.size(0),1) * torch.FloatTensor(params)
            self.all_params += [params]

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_params = torch.cat(self.all_params, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_params = torch.stack(self.all_params, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

        print(self.all_rays.size(), self.all_params.size())



class ParaviewDataset(Dataset):
    def __init__(self, args, split='train'):
        self.resolution = args.input_res+args.input_res if split == 'train' else args.output_res+args.output_res
        self.N_vis = -1
        self.root_dir = args.train_path if split == 'train' else args.test_path
        self.split = split
        self.is_stack = False if split == 'train' else True
        self.img_wh = self.resolution
        print('{} resolution: {}'.format(split, self.img_wh))
        self.transform = T.ToTensor()
        self.scene_bbox = torch.tensor(args.bbox)
        self.min_params = np.array(args.min_params)
        self.max_params = np.array(args.max_params)
        self.n_params = args.nParams
        print(self.scene_bbox)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        self.white_bg = True
        # self.near_far = [2.0,6.0]
        self.near_far = [0.1,2.0]
        # self.near_far = [0.5,4.0]
        # self.near_far = [0.01, 6.0]
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.load_all()

    
    def load_all(self):
        w, h = self.img_wh
        self.focal = 0.5 / np.tan(0.5 * 30/180*math.pi) * self.img_wh[0]  # original focal length


        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_params = []

        img_files = os.listdir('{}'.format(self.root_dir))
        # img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[2]))
        img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[self.n_params+1]))


        for i, filename in enumerate(img_files):
            # _,timestep,theta,phi, x, y, z = 
            parse_list = filename.split('.png')[0].split('_')[1:]
            params = []
            for j in range(self.n_params):
                params.append(float(parse_list[j]))
            theta,phi, x, y, z = parse_list[self.n_params:]
            # timestep = int(timestep)
            theta = float(theta) / 90.0
            phi = float(phi) / 180.0
            x = float(x)
            y = float(y)
            z = float(z)

            camera_pivot = torch.FloatTensor([0,0,0])
            # camera_origins = torch.FloatTensor([x,y,z]) * 4.0311
            camera_origins = torch.FloatTensor([x,y,z])

            swap_row = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]])
            blender2opencv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            c2w = create_cam2world_matrix(normalize_vecs(camera_pivot + 1e-8 - camera_origins), camera_origins)
            c2w = c2w[0]
            c2w = swap_row @ c2w
            c2w = c2w * mask
            c2w = c2w + 1e-8
            print(c2w)
            exit()
            c2w = c2w @ blender2opencv

            self.poses += [c2w]

            image_path = '{}/{}'.format(self.root_dir, filename)
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if img.size != self.resolution:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

            rays = get_rays(self.directions, c2w)

            self.all_rgbs.append(img)
            self.all_rays.append(rays)

            param_list = []
            for j in range(self.n_params):
                p = (params[j] - self.min_params[j]) / (self.max_params[j] - self.min_params[j]) * 2.0 - 1.0
                param_j = torch.ones(rays.size(0),1) * p
                # print(param.size())
                param_list.append(param_j)
            # time = torch.ones(rays.size(0),1) * (timestep - self.time_range[0]) / (self.time_range[1] - self.time_range[0]) * 2.0 - 1.0
            # if self.n_params > 1:
            param = torch.cat(param_list, -1)
                # print(param.size())
                # print(param.min(), param.max())
            self.all_params += [param]
            # print(self.all_params)
            # else:
                # self.all_params += [param_list]

            # self.all_params = torch.tensor(self.all_params)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_params = torch.cat(self.all_params, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_params = torch.stack(self.all_params, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)