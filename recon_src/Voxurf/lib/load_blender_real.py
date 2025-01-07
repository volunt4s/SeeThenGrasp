import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
from tqdm import tqdm

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir):
    basedir = './data/puang'

    with open(os.path.join(basedir, 'frames.json'), 'r') as fp:
        meta = json.load(fp)

    print("[DATA] Start data loading")

    imgs = []
    masks = []
    poses = []

    theta_x = np.pi
    theta = np.pi
    theta = np.pi/2

    x_axis_convert = lambda theta: np.array([[1.0, 0.0, 0.0, 0.0],
                                             [0.0, np.cos(theta), -np.sin(theta), 0.0],
                                             [0.0, np.sin(theta), np.cos(theta), 0.0],
                                             [0.0, 0.0, 0.0, 1.0]
                                             ])
    
    y_axis_convert = lambda theta: np.array([[np.cos(theta), 0.0, np.sin(theta), 0.0],
                                             [0.0, 1.0, 0.0, 0.0],
                                             [-np.sin(theta), 0.0, np.cos(theta), 0.0],
                                             [0.0, 0.0, 0.0, 1.0]
                                             ])

    z_axis_convert = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0.0, 0.0],
                                             [np.sin(theta), np.cos(theta), 0.0, 0.0],
                                             [0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0]
                                             ])

    for frame in tqdm(meta['frames']):
        fname = os.path.join(basedir, frame['file_name'])
        imgs.append(imageio.v2.imread(fname))
        mask_name = os.path.join(basedir, frame['mask_path'])
        mask_ = (imageio.v2.imread(mask_name) / 255.).astype(np.float32)
        if mask_.ndim == 3:
            masks.append(mask_[..., :3])
        else:
            masks.append(mask_[..., None])

        pose = np.array(frame['transform_matrix'])

        ## FOR COLMAP FITTING
        pose = np.matmul(x_axis_convert(np.pi/1.5), pose)
        # pose[:3, 3] = pose[:3, 3] * 5.3333333
        
        pose[:3, 3] = pose[:3, 3] * 7.0
        poses.append(pose)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)


    print(np.linalg.norm(poses[0, :3, 3]))
    print(np.linalg.norm(poses[1, :3, 3]))

    masks = (np.array(masks)).astype(np.float32)
    bg = 0.
    imgs = imgs * masks + bg * (1 - masks)

    H, W = imgs[0].shape[:2]
    cam_info = meta['cam_info']
    focal = cam_info['K'][0]
    # focal = 5.74474670e+02

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal], masks