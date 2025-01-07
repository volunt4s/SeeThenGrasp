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

    with open(os.path.join(basedir, 'frames.json'), 'r') as fp:
        meta = json.load(fp)

    print("[DATA] Start data loading")

    imgs = []
    poses = []

    x_axis_convert = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    for frame in tqdm(meta['frames']):
        fname = os.path.join(basedir, frame['file_name'])
        imgs.append(imageio.v2.imread(fname))
        pose = np.array(frame['transform_matrix'])
        # x axis convert to fit nerf camera format
        pose = np.matmul(pose, x_axis_convert)
        poses.append(pose)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    cam_info = meta['cam_info']
    focal = cam_info['K'][0]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal]