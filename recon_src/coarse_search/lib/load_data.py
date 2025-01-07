import numpy as np
from .load_dtu import load_dtu_data

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

def load_data(args):

    K, depths = None, None
    reso_level = 1
    wmask = True
    white_bg = False

    images, poses, render_poses, hwf, K, i_split, scale_mats_np, masks = load_dtu_data(args.datadir, reso_level=reso_level, mask=wmask, white_bg=white_bg)
    print('Loaded dtu', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split
    
    train_all = True
    if train_all:
        i_train = np.arange(int(images.shape[0]))

    # near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])
    near, far = 0.001, 1.0

    assert images.shape[-1] == 3

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]
    i_train = np.arange(images.shape[0])

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        poses=poses, render_poses=render_poses, i_train=i_train,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
    )
    return data_dict


