import numpy as np
import os

from .load_blender_real import load_blender_data
from .load_dtu import load_dtu_data

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far



import numpy as np
import os

from .load_blender import load_blender_data
from .load_dtu import load_dtu_data

def load_data(args, reso_level=1, train_all=True, wmask=True, white_bg=True):
    print("[ resolution level {} | train all {} | wmask {} | white_bg {}]".format(reso_level, train_all, wmask, white_bg))
    K, depths = None, None
    scale_mats_np = None
    masks = None

    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, masks = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'dtu':
        images, poses, render_poses, hwf, K, i_split, scale_mats_np, masks = load_dtu_data(args.datadir, reso_level=reso_level, mask=wmask, white_bg=white_bg)
        print('Loaded dtu', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

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
    print("Split: train {} | validate {} | test {}".format(
        len(i_train), len(i_val), len(i_test)))
    print('near, far: ', near, far)
    if wmask and masks is None:
        masks = images.mean(-1) > 0


    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        scale_mats_np=scale_mats_np,
        masks=masks
    )
    return data_dict