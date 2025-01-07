import numpy as np
import os
import re

cam1_cube_info = np.load("pre_generated_data/cam1_cube_info.npz")
cam2_cube_info = np.load("pre_generated_data/cam2_cube_info.npz")

cam_dict = {}

for one_key in cam1_cube_info.files:
    if "init" in one_key:
        continue
    idx = re.search(r'^\d+', one_key).group(0)
    intrinsic = cam1_cube_info[f"{idx}_int"]
    cam2center = cam1_cube_info[f"{idx}_ext"]
    world_mat = intrinsic @ cam2center
    world_mat = world_mat.astype(np.float32)
    idx = int(idx)
    cam_dict['camera_mat_{}'.format(idx)] = intrinsic
    cam_dict['camera_mat_inv_{}'.format(idx)] = np.linalg.inv(intrinsic)
    cam_dict['world_mat_{}'.format(idx)] = world_mat
    cam_dict['world_mat_inv_{}'.format(idx)] = np.linalg.inv(world_mat)
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    cam_dict['scale_mat_{}'.format(idx)] = scale_mat
    cam_dict['scale_mat_inv_{}'.format(idx)] = np.linalg.inv(scale_mat)

np.savez(os.path.join('pre_generated_data/cameras_sphere_nonpick.npz'), **cam_dict)

pass_idx = ["109", "116", "117", "120"]
for one_key in cam2_cube_info.files:
    idx = re.search(r'^\d+', one_key).group(0)
    if idx in pass_idx:
        print(f"{idx} pass")
        continue
    try:
        intrinsic = cam2_cube_info[f"{idx}_int"]
        cam2center = cam2_cube_info[f"{idx}_ext"]
        if cam2center.shape == (3, 3):
            print(f"{idx} not detected")
            image_path = f"pre_generated_data/image_obj_cam2/{idx}.png"
            mask_path = f"pre_generated_data/mask/{idx}.png"
            image_rmbg_path = f"pre_generated_data/image_rmbg/{idx}.png"
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"{image_path} : removed")
            if os.path.exists(mask_path):
                os.remove(mask_path)
                print(f"{mask_path} : removed")
            if os.path.exists(image_rmbg_path):
                os.remove(image_rmbg_path)
                print(f"{image_rmbg_path} : removed")
            continue
        world_mat = intrinsic @ cam2center
        world_mat = world_mat.astype(np.float32)
        idx = int(idx)
        cam_dict['camera_mat_{}'.format(idx)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(idx)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(idx)] = world_mat
        cam_dict['world_mat_inv_{}'.format(idx)] = np.linalg.inv(world_mat)
        scale_mat = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        cam_dict['scale_mat_{}'.format(idx)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(idx)] = np.linalg.inv(scale_mat)
    except:
        print(f"{idx} not found. is detected?")

print(len(cam_dict.keys())/6)

np.savez(os.path.join('pre_generated_data/cameras_sphere_all.npz'), **cam_dict)