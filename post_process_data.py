import numpy as np
import os
import re
import glob
import shutil


nonpick = np.load("generated_data/nonpick_search.npy")
pick = np.load("generated_data/pick_search.npy")

image_nonpick_path = sorted(glob.glob("generated_data/image/*.png"))
image_pick_path = sorted(glob.glob("generated_data/image_2/*.png"))

mask_nonpick_path = sorted(glob.glob("generated_data/mask/*.png"))
mask_pick_path = sorted(glob.glob("generated_data/mask_2/*.png"))

print("nonpick shape:", nonpick.shape)
print("pick shape:   ", pick.shape)

print("Total extrinsics:", len(nonpick) + len(pick))

# For your hand eye camera intrinsic
K_1 = np.diag([1.0, 1.0, 1.0, 1.0])
intrinsic_cam1 = np.array([602.7904663085938, 0.0, 330.9473876953125,
                           0.0, 602.8036499023438, 246.41323852539062,
                           0.0, 0.0, 1.0]).reshape((3, 3))
K_1[:3, :3] = intrinsic_cam1

# For your extrinsic fixed camera intrinsic
K_2 = np.diag([1.0, 1.0, 1.0, 1.0])
intrinsic_cam2 = np.array([607.0380859375, 0.0, 313.9091796875,
                           0.0, 606.7507934570312, 256.9510803222656,
                           0.0, 0.0, 1.0]).reshape((3, 3))
K_2[:3, :3] = intrinsic_cam2

single_folder = "STG_data"
os.makedirs(os.path.join(single_folder, "image"), exist_ok=True)
os.makedirs(os.path.join(single_folder, "mask"), exist_ok=True)

cam_dict = {}

# ==============
# 1) nonpick
# ==============
for i, pose in enumerate(nonpick):
    idx_str = re.search(r'\d+', image_nonpick_path[i]).group(0)
    idx = int(idx_str)
    
    shutil.copy(image_nonpick_path[i],
                os.path.join(single_folder, "image", f"{idx_str}.png"))
    shutil.copy(mask_nonpick_path[i],
                os.path.join(single_folder, "mask", f"{idx_str}.png"))
    
    cam2center = np.linalg.inv(pose)
    world_mat = K_1 @ cam2center
    world_mat = world_mat.astype(np.float32)
    
    cam_dict[f'camera_mat_{idx}'] = K_1
    cam_dict[f'camera_mat_inv_{idx}'] = np.linalg.inv(K_1)
    cam_dict[f'world_mat_{idx}'] = world_mat
    cam_dict[f'world_mat_inv_{idx}'] = np.linalg.inv(world_mat)
    
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    cam_dict[f'scale_mat_{idx}'] = scale_mat
    cam_dict[f'scale_mat_inv_{idx}'] = np.linalg.inv(scale_mat)

for i, pose in enumerate(pick):
    idx_str = re.search(r'image_2/(\d+)', image_pick_path[i]).group(1)
    idx = int(idx_str)
    
    shutil.copy(image_pick_path[i],
                os.path.join(single_folder, "image", f"{idx_str}.png"))
    shutil.copy(mask_pick_path[i],
                os.path.join(single_folder, "mask", f"{idx_str}.png"))
    
    cam2center = np.linalg.inv(pose)
    world_mat = K_2 @ cam2center
    world_mat = world_mat.astype(np.float32)
    
    cam_dict[f'camera_mat_{idx}'] = K_2
    cam_dict[f'camera_mat_inv_{idx}'] = np.linalg.inv(K_2)
    cam_dict[f'world_mat_{idx}'] = world_mat
    cam_dict[f'world_mat_inv_{idx}'] = np.linalg.inv(world_mat)
    
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    cam_dict[f'scale_mat_{idx}'] = scale_mat
    cam_dict[f'scale_mat_inv_{idx}'] = np.linalg.inv(scale_mat)

np.savez(os.path.join(single_folder, 'cameras_sphere.npz'), **cam_dict)

destination_folder = f"recon_src/Voxurf/data/STG_data"
os.makedirs(destination_folder, exist_ok=True)

for item in os.listdir(single_folder):
    source_path = os.path.join(single_folder, item)
    destination_path = os.path.join(destination_folder, item)

    if os.path.isdir(source_path):
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    else:
        shutil.copy2(source_path, destination_path)
