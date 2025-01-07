from tqdm import tqdm, trange
from ultralytics import YOLO
import os, sys, copy, glob, json, time, random, argparse, gc
import matplotlib.pyplot as plt
import mmengine
import imageio
import numpy as np
import cv2
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import tf
import rospy
import moveit_commander
from fr_msgs.srv import JointSer
from fr_msgs.srv import EmptySer
from std_srvs.srv import Empty

from recon_src.coarse_search.lib import utils
from recon_src.coarse_search.lib import dvgo_cam2 as dvgo
from recon_src.coarse_search.lib.load_data import load_data
from control_src.image_saver import ImageSaver
from sam_util.generate_mask import SAMMaskGenerator
import recon_src.coarse_search.active_util as autil

import warnings
warnings.filterwarnings(action='ignore')


torch_uniform_sample = lambda b1, b2: (b2 - b1) * torch.rand(1) + b1


def raw_image_process(current_image,
                      yolo_model,
                      sam_generator):
    """
    Post processing raw image

    Args
        - current_image : current image source (H, W, 3)
        - yolo_model : object detection using YOLO model
        - sam_generator : sementation class
        - reso_level : to compress resolution level 
    
    Return
        - processed_image : post processed image
        - processed_mask : post processed segmentation mask
        - masked_image_reso : compressed (H/reso, W/reso, 3)
        - processed_mask_reso : compressed (H/reso, W/reso, 1)
        - pose : detedcted camera extrinsic
    """
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) # yolo 들어갈때 BGR로 변환되는데 왜그런지 모르겠음

    # object detection using YOLO
    print("[YOLO] Detecting ...")
    detection_result = yolo_model.predict(current_image, device="cuda:0", verbose=False)
    bbox_xyxy = detection_result[0].boxes[0].xyxy.cpu().numpy().astype(int).squeeze()

    # get segmentation mask
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) # BGR 다시 RGB 변환
    processed_img, processed_mask, original_img = sam_generator.generate_mask(current_image,
                                                                              bbox=bbox_xyxy)
    processed_mask = (processed_mask / 255.)

    # type casting
    current_image = np.array(current_image / 255.).astype(np.float32)
    processed_img = np.array(processed_img / 255.).astype(np.float32)
    masked_img = current_image * processed_mask[..., None] # apply mask
    return masked_img, processed_mask, original_img


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_coarse", action='store_true')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    disps = []
    entropys = []
    mean_entropys = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp', 'entropy']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(32, 0), rays_d.split(32, 0), viewdirs.split(32, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()
        entropy = render_result['entropy'].cpu().numpy().reshape((H, W))
        mean_entropy = np.mean(entropy)
        
        rgbs.append(rgb)
        disps.append(disp)
        entropys.append(entropy)
        mean_entropys.append(mean_entropy)


        if i==0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))


    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        '''
        print('Testing psnr', [f'{p:.3f}' for p in psnrs])
        if eval_ssim: print('Testing ssim', [f'{p:.3f}' for p in ssims])
        if eval_lpips_vgg: print('Testing lpips (vgg)', [f'{p:.3f}' for p in lpips_vgg])
        if eval_lpips_alex: print('Testing lpips (alex)', [f'{p:.3f}' for p in lpips_alex])
        '''
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    return rgbs, disps, np.mean(psnrs)


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    seed_num = args.seed
    return seed_num

def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)
    
    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'i_train',
            'irregular_shape', 'poses', 'render_poses', 'images', 'masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)
    data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

@torch.no_grad()
def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # initialize ROS sevices
    go_to_joint_target = rospy.ServiceProxy('/FR_Robot/go_to_joint_target',
                                            JointSer)
    go_to_look_obj = rospy.ServiceProxy('/FR_Robot/look_obj',
                                        EmptySer)
    add_table = rospy.ServiceProxy('/FR_Robot/add_table', Empty)
    remove_table = rospy.ServiceProxy('/FR_Robot/remove_table', Empty)
    resp = add_table()

    # get pre-generated apriltag cube information (Stage 2)
    cam2_joint_info = np.load("pre_generated_data/cam2_joint_info.npz")
    cam2_cube_info = np.load("pre_generated_data/cam2_cube_info.npz")

    # initialize object detection model (YOLO)
    yolo_model = YOLO("sam_util/runs_bulldozer_gripper/detect/train/weights/best.pt")

    # initialize segmentation model (SAM)    
    sam_model = "vit_l_hq"
    is_white = False
    ckpt_path = "sam_util/sam_hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    sam_generator = SAMMaskGenerator(sam_model, ckpt_path, is_white)

    # initialize CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    HW, Ks, near, far, i_train, poses, render_poses, images, masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'poses', 'render_poses', 'images', 'masks'
        ]
    ]
    images_np = images.detach().clone().cpu().numpy()
    original_i_train = i_train
    original_train_poses = poses[original_i_train] # get original poses for debug

    # get pre-generated image in stage 2
    pre_image_path = sorted(glob.glob("generated_data/image/*.png"))
    pre_mask_path = sorted(glob.glob("generated_data/mask/*.png"))
    pre_pose = np.load("generated_data/nonpick_search.npy")
    pre_image_lst = []
    pre_mask_lst = []
    for one_image_path, one_mask_path in zip(pre_image_path, pre_mask_path):
        one_image = imageio.imread(one_image_path)
        one_mask = imageio.imread(one_mask_path)
        pre_image_lst.append(one_image)
        pre_mask_lst.append(one_mask[..., None])
    pre_image_np = (np.array(pre_image_lst) / 255.).astype(np.float32)
    pre_mask_np = (np.array(pre_mask_lst) / 255.).astype(np.float32)
    pre_image_np = pre_image_np * pre_mask_np  
    i_train = np.arange(pre_image_np.shape[0]).tolist()
    images = torch.from_numpy(pre_image_np).float()
    poses = torch.from_numpy(pre_pose).float().to(device)
    pre_i_train_len = len(i_train)
    HW = HW[:len(i_train)]
    Ks = Ks[:len(i_train)]
    image_saver = ImageSaver()
    cam2_intrinsic = image_saver.get_intrinsic("camera2")

    print(f"[Stage 2] get pre dataset in stage 1 : {i_train}")
    print(f"[Stage 2] get pre dataset in stage 1 : {images.shape}")
    print(f"[Stage 2] get pre dataset in stage 1 : {poses.shape}")

    # rescale object bbox
    print(f"{xyz_min} // {xyz_max}")
    xyz_min = xyz_min - 0.01
    xyz_max = xyz_max + 0.01
    print(f"{xyz_min} // {xyz_max}")
        
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    # find whether there is existing checkpoint path
    reload_ckpt_path = None

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) and reload_ckpt_path is None:
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    model = dvgo.DirectVoxGO(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        mask_cache_path=coarse_ckpt_path,
        **model_kwargs)
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)
    
    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # load checkpoint if there is
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    
    # init batch rays sampler
    def gather_training_rays():
        rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
            rgb_tr=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW, Ks=Ks, ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler
        
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()
    
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    ##############################
    # Active learning parameters #
    ##############################
    active_learning = True
    if active_learning:
        theta_bound = [torch.pi/2 - torch.pi/6, torch.pi]
        phi_bound = [0.0, 2*torch.pi]

        alpha_grid_init = model.activate_density(model.density, interval=0.5).clone().detach()    
        radius = 0.35
        
        pose_param = autil.init_opt_model(
            init_theta=torch_uniform_sample(theta_bound[0], theta_bound[1]),
            init_phi=torch_uniform_sample(phi_bound[0], phi_bound[1]),
            radius=radius)
        
        active_add_step = 2000
        opt_step = 10
        render_kwargs.update({
            "HW" : HW[0],
            "K" : cam2_intrinsic
        })
        active_cnt = 0
        pose_to_out = []
    
    original_img_path = os.path.join("generated_data", "image_2")
    mask_path = os.path.join("generated_data", "mask_2")
    rmbg_path = os.path.join("generated_data", "image_rmbg_2")
    
    os.makedirs(original_img_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(rmbg_path, exist_ok=True)
    ##############################
    for global_step in trange(1+start, 1+cfg_train.N_iters):
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            model.scale_volume_grid(model.num_voxels * 2)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.density.data.sub_(1)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()).item()
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.weight_tv_density>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_density * model.density_total_variation()
        if cfg_train.weight_tv_k0>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_k0 * model.k0_total_variation()
        loss.backward()
        optimizer.step()
        psnr_lst.append(psnr)

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor
        
        # synchronize uncertainty volume
        alpha_grid_after = render_result['alpha_grid']
        changed_indices = (alpha_grid_init != alpha_grid_after)
        model.synchronize_entropy_volume(changed_indices, **render_kwargs)


        ####################
        # Active inference #
        ####################
        if active_learning:
            if global_step % active_add_step == 0 and global_step != 0:
                model.entropy_inferencing = True

                if active_cnt != 0:                    
                    pose_param = autil.init_opt_model(
                                init_theta=torch_uniform_sample(theta_bound[0], theta_bound[1]),
                                init_phi=torch_uniform_sample(phi_bound[0], phi_bound[1]),
                                radius=radius)
                alpha_grid_init = model.activate_density(model.density, interval=0.5).clone().detach()

                # get NBV pose
                current_train_poses = poses[i_train]
                cube_closest_key, optimized_params = active_add_dataset_model(original_train_poses,
                                                                              model,
                                                                              pose_param,
                                                                              opt_step,
                                                                              render_kwargs,
                                                                              global_step,
                                                                              current_train_poses,
                                                                              cam2_cube_info,
                                                                              theta_bound,
                                                                              phi_bound) # 동작 확인 : cam2 closest key 받아오기
                # match NBV pose with apriltag cube information
                idx_name = re.search(r'^\d+', cube_closest_key).group(0)
                to_go_joint_key = f"{idx_name}_joint"
                to_go_joint_value = cam2_joint_info[to_go_joint_key]

                # control robot to NBV pose
                msg_joint_goal = JointSer._request_class()
                msg_joint_goal.joint_angles = to_go_joint_value
                resp = go_to_joint_target(msg_joint_goal)
                reached = resp.success

                if reached:
                    # capture image and pose processing (external fixed camera)
                    image_saver = ImageSaver()
                    current_image = image_saver.get_current_image(camera='camera2')    
                    masked_img, processed_mask, original_img = raw_image_process(current_image,
                                                                                yolo_model,
                                                                                sam_generator)
                    pose_to_add = np.linalg.inv(cam2_cube_info[cube_closest_key])
                    pose_to_out.append(pose_to_add)

                    # actively add to current train dataset
                    to_add_index = active_cnt + pre_i_train_len
                    i_train.append(to_add_index)
                    pose_to_add = torch.from_numpy(pose_to_add).float().to(device)
                    poses = torch.vstack([poses, pose_to_add[None,...]])
                    masked_img_torch = torch.from_numpy(masked_img).float()
                    images = torch.vstack([images, masked_img_torch[None,...]])
                    Ks = np.vstack([Ks, cam2_intrinsic[np.newaxis, :]])
                    
                    # get pose param
                    if active_cnt != 0:
                        pose_param = autil.init_opt_model(
                                init_theta=torch_uniform_sample(theta_bound[0], theta_bound[1]),
                                init_phi=torch_uniform_sample(phi_bound[0], phi_bound[1]),
                                radius=radius)
                        
                    # re generate training rays to active learning
                    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
                    print(f"[ACTIVE] all image indicies : {rays_o_tr.shape[0]}")
                    print(f"[ACTIVE] i_train : {i_train}")
                    
                    # save images to perform fine optimization
                    if to_add_index < 100:
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(mask_path, f"0{to_add_index}.png"), processed_mask * 255)
                        cv2.imwrite(os.path.join(original_img_path, f"0{to_add_index}.png"), current_image)
                        plt.imsave(os.path.join(rmbg_path, f"0{to_add_index}.png"), masked_img)
                    else:
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(mask_path, f"{to_add_index}.png"), processed_mask * 255)
                        cv2.imwrite(os.path.join(original_img_path, f"{to_add_index}.png"), current_image)
                        plt.imsave(os.path.join(rmbg_path, f"{to_add_index}.png"), masked_img)
                    
                    active_cnt = active_cnt + 1

                else:
                    print("control failed")
                    pass
                
                # re initialize model (to avoid overfitting issue)
                model.entropy_inferencing = False
                model = dvgo.DirectVoxGO(
                    xyz_min=xyz_min, xyz_max=xyz_max,
                    num_voxels=num_voxels,
                    mask_cache_path=coarse_ckpt_path,
                    **model_kwargs)
                model = model.to(device)
                optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []
    
    # Stage 2 end => save data
    pose_to_out = np.array(pose_to_out)
    np.save("generated_data/pick_search.npy", pose_to_out)

    # reset robot configuration
    resp = go_to_look_obj()
    resp = remove_table()

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

def active_add_dataset_model(origianal_train_poses,
                             model,
                             pose_param,
                             opt_step,
                             render_kwargs,
                             global_step,
                             train_poses,
                             cam2_cube_info,
                             theta_bound,
                             phi_bound):
    """
    Input : data_dict, model, pose_param
    Output : most uncertain datapoint index
    """
    spherical_model, pose_optim, radius = pose_param
    pose_to_save = []

    for i in range(opt_step):
        theta, phi = spherical_model()
        theta = theta + torch.randn(1) * 1e-3
        phi = phi + torch.randn(1) * 1e-3
        generated_pose = autil.generate_pose(radius, theta, phi)

        # generate ray
        rays_o_unc, rays_d_unc, viewdirs_unc = dvgo.get_rays_of_a_view(H=render_kwargs["HW"][0],
                                                                       W=render_kwargs["HW"][1],
                                                                       K=render_kwargs["K"],
                                                                       c2w=generated_pose,
                                                                       ndc=False,
                                                                       inverse_y=render_kwargs["inverse_y"],
                                                                       flip_x=render_kwargs["flip_x"],
                                                                       flip_y=render_kwargs["flip_y"])
        keys = ['entropy']        
        render_result_chunks_unc = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o_unc.split(16, 0), rays_d_unc.split(16, 0), viewdirs_unc.split(16, 0))
        ]
        render_result_unc = {
            k: torch.cat([ret[k] for ret in render_result_chunks_unc])
            for k in render_result_chunks_unc[0].keys()
        }
        
        # calculate 2D uncertainty loss
        current_view_entropy = render_result_unc['entropy']
        entropy_loss = torch.mean(current_view_entropy)

        # calculate fvs loss
        fvs_loss = 0
        for one_train_pose in train_poses:
            one_pose_err = autil.calc_pose_error_loss(generated_pose, one_train_pose, EPS=1e-3)
            one_pose_err_norm = torch.linalg.norm(one_pose_err)
            fvs_loss += (one_pose_err_norm / len(train_poses))
        fvs_loss = fvs_loss * 0.5

        uncertainty_loss = - entropy_loss - fvs_loss

        print(f"[ACTIVE] uncertainty loss     : {-uncertainty_loss}")
        print(f"\t[ACTIVE] entropy loss   : {entropy_loss.item():.4f}")
        print(f"\t[ACTIVE] fvs loss       : {fvs_loss}")
        print(f"\t[ACTIVE] params theta   : {theta.item():.4f}")
        print(f"\t[ACTIVE] params phi     : {phi.item():.4f}")
        print()
        
        pose_optim.zero_grad()
        uncertainty_loss.backward()
        pose_optim.step()
        
        with torch.no_grad():
            theta, phi = spherical_model()
            theta.clamp_(theta_bound[0], theta_bound[1]) 
            phi.clamp_(phi_bound[0], phi_bound[1])
    
        pose_to_save.append(generated_pose.clone().detach().cpu().numpy())

    # find closest cube info
    with torch.no_grad():
        cube_keys = cam2_cube_info.files
        cube_key_temp = []
        cube_pose_lst = []
        optimized_params = [theta.item(), phi.item()]
        generated_pose = autil.generate_pose(radius,
                                             torch.tensor(optimized_params[0]),
                                             torch.tensor(optimized_params[1]))

        for one_key in cube_keys:
            if "ext" in one_key and "init" not in one_key:
                if cam2_cube_info[one_key].shape == (4, 4):
                    cube_key_temp.append(one_key)
                    cube_pose_lst.append(np.linalg.inv(cam2_cube_info[one_key]))
        cube_pose_np = np.array(cube_pose_lst)
        cube_pose_torch = torch.from_numpy(cube_pose_np).float().to("cuda:0")
        cube_closest_pose, cube_closest_idx, err = autil.find_closest_pose(generated=generated_pose,
                                                                           candidates=cube_pose_torch)
        closest_key = cube_key_temp[cube_closest_idx]
        print(f"[ACTIVE] cube closest key : {closest_key}, err : {err}")
    return closest_key, optimized_params


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
            xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
            data_dict=data_dict, stage='coarse')
    eps_coarse = time.time() - eps_coarse
    eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
    print('train: coarse geometry searching in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':
    rospy.init_node("coarse_search")

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmengine.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        gpu_num = 0
        device = torch.device(f'cuda:{gpu_num}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    seed_num = seed_everything()
    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)


    # export scene bbox and camera poses in 3d for debugging and visualization
    export_bbox_and_cams_only = args.export_bbox_and_cams_only
    if export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

    # render trainset and eval
    if args.render_train:
        stepsize = cfg.coarse_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    print('Done')

