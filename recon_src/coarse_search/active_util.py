import torch
import numpy as np
import matplotlib as mpl


def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z])

def vector_to_rotation_matrix(vector, theta):
    vector = vector / torch.linalg.norm(vector)

    I = torch.eye(3)
    
    tensor_0 = torch.zeros(1).squeeze()
    
    K = torch.stack([
                    torch.stack([tensor_0, -vector[2], vector[1]]),
                    torch.stack([vector[2], tensor_0, -vector[0]]),
                    torch.stack([-vector[1], vector[0], tensor_0])])
    
    # Rodrigues' formula
    R = I + (torch.sin(theta) * K) + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R

def generate_pose_old(r, theta, phi):
    theta = theta.squeeze()
    phi = phi.squeeze()
    device = theta.device

    # obj_origin_point = torch.Tensor([0.025*11, 0.025*7, 0], device=device)
    point = spherical_to_cartesian(r, theta, phi)

    x, y, z = point
    axis = torch.tensor([0, 0, -1.], device=device)
    vector = torch.stack([-x, -y, -z])
    
    tensor_0 = torch.zeros(1, device=device).squeeze()
    tensor_1 = torch.ones(1, device=device).squeeze()

    rotation_axis = torch.cross(point, axis)
    cos_theta = torch.dot(-point, axis) / (torch.linalg.norm(point) * torch.linalg.norm(axis))
    rotation_theta = torch.arccos(cos_theta)
    
    rot_mat = vector_to_rotation_matrix(rotation_axis, rotation_theta)
    x_axis_flip = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, -tensor_1, tensor_0]),
        torch.stack([tensor_0, tensor_0, -tensor_1])
    ])
    z_axis_flip = torch.stack([
        torch.stack([-tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, -tensor_1, tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])
    ])
    rot_mat = torch.matmul(rot_mat, x_axis_flip)
    rot_mat = torch.matmul(rot_mat, z_axis_flip)

    # point = point + obj_origin_point
    tf_mat = torch.concat([rot_mat, point.reshape((3, 1))], dim=1)
    tf_mat = torch.concat([tf_mat, torch.stack([tensor_0, tensor_0, tensor_0, tensor_1]).reshape((1, 4))], dim=0)
    return tf_mat

def generate_pose(r, theta, phi):
    theta = theta.squeeze()
    phi = phi.squeeze()
    device = theta.device
    point = spherical_to_cartesian(r, theta, phi)
    x, y, z = point

    tensor_0 = torch.zeros(1, device=device).squeeze()
    tensor_1 = torch.ones(1, device=device).squeeze()

    rz = torch.stack([-x, -y, -z])
    rz = rz / torch.norm(rz)
    rx = torch.stack([-y, x, tensor_0])
    rx = rx / torch.norm(rx)
    ry = torch.cross(rz, rx)

    tf_mat = torch.stack([
        torch.stack([rx[0], ry[0], rz[0], x]),
        torch.stack([rx[1], ry[1], rz[1], y]),
        torch.stack([rx[2], ry[2], rz[2], z]),
        torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])
    ])
    return tf_mat

# def generate_pose2(r, theta, phi):
#     theta = theta.squeeze()
#     phi = phi.squeeze()
#     device = theta.device
#     point = spherical_to_cartesian(r, theta, phi)
#     x, y, z = point
#     rz = torch.tensor([-x, -y, -z])
#     rz = rz / torch.norm(rz)
#     rx = torch.tensor([-y, ])

def find_closest_pose(generated, candidates):
    err_container = []

    for one_candidate in candidates:
        pose_err = calc_pose_error(cur_pose=generated,
                                   tar_pose=one_candidate,
                                   EPS=1e-3)
        err_norm = torch.linalg.norm(pose_err)
        err_container.append(err_norm)
    
    min_err = min(err_container)
    closest_idx = err_container.index(min_err)
    closest_pose = candidates[closest_idx]
    return closest_pose, closest_idx, min_err


def calc_pose_error(cur_pose, tar_pose, EPS):
    pos_err = tar_pose[:3, -1] - cur_pose[:3, -1]
    rot_err = torch.matmul(cur_pose[:3, :3].T, tar_pose[:3, :3])
    w_err = torch.matmul(cur_pose[:3, :3], rot_to_omega(rot_err, EPS))
    return torch.vstack([pos_err.T, w_err])

def calc_pose_error_loss(cur_pose, tar_pose, EPS):
    pos_err = tar_pose[:3, -1] - cur_pose[:3, -1]
    rot_err = torch.matmul(cur_pose[:3, :3].T, tar_pose[:3, :3]) * 0.5
    w_err = torch.matmul(cur_pose[:3, :3], rot_to_omega(rot_err, EPS))
    return torch.vstack([pos_err.T, w_err])

def rot_to_omega(R, EPS):
    # referred p36
    el = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm_el = torch.linalg.norm(el)
    if norm_el > EPS:
        w = torch.atan2(norm_el, torch.trace(R) - 1) * el / norm_el
    elif R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0:
        w = torch.zeros((3, 1))
    else:
        w = (torch.pi / 2) * torch.stack([R[0, 0] + 1, R[1, 1] + 1, R[2, 2] + 1])
    
    return w

def init_opt_model(init_theta=None, init_phi=None, radius=None,
                       optimized_params=None):
    if optimized_params is not None:
        spherical_model = SphericalParameter(optimized_params=optimized_params)
    else:
        spherical_model = SphericalParameter(init_phi=init_phi, init_theta=init_theta)
    
    pose_optim = torch.optim.Adam([
        {'params': [spherical_model.theta], 'lr':0.01},
        {'params': [spherical_model.phi], 'lr':0.5}
    ])
    return [spherical_model, pose_optim, radius]


class SphericalParameter(torch.nn.Module):
    def __init__(self, init_phi=None, init_theta=None, optimized_params=None):
        super(SphericalParameter, self).__init__()
        if optimized_params is not None:
            self.theta = torch.nn.Parameter(torch.tensor([optimized_params[0]]))
            self.phi = torch.nn.Parameter(torch.tensor([optimized_params[1]]))
        else:
            self.theta = torch.nn.Parameter(init_theta)
            self.phi = torch.nn.Parameter(init_phi)
    
    def forward(self):
        return self.theta, self.phi

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


c1 = '#1f77b4' #blue
c2 = 'green' #green
n = 50
