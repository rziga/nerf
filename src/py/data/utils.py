import torch

def normalize(vec, axis):
    return vec / (vec.norm(dim=axis) + 1e-10)

def look_at(centers):
    forward_vec = normalize(centers, axis=-1)
    up_vec = torch.tensor([[0, 0, -1.]])
    right_vec = normalize(torch.cross(forward_vec, up_vec, -1), -1)
    up_vec = normalize(torch.cross(forward_vec, right_vec, -1), -1)

    poses = torch.eye(4, dtype=torch.float32)
    poses[:3, :3] = torch.stack([right_vec, up_vec, forward_vec], -1)
    poses[:3, -1] = centers
    return poses

def get_c2w(center):
    cam_pos = (center * (4.031128857175551 / 1.5)).unsqueeze(0)
    return look_at(cam_pos)

def get_ray_origin_dir(H, W, focal, pose):
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy")
    dirs = torch.stack([
        +(i - H/2) / focal,
        -(j - W/2) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    ray_dirs = (dirs @ pose[:3, :3].T).permute(2, 0, 1)
    ray_origin = pose[:3, -1].view(3, 1, 1)
    return ray_origin, ray_dirs

def get_rays(H, W, focal, pose, near, far, num_samples, rand):
    rayo, rayd = get_ray_origin_dir(H, W, focal, pose)
    depths = torch.linspace(near, far, num_samples).view(num_samples, 1, 1, 1)
    if rand:
        noise = torch.rand([num_samples, 1, H, W])
        depths = depths + noise
    rays = rayo + rayd * depths
    rays = rays.reshape(-1, H, W)
    return rays