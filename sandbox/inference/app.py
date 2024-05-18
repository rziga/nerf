from flask import Flask, render_template, send_file
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from math import sin, cos, radians

app = Flask(__name__, static_folder="static")

class ResNetBlock(nn.Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        skip = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + skip

class SuperResBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, num_convs):
        layers = [
            #nn.Upsample(scale_factor=2),
            #nn.Conv2d(in_channels, out_channels, 1)
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        ]
        for _ in range(num_convs):
            layers.append(ResNetBlock(out_channels, 1))
        super().__init__(*layers)

class MobileR2L(nn.Sequential):

    def __init__(self, in_channels, hidden_channels, num_layers, num_sr_modules):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1), nn.ReLU()
        )
        self.body = nn.Sequential(*[
            ResNetBlock(hidden_channels, 1)
            for _ in range(num_layers)
        ])
        self.upscale = nn.Sequential(*[
            SuperResBlock(hidden_channels//(2**i), hidden_channels//(2**(i+1)), 2)
            for i in range(num_sr_modules)
        ])
        self.tail = nn.Conv2d(hidden_channels//(2**num_sr_modules), 3, 1)
    
    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.upscale(x)
        return self.tail(x)
    
def encode_pos(x, emb_dim):
    rets = [x]
    for i in range(emb_dim):
        rets.append(torch.sin(2.**i * x))
        rets.append(torch.cos(2.**i * x))
    return torch.cat(rets, dim=0)

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

def get_rays(H, W, focal, pose, near, far, num_samples, emb_dim):
    rayo, rayd = get_ray_origin_dir(H, W, focal, pose)
    depths = torch.linspace(near, far, num_samples).view(-1, 1, 1)
    rays = rayo + (rayd.unsqueeze(0) * depths.view(-1, 1, 1, 1))
    rays = rays.reshape(-1, H, W)
    return encode_pos(rays, emb_dim)

def view_gen():
    H, W, focal = 100, 100, 138
    near, far = 2, 6
    num_samples = 16
    emb_dim = 6

    tr_r = lambda r : torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,r],
        [0,0,0,1.],
    ])

    rot_phi = lambda phi: torch.tensor([
        [1,0,0,0],
        [0,cos(phi),-sin(phi),0],
        [0,sin(phi), cos(phi),0],
        [0,0,0,1.],
    ])

    rot_theta = lambda th : torch.tensor([
        [cos(th),0,-sin(th),0],
        [0,1,0,0],
        [sin(th),0, cos(th),0],
        [0,0,0,1.],
    ])

    def get_pose_spherical(r, phi, theta):
        perm = torch.tensor([[-1.,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        return perm @ rot_theta(radians(theta)) @ rot_phi(radians(phi)) @ tr_r(r)

    th = 0
    while True:
        pose = get_pose_spherical(4,-30, th)
        rays = get_rays(H, W, focal, pose, near, far, num_samples, emb_dim).cuda()
        with torch.no_grad():
            pred_img = F.sigmoid(model(rays.unsqueeze(0))).squeeze(0).permute(1, 2, 0)
        yield (pred_img * 255).to(torch.uint8).cpu()
        th += 1


model = torch.load("../tiny_nelf.pt").cuda()
gen = view_gen()

@app.route("/image", methods=["GET"])
def render_viewframe():
    pred_img = next(gen)
    img = Image.fromarray(pred_img.numpy())

    io = BytesIO()
    img.save(io, "PNG")
    io.seek(0)
    return send_file(io, mimetype="image/png")

@app.route("/")
def main():
    return render_template("index.html")

if __name__ == "__main__":  
    app.run()
