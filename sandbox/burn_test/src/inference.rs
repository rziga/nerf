use crate::model::nelf::Model;

use burn::{prelude::*, tensor::activation};

pub struct NeLFInferencer<B: Backend> {
    model: Model<B>,
}
impl<B: Backend> NeLFInferencer<B> {
    pub fn new() -> Self {
        let model = Model::<B>::default();
        Self{ model }
    }

    pub fn render_rays(&self, rays: Tensor<B, 3>) -> Tensor<B, 3> {
        let rays = rays.unsqueeze_dim(0);
        let out = self.model.forward(rays);
        activation::relu(out.squeeze(0))
    }
}

fn encode_pos<B: Backend, const D: usize>(x: Tensor<B, D>, emb_dim: usize) -> Tensor<B, D> {
    let mut ret = vec![x.clone()];
    for i in 0..emb_dim {
        let xpow2 = x.clone() * 2f32.powf(i as f32);
        ret.push(xpow2.clone().sin());
        ret.push(xpow2.clone().cos());
    }
    let emb = Tensor::cat(ret, 0);
    emb
}

fn get_ray_origin_dir<B: Backend>(h: usize, w: usize, focal: f32, pose: Tensor<B, 2>, device: &<B>::Device) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let coorsi: Tensor<B, 1> = Tensor::arange(0..(h as i64), device).float();
    let coorsj: Tensor<B, 2> = Tensor::arange(0..(w as i64), device).unsqueeze_dim(1).float();
    let i: Tensor<B, 2> = coorsi.expand([h, w]);
    let j: Tensor<B, 2> = coorsj.expand([h, w]);

    // dirs
    let dirs: Tensor<B, 3> = Tensor::stack(vec![
         (i.clone() - (h as f32)/2f32) / focal,
        -(j.clone() - (h as f32)/2f32) / focal,
        -i.ones_like(),
    ], 2);

    // pose rotation and location
    let rot = pose.clone().slice([0..3, 0..3]).transpose().unsqueeze_dim(0);
    let loc = pose.clone().slice([0..3, 3..4]).reshape([3, 1, 1]);

    // ray dir and origin
    let ray_dirs = dirs.matmul(rot).permute([2, 0, 1]);
    let ray_origin = loc;

    (ray_origin, ray_dirs)
}

pub fn get_rays<B: Backend>(h: usize, w: usize, focal: f32, pose: Tensor<B, 2>, near: f32, far: f32, num_samples: usize, emb_dim: usize, device: &<B>::Device) -> Tensor<B, 3>{
    let (ray_origin, ray_dirs) = get_ray_origin_dir(h, w, focal, pose, device);
    let depths = linspace::<B>(near, far, num_samples, device);
    let rays = ray_origin.unsqueeze_dim(0) + ray_dirs.unsqueeze_dim(0) * depths.reshape([-1i32, 1, 1, 1]);
    let rays = rays.reshape([-1i32, h as i32, w as i32]);
    let rays = encode_pos(rays, emb_dim);
    rays
}

pub fn linspace<B: Backend>(start: f32, stop: f32, num_samples: usize, device: &<B>::Device) -> Tensor<B, 1> {
    let n = num_samples as i64;
    let points = Tensor::arange(0..n, device).float();
    let len = stop - start;
    let linspace = points / (n-1) * len + start;
    linspace
}