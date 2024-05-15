pub mod model;
pub mod inference;

use burn::{backend, prelude::*};
use burn_ndarray;
use inference::*;

use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, ImageData};

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub async fn draw(
    ctx: &CanvasRenderingContext2d,
    r: f64,
    phi: f64,
    theta: f64,
) -> Result<(), JsValue> {
    // The real workhorse of this algorithm, generating pixel data
    let data = render().await;
    let data = ImageData::new_with_u8_clamped_array_and_sh(Clamped(&data), 100, 100)?;
    ctx.put_image_data(&data, 0.0, 0.0)
}

async fn render() -> Vec<u8> {
    // backend and device
    type Backend = backend::NdArray;
    let device = &burn_ndarray::NdArrayDevice::default();
    //type Backend = backend::Wgpu;
    //let device = &backend::wgpu::WgpuDevice::default();
    
    // nelf inferencer
    let runner = NeLFInferencer::new();

    // consts
    let h: usize = 100;
    let w: usize = 100;
    let focal: f32 = 138.0;
    let emb_dim: usize = 6;
    let num_samples: usize = 16;
    let near = 2.;
    let far  = 6.;
    let pose: Tensor<Backend, 2, Float> = Tensor::from_floats([
        [-9.9990219e-01,  4.1922452e-03, -1.3345719e-02, -5.3798322e-02],
        [-1.3988681e-02, -2.9965907e-01,  9.5394367e-01,  3.8454704e+00],
        [-4.6566129e-10,  9.5403719e-01,  2.9968831e-01,  1.2080823e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
    ], device);

    let rays = get_rays(h, w, focal, pose, near, far, num_samples, emb_dim, device);
    //println!("{:#?}", rays.clone().slice([0..3, 0..4, 0..4]));
    let pred = runner.render_rays(rays) * 255;
    //println!("{:#?}", out.slice([0..3, 0..4, 0..4]));
    
    let data = pred.permute([1, 2, 0]).into_data().await;
    
    let mut image_data = vec![];
    for pixel in data.value.chunks(3) {
        image_data.push(pixel[0] as u8);
        image_data.push(pixel[1] as u8);
        image_data.push(pixel[2] as u8);
        image_data.push(255 as u8);
    }
    image_data
}
