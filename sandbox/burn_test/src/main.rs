use burn::{backend, prelude::*};
use burn_ndarray;

use burn_test::inference::*;

use yew::prelude::*;

#[function_component]
fn App() -> Html {
    let counter = use_state(|| 0);
    let onclick = {
        let counter = counter.clone();
        move |_| {
            let value = *counter + 1;
            counter.set(value);
        }
    };

    html! {
        <div>
            <button {onclick}>{ "+1" }</button>
            <p>{ *counter }</p>
        </div>
    }
}

fn main() {
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
    println!("{:#?}", rays.clone().slice([0..3, 0..4, 0..4]));
    let out = runner.render_rays(rays);
    println!("{:#?}", out.slice([0..3, 0..4, 0..4]));
}