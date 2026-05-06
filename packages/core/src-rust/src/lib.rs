use napi_derive::napi;

pub mod math;
pub mod activation;
pub mod optimizer;
pub mod loss;
pub mod layers;

#[napi]
pub fn clip_gradients_native(mut data: napi::bindgen_prelude::Float32Array, limit: f64) {
    let limit = limit as f32;
    for i in 0..data.len() {
        if data[i] > limit {
            data[i] = limit;
        } else if data[i] < -limit {
            data[i] = -limit;
        }
    }
}
