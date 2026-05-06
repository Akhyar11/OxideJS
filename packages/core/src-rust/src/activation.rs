use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn relu_native_into(input: Float32Array, mut out_res: Float32Array, mut out_grad: Float32Array) {
    for i in 0..input.len() {
        let val = input[i];
        if val > 0.0 {
            out_res[i] = val;
            out_grad[i] = 1.0;
        } else {
            out_res[i] = 0.0;
            out_grad[i] = 0.0;
        }
    }
}

#[napi]
pub fn sigmoid_native_into(input: Float32Array, mut out_res: Float32Array, mut out_grad: Float32Array) {
    for i in 0..input.len() {
        let val = 1.0 / (1.0 + (-input[i]).exp());
        out_res[i] = val;
        out_grad[i] = val * (1.0 - val);
    }
}

#[napi]
pub fn tanh_native_into(input: Float32Array, mut out_res: Float32Array, mut out_grad: Float32Array) {
    for i in 0..input.len() {
        let val = input[i].tanh();
        out_res[i] = val;
        out_grad[i] = 1.0 - val * val;
    }
}
