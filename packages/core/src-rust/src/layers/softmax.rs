use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn softmax_native_into(data: Float32Array, rows: u32, cols: u32, is_row: bool, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    out.copy_from_slice(&data);

    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..c { if out[offset + j] > max_val { max_val = out[offset + j]; } }
            if !max_val.is_finite() {
                let uniform = 1.0 / c as f32;
                for j in 0..c { out[offset + j] = uniform; }
                continue;
            }
            let mut sum_exp = 0.0f32;
            for j in 0..c {
                let exp_val = (out[offset + j] - max_val).exp();
                out[offset + j] = exp_val;
                sum_exp += exp_val;
            }
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                let uniform = 1.0 / c as f32;
                for j in 0..c { out[offset + j] = uniform; }
                continue;
            }
            for j in 0..c { out[offset + j] /= sum_exp; }
        }
    } else {
        for j in 0..c {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..r { if out[i * c + j] > max_val { max_val = out[i * c + j]; } }
            if !max_val.is_finite() {
                let uniform = 1.0 / r as f32;
                for i in 0..r { out[i * c + j] = uniform; }
                continue;
            }
            let mut sum_exp = 0.0f32;
            for i in 0..r {
                let idx = i * c + j;
                let exp_val = (out[idx] - max_val).exp();
                out[idx] = exp_val;
                sum_exp += exp_val;
            }
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                let uniform = 1.0 / r as f32;
                for i in 0..r { out[i * c + j] = uniform; }
                continue;
            }
            for i in 0..r { out[i * c + j] /= sum_exp; }
        }
    }
}

#[napi]
pub fn softmax_backward_native_into(s_data: Float32Array, g_data: Float32Array, rows: u32, cols: u32, is_row: bool, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut sum_grad_s = 0.0;
            for j in 0..c { sum_grad_s += s_data[offset + j] * g_data[offset + j]; }
            for j in 0..c {
                let idx = offset + j;
                out[idx] = s_data[idx] * (g_data[idx] - sum_grad_s);
            }
        }
    } else {
        for j in 0..c {
            let mut sum_grad_s = 0.0;
            for i in 0..r { sum_grad_s += s_data[i * c + j] * g_data[i * c + j]; }
            for i in 0..r {
                let idx = i * c + j;
                out[idx] = s_data[idx] * (g_data[idx] - sum_grad_s);
            }
        }
    }
}
