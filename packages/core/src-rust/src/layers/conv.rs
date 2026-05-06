use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn convolution_native_into(
    a_data: Float32Array,
    a_rows: u32,
    a_cols: u32,
    k_data: Float32Array,
    k_rows: u32,
    k_cols: u32,
    mut out: Float32Array
) {
    let ac = a_cols as usize;
    let kr = k_rows as usize;
    let kc = k_cols as usize;
    let out_rows = (a_rows - k_rows + 1) as usize;
    let out_cols = (a_cols - k_cols + 1) as usize;

    for i in 0..out_rows {
        let r_offset = i * out_cols;
        for j in 0..out_cols {
            let mut sum = 0.0;
            for k in 0..kr {
                let a_offset = (i + k) * ac + j;
                let k_offset = k * kc;
                for l in 0..kc {
                    sum += a_data[a_offset + l] * k_data[k_offset + l];
                }
            }
            out[r_offset + j] = sum;
        }
    }
}

#[napi]
pub fn conv_backward_input_native_into(
    err_data: Float32Array,
    err_rows: u32,
    err_cols: u32,
    input_data: Float32Array,
    input_rows: u32,
    input_cols: u32,
    _out_rows: u32, 
    out_cols: u32,
    mut out: Float32Array
) {
    let er = err_rows as usize;
    let ec = err_cols as usize;
    let ic = input_cols as usize;
    let oc = out_cols as usize;

    for k in 0..er {
        for l in 0..ec {
            let err_val = err_data[k * ec + l];
            if err_val == 0.0 { continue; }
            for m in 0..input_rows as usize {
                for n in 0..ic {
                    out[(m + k) * oc + (n + l)] += err_val * input_data[m * ic + n];
                }
            }
        }
    }
}
