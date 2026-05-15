use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use crate::math::{SafeRawPtr, SafeRawPtrMut};

const CONV_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi]
pub fn convolution_native_into(
    a_data: Float32Array,
    a_rows: u32,
    a_cols: u32,
    k_data: Float32Array,
    k_rows: u32,
    k_cols: u32,
    mut out: Float32Array,
) {
    let ac = a_cols as usize;
    let kr = k_rows as usize;
    let kc = k_cols as usize;
    let out_rows = (a_rows - k_rows + 1) as usize;
    let out_cols = (a_cols - k_cols + 1) as usize;
    let a_len = a_data.len();
    let k_len = k_data.len();
    let out_len = out.len();
    let a_p = SafeRawPtr(a_data.as_ptr() as usize);
    let k_p = SafeRawPtr(k_data.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    if out_len < CONV_PARALLEL_THRESHOLD {
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
    } else {
        (0..out_rows).into_par_iter().for_each(|i| {
            unsafe {
                let a_ptr = std::slice::from_raw_parts(a_p.0 as *const f32, a_len);
                let k_ptr = std::slice::from_raw_parts(k_p.0 as *const f32, k_len);
                let out_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, out_len);

                let r_offset = i * out_cols;
                for j in 0..out_cols {
                    let mut sum = 0.0;
                    for k in 0..kr {
                        let a_offset = (i + k) * ac + j;
                        let k_offset = k * kc;
                        for l in 0..kc {
                            sum += a_ptr[a_offset + l] * k_ptr[k_offset + l];
                        }
                    }
                    out_ptr[r_offset + j] = sum;
                }
            }
        });
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
    mut out: Float32Array,
) {
    let er = err_rows as usize;
    let ec = err_cols as usize;
    let ic = input_cols as usize;
    let oc = out_cols as usize;

    let err_len = err_data.len();
    let input_len = input_data.len();
    let out_len = out.len();
    let err_p = SafeRawPtr(err_data.as_ptr() as usize);
    let input_p = SafeRawPtr(input_data.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    let ir = input_rows as usize;
    let or = (er + ir - 1); // Approximate out rows for partitioning

    (0..or).into_par_iter().for_each(|m_plus_k| {
        unsafe {
            let err_ptr = std::slice::from_raw_parts(err_p.0 as *const f32, err_len);
            let input_ptr = std::slice::from_raw_parts(input_p.0 as *const f32, input_len);
            let out_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, out_len);

            // For each row in the output, find which combinations of (m, k) contribute to it
            // m + k = row_idx
            for k in 0..er {
                if m_plus_k < k { continue; }
                let m = m_plus_k - k;
                if m >= ir { continue; }

                let err_row = &err_ptr[k * ec..(k + 1) * ec];
                let input_row = &input_ptr[m * ic..(m + 1) * ic];
                let out_row = &mut out_ptr[m_plus_k * oc..(m_plus_k + 1) * oc];

                for l in 0..ec {
                    let err_val = err_row[l];
                    if err_val == 0.0 { continue; }
                    for n in 0..ic {
                        out_row[n + l] += err_val * input_row[n];
                    }
                }
            }
        }
    });
}
