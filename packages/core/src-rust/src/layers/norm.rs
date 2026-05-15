use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use crate::math::{SafeRawPtr, SafeRawPtrMut};

const LAYER_NORM_STATS_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi]
pub fn layer_norm_native_into(
    x_data: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    rows: u32,
    cols: u32,
    eps: f64,
    mut out_res: Float32Array,
    mut out_norm: Float32Array,
    mut out_means: Float32Array,
    mut out_stds: Float32Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    let eps_f32 = eps as f32;
    let x_len = x_data.len();
    let res_len = out_res.len();
    let norm_len = out_norm.len();
    let means_len = out_means.len();
    let stds_len = out_stds.len();

    let x_p = SafeRawPtr(x_data.as_ptr() as usize);
    let gamma_p = SafeRawPtr(gamma.as_ptr() as usize);
    let beta_p = SafeRawPtr(beta.as_ptr() as usize);
    let out_res_p = SafeRawPtrMut(out_res.as_ptr() as usize);
    let out_norm_p = SafeRawPtrMut(out_norm.as_ptr() as usize);
    let out_means_p = SafeRawPtrMut(out_means.as_ptr() as usize);
    let out_stds_p = SafeRawPtrMut(out_stds.as_ptr() as usize);

    let gamma_len = gamma.len();
    let beta_len = beta.len();

    // 1. Compute Statistics (Mean & Std) per column (units)
    (0..c).into_par_iter().for_each(|j| {
        unsafe {
            let x_ptr = std::slice::from_raw_parts(x_p.0 as *const f32, x_len);
            let m_ptr = std::slice::from_raw_parts_mut(out_means_p.0 as *mut f32, means_len);
            let s_ptr = std::slice::from_raw_parts_mut(out_stds_p.0 as *mut f32, stds_len);

            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for i in 0..r {
                let value = x_ptr[i * c + j];
                sum += value;
                sum_sq += value * value;
            }
            let mean = sum / (r as f32);
            let variance = (sum_sq / (r as f32) - mean * mean).max(0.0);
            m_ptr[j] = mean;
            s_ptr[j] = (variance + eps_f32).sqrt();
        }
    });

    // 2. Normalize and Apply Gamma/Beta
    (0..r).into_par_iter().for_each(|i| {
        unsafe {
            let x_ptr = std::slice::from_raw_parts(x_p.0 as *const f32, x_len);
            let g_ptr = std::slice::from_raw_parts(gamma_p.0 as *const f32, gamma_len);
            let b_ptr = std::slice::from_raw_parts(beta_p.0 as *const f32, beta_len);
            let m_ptr = std::slice::from_raw_parts(out_means_p.0 as *const f32, means_len);
            let s_ptr = std::slice::from_raw_parts(out_stds_p.0 as *const f32, stds_len);
            
            let res_ptr = std::slice::from_raw_parts_mut(out_res_p.0 as *mut f32, res_len);
            let norm_ptr = std::slice::from_raw_parts_mut(out_norm_p.0 as *mut f32, norm_len);

            let row_offset = i * c;
            let g = g_ptr[i];
            let b = b_ptr[i];

            for j in 0..c {
                let idx = row_offset + j;
                let norm = (x_ptr[idx] - m_ptr[j]) / s_ptr[j];
                norm_ptr[idx] = norm;
                res_ptr[idx] = norm * g + b;
            }
        }
    });
}

#[napi]
pub fn layer_norm_backward_native_into(
    err_data: Float32Array,
    norm_data: Float32Array,
    gamma_data: Float32Array,
    rows: u32,
    cols: u32,
    std_data: Float32Array,
    mut d_gamma_out: Float32Array,
    mut d_beta_out: Float32Array,
    mut dx_out: Float32Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    let err_slice = &*err_data;
    let norm_slice = &*norm_data;
    let gamma_slice = &*gamma_data;
    let std_slice = &*std_data;

    let d_gamma_slice = &mut *d_gamma_out;
    let d_beta_slice = &mut *d_beta_out;
    let dx_slice = &mut *dx_out;

    d_gamma_slice
        .par_iter_mut()
        .zip(d_beta_slice.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dg, db))| {
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let row_offset = i * c;
            for j in 0..c {
                let idx = row_offset + j;
                sum_g += err_slice[idx] * norm_slice[idx];
                sum_b += err_slice[idx];
            }
            *dg = sum_g;
            *db = sum_b;
        });

    let mut sum1_cols = vec![0.0f32; c];
    let mut sum2_cols = vec![0.0f32; c];
    sum1_cols
        .par_iter_mut()
        .zip(sum2_cols.par_iter_mut())
        .enumerate()
        .for_each(|(j, (sum1_out, sum2_out))| {
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            for i in 0..r {
                let idx = i * c + j;
                let e = err_slice[idx] * gamma_slice[i];
                sum1 += e;
                sum2 += e * norm_slice[idx];
            }
            *sum1_out = sum1;
            *sum2_out = sum2;
        });

    let inv_r = 1.0f32 / (r as f32);
    dx_slice
        .par_chunks_mut(c)
        .enumerate()
        .for_each(|(i, dx_row)| {
            let g = gamma_slice[i];
            let row_offset = i * c;
            for j in 0..c {
                let idx = row_offset + j;
                dx_row[j] = (g * err_slice[idx]
                    - (sum1_cols[j] * inv_r)
                    - (norm_slice[idx] * sum2_cols[j] * inv_r))
                    / std_slice[j];
            }
        });
}
