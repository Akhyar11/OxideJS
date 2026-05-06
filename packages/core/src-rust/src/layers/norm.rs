use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

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
    let x_slice = &*x_data;
    let gamma_slice = &*gamma;
    let beta_slice = &*beta;
    let means_slice = &mut *out_means;
    let stds_slice = &mut *out_stds;
    let norm_slice = &mut *out_norm;
    let res_slice = &mut *out_res;

    means_slice
        .par_iter_mut()
        .zip(stds_slice.par_iter_mut())
        .enumerate()
        .for_each(|(j, (mean_out, std_out))| {
            let mut sum = 0.0f32;
            for i in 0..r {
                sum += x_slice[i * c + j];
            }
            let m = sum / (r as f32);
            *mean_out = m;

            let mut sum_sq = 0.0f32;
            for i in 0..r {
                let diff = x_slice[i * c + j] - m;
                sum_sq += diff * diff;
            }
            *std_out = (sum_sq / (r as f32) + eps_f32).sqrt();
        });

    norm_slice
        .par_chunks_mut(c)
        .zip(res_slice.par_chunks_mut(c))
        .enumerate()
        .for_each(|(i, (norm_row, res_row))| {
            let g = gamma_slice[i];
            let b = beta_slice[i];
            let row_offset = i * c;
            for j in 0..c {
                let idx = row_offset + j;
                let norm = (x_slice[idx] - means_slice[j]) / stds_slice[j];
                norm_row[j] = norm;
                res_row[j] = norm * g + b;
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
                dx_row[j] =
                    (g * err_slice[idx] - (sum1_cols[j] * inv_r) - (norm_slice[idx] * sum2_cols[j] * inv_r))
                        / std_slice[j];
            }
        });
}
