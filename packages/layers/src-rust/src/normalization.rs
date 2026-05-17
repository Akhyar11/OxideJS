use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

#[derive(Copy, Clone)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    pub fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Copy, Clone)]
struct SendPtrConst(*const f32);
unsafe impl Send for SendPtrConst {}
unsafe impl Sync for SendPtrConst {}

impl SendPtrConst {
    #[inline(always)]
    pub fn get(self) -> *const f32 {
        self.0
    }
}

#[napi]
pub fn layer_normalization_forward_native(
    inputs: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    epsilon: f64,
    mut out: Float32Array,
    mut mean: Float32Array,
    mut inv_std: Float32Array,
) {
    let total_len = inputs.len();
    let cols = gamma.len();
    if cols == 0 {
        return;
    }
    let rows = total_len / cols;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let gamma_ptr = SendPtrConst(gamma.as_ptr());
    let beta_ptr = SendPtrConst(beta.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let mean_ptr = SendPtr(mean.as_mut_ptr());
    let inv_std_ptr = SendPtr(inv_std.as_mut_ptr());

    (0..rows).into_par_iter().for_each(move |i| {
        let row_offset = i * cols;
        unsafe {
            let inputs_slice = std::slice::from_raw_parts(inputs_ptr.get().add(row_offset), cols);
            let out_slice = std::slice::from_raw_parts_mut(out_ptr.get().add(row_offset), cols);
            let gamma_slice = std::slice::from_raw_parts(gamma_ptr.get(), cols);
            let beta_slice = std::slice::from_raw_parts(beta_ptr.get(), cols);

            let mut sum = 0.0;
            for j in 0..cols {
                sum += inputs_slice[j];
            }
            let m = sum / cols as f32;
            *mean_ptr.get().add(i) = m;

            let mut var_sum = 0.0;
            for j in 0..cols {
                let diff = inputs_slice[j] - m;
                var_sum += diff * diff;
            }
            let var = var_sum / cols as f32;
            let istd = 1.0 / (var + epsilon as f32).sqrt();
            *inv_std_ptr.get().add(i) = istd;

            for j in 0..cols {
                let x_centered = inputs_slice[j] - m;
                let x_norm = x_centered * istd;
                out_slice[j] = x_norm * gamma_slice[j] + beta_slice[j];
            }
        }
    });
}

#[napi]
pub fn layer_normalization_backward_native(
    grad_out: Float32Array,
    inputs: Float32Array,
    mean: Float32Array,
    inv_std: Float32Array,
    gamma: Float32Array,
    mut grad_in: Float32Array,
    mut grad_gamma: Float32Array,
    mut grad_beta: Float32Array,
) {
    let cols = gamma.len();
    if cols == 0 {
        return;
    }
    let rows = grad_out.len() / cols;

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let mean_ptr = SendPtrConst(mean.as_ptr());
    let inv_std_ptr = SendPtrConst(inv_std.as_ptr());
    let gamma_ptr = SendPtrConst(gamma.as_ptr());
    let grad_in_ptr = SendPtr(grad_in.as_mut_ptr());

    let (final_grad_gamma, final_grad_beta) = (0..rows)
        .into_par_iter()
        .fold(
            move || (vec![0.0f32; cols], vec![0.0f32; cols]),
            move |(mut local_gg, mut local_gb), i| {
                let row_offset = i * cols;
                unsafe {
                    let dy_slice = std::slice::from_raw_parts(grad_out_ptr.get().add(row_offset), cols);
                    let x_slice = std::slice::from_raw_parts(inputs_ptr.get().add(row_offset), cols);
                    let dx_slice = std::slice::from_raw_parts_mut(grad_in_ptr.get().add(row_offset), cols);
                    let gamma_slice = std::slice::from_raw_parts(gamma_ptr.get(), cols);

                    let m = *mean_ptr.get().add(i);
                    let istd = *inv_std_ptr.get().add(i);

                    let mut sum_dhat = 0.0;
                    let mut sum_dhat_xhat = 0.0;

                    for j in 0..cols {
                        let dy = dy_slice[j];
                        let xhat = (x_slice[j] - m) * istd;
                        let dhat = dy * gamma_slice[j];

                        sum_dhat += dhat;
                        sum_dhat_xhat += dhat * xhat;

                        local_gg[j] += dy * xhat;
                        local_gb[j] += dy;
                    }

                    let mean_dhat = sum_dhat / cols as f32;
                    let mean_dhat_xhat = sum_dhat_xhat / cols as f32;

                    for j in 0..cols {
                        let xhat = (x_slice[j] - m) * istd;
                        let dhat = dy_slice[j] * gamma_slice[j];
                        dx_slice[j] = istd * (dhat - mean_dhat - xhat * mean_dhat_xhat);
                    }
                }
                (local_gg, local_gb)
            },
        )
        .reduce(
            move || (vec![0.0f32; cols], vec![0.0f32; cols]),
            move |(mut gg1, mut gb1), (gg2, gb2)| {
                for j in 0..cols {
                    gg1[j] += gg2[j];
                    gb1[j] += gb2[j];
                }
                (gg1, gb1)
            },
        );

    let grad_gamma_slice = &mut *grad_gamma;
    let grad_beta_slice = &mut *grad_beta;
    for j in 0..cols {
        grad_gamma_slice[j] += final_grad_gamma[j];
        grad_beta_slice[j] += final_grad_beta[j];
    }
}

#[napi]
pub fn batch_normalization_forward_native(
    inputs: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    mut moving_mean: Float32Array,
    mut moving_variance: Float32Array,
    epsilon: f64,
    momentum: f64,
    training: bool,
    mut out: Float32Array,
    mut mean: Float32Array,
    mut inv_std: Float32Array,
) {
    let cols = gamma.len();
    if cols == 0 {
        return;
    }
    let rows = inputs.len() / cols;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let gamma_ptr = SendPtrConst(gamma.as_ptr());
    let beta_ptr = SendPtrConst(beta.as_ptr());
    let moving_mean_ptr = SendPtr(moving_mean.as_mut_ptr());
    let moving_variance_ptr = SendPtr(moving_variance.as_mut_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let mean_ptr = SendPtr(mean.as_mut_ptr());
    let inv_std_ptr = SendPtr(inv_std.as_mut_ptr());

    if training {
        (0..cols).into_par_iter().for_each(move |j| {
            let mut sum = 0.0;
            for i in 0..rows {
                unsafe {
                    sum += *inputs_ptr.get().add(i * cols + j);
                }
            }
            let m = sum / rows as f32;
            unsafe {
                *mean_ptr.get().add(j) = m;
            }

            let mut var_sum = 0.0;
            for i in 0..rows {
                unsafe {
                    let diff = *inputs_ptr.get().add(i * cols + j) - m;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / rows as f32;
            let istd = 1.0 / (var + epsilon as f32).sqrt();
            unsafe {
                *inv_std_ptr.get().add(j) = istd;

                *moving_mean_ptr.get().add(j) = *moving_mean_ptr.get().add(j) * momentum as f32 + m * (1.0 - momentum as f32);
                *moving_variance_ptr.get().add(j) = *moving_variance_ptr.get().add(j) * momentum as f32 + var * (1.0 - momentum as f32);
            }
        });
    } else {
        (0..cols).into_par_iter().for_each(move |j| {
            unsafe {
                let m = *moving_mean_ptr.get().add(j);
                *mean_ptr.get().add(j) = m;
                let var = *moving_variance_ptr.get().add(j);
                *inv_std_ptr.get().add(j) = 1.0 / (var + epsilon as f32).sqrt();
            }
        });
    }

    (0..rows).into_par_iter().for_each(move |i| {
        let row_offset = i * cols;
        for j in 0..cols {
            unsafe {
                let m = *mean_ptr.get().add(j);
                let istd = *inv_std_ptr.get().add(j);
                let x_centered = *inputs_ptr.get().add(row_offset + j) - m;
                let x_norm = x_centered * istd;
                *out_ptr.get().add(row_offset + j) = x_norm * *gamma_ptr.get().add(j) + *beta_ptr.get().add(j);
            }
        }
    });
}

#[napi]
pub fn batch_normalization_backward_native(
    grad_out: Float32Array,
    inputs: Float32Array,
    mean: Float32Array,
    inv_std: Float32Array,
    gamma: Float32Array,
    training: bool,
    mut grad_in: Float32Array,
    mut grad_gamma: Float32Array,
    mut grad_beta: Float32Array,
) {
    let cols = gamma.len();
    if cols == 0 {
        return;
    }
    let rows = grad_out.len() / cols;

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let mean_ptr = SendPtrConst(mean.as_ptr());
    let inv_std_ptr = SendPtrConst(inv_std.as_ptr());
    let gamma_ptr = SendPtrConst(gamma.as_ptr());
    let grad_in_ptr = SendPtr(grad_in.as_mut_ptr());
    let grad_gamma_ptr = SendPtr(grad_gamma.as_mut_ptr());
    let grad_beta_ptr = SendPtr(grad_beta.as_mut_ptr());

    if training {
        (0..cols).into_par_iter().for_each(move |j| {
            unsafe {
                let m = *mean_ptr.get().add(j);
                let istd = *inv_std_ptr.get().add(j);
                let gam = *gamma_ptr.get().add(j);

                let mut sum_dhat = 0.0;
                let mut sum_dhat_xhat = 0.0;

                let mut local_gg = 0.0;
                let mut local_gb = 0.0;

                for i in 0..rows {
                    let idx = i * cols + j;
                    let dy = *grad_out_ptr.get().add(idx);
                    let xhat = (*inputs_ptr.get().add(idx) - m) * istd;
                    let dhat = dy * gam;

                    sum_dhat += dhat;
                    sum_dhat_xhat += dhat * xhat;

                    local_gg += dy * xhat;
                    local_gb += dy;
                }

                *grad_gamma_ptr.get().add(j) += local_gg;
                *grad_beta_ptr.get().add(j) += local_gb;

                let mean_dhat = sum_dhat / rows as f32;
                let mean_dhat_xhat = sum_dhat_xhat / rows as f32;

                for i in 0..rows {
                    let idx = i * cols + j;
                    let xhat = (*inputs_ptr.get().add(idx) - m) * istd;
                    let dhat = *grad_out_ptr.get().add(idx) * gam;
                    *grad_in_ptr.get().add(idx) = istd * (dhat - mean_dhat - xhat * mean_dhat_xhat);
                }
            }
        });
    } else {
        (0..cols).into_par_iter().for_each(move |j| {
            unsafe {
                let m = *mean_ptr.get().add(j);
                let istd = *inv_std_ptr.get().add(j);
                let gam = *gamma_ptr.get().add(j);

                let mut local_gg = 0.0;
                let mut local_gb = 0.0;

                for i in 0..rows {
                    let idx = i * cols + j;
                    let dy = *grad_out_ptr.get().add(idx);
                    let xhat = (*inputs_ptr.get().add(idx) - m) * istd;

                    local_gg += dy * xhat;
                    local_gb += dy;

                    *grad_in_ptr.get().add(idx) = dy * gam * istd;
                }

                *grad_gamma_ptr.get().add(j) += local_gg;
                *grad_beta_ptr.get().add(j) += local_gb;
            }
        });
    }
}
