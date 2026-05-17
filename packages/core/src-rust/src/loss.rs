use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi]
pub fn mse_native(y_true: Float32Array, y_pred: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let n = y_true.len() as f32;
    let sum_sq = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum_sq = 0.0;
        for i in 0..y_true_slice.len() {
            let diff = y_true_slice[i] - y_pred_slice[i];
            sum_sq += diff * diff;
        }
        sum_sq
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .map(|(y_true_val, y_pred_val)| {
                let diff = *y_true_val - *y_pred_val;
                diff * diff
            })
            .sum()
    };
    vec![(sum_sq / n) as f64]
}

#[napi]
pub fn mae_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    
    let sum_abs = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let diff = y_pred_slice[i] - y_true_slice[i];
            sum += diff.abs();
            out_grad_slice[i] = if diff > 0.0 { 1.0 / n } else if diff < 0.0 { -1.0 / n } else { 0.0 };
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((y_true_val, y_pred_val), grad)| {
                let diff = *y_pred_val - *y_true_val;
                *grad = if diff > 0.0 { 1.0 / n } else if diff < 0.0 { -1.0 / n } else { 0.0 };
                diff.abs()
            })
            .sum()
    };
    vec![(sum_abs / n) as f64]
}

#[napi]
pub fn huber_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array, delta: f64) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    let delta = delta as f32;
    
    let sum_loss = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let diff = y_pred_slice[i] - y_true_slice[i];
            let abs_diff = diff.abs();
            if abs_diff <= delta {
                sum += 0.5 * diff * diff;
                out_grad_slice[i] = diff / n;
            } else {
                sum += delta * (abs_diff - 0.5 * delta);
                out_grad_slice[i] = (delta * diff.signum()) / n;
            }
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((y_true_val, y_pred_val), grad)| {
                let diff = *y_pred_val - *y_true_val;
                let abs_diff = diff.abs();
                if abs_diff <= delta {
                    *grad = diff / n;
                    0.5 * diff * diff
                } else {
                    *grad = (delta * diff.signum()) / n;
                    delta * (abs_diff - 0.5 * delta)
                }
            })
            .sum()
    };
    vec![(sum_loss / n) as f64]
}

#[napi]
pub fn logcosh_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    
    let sum_loss = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let diff = y_pred_slice[i] - y_true_slice[i];
            sum += (diff.cosh() + 1e-12).ln();
            out_grad_slice[i] = diff.tanh() / n;
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((y_true_val, y_pred_val), grad)| {
                let diff = *y_pred_val - *y_true_val;
                *grad = diff.tanh() / n;
                (diff.cosh() + 1e-12).ln()
            })
            .sum()
    };
    vec![(sum_loss / n) as f64]
}

#[napi]
pub fn hinge_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    
    let sum_loss = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let tr = y_true_slice[i];
            let pr = y_pred_slice[i];
            let v = tr * pr;
            let val = 1.0 - v;
            if val > 0.0 {
                sum += val;
                out_grad_slice[i] = -tr / n;
            } else {
                out_grad_slice[i] = 0.0;
            }
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((tr, pr), grad)| {
                let v = *tr * *pr;
                let val = 1.0 - v;
                if val > 0.0 {
                    *grad = -*tr / n;
                    val
                } else {
                    *grad = 0.0;
                    0.0
                }
            })
            .sum()
    };
    vec![(sum_loss / n) as f64]
}

#[napi]
pub fn squared_hinge_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    
    let sum_loss = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let tr = y_true_slice[i];
            let pr = y_pred_slice[i];
            let v = tr * pr;
            let val = 1.0 - v;
            if val > 0.0 {
                sum += val * val;
                out_grad_slice[i] = (-2.0 * val * tr) / n;
            } else {
                out_grad_slice[i] = 0.0;
            }
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((tr, pr), grad)| {
                let v = *tr * *pr;
                let val = 1.0 - v;
                if val > 0.0 {
                    *grad = (-2.0 * val * *tr) / n;
                    val * val
                } else {
                    *grad = 0.0;
                    0.0
                }
            })
            .sum()
    };
    vec![(sum_loss / n) as f64]
}

#[napi]
pub fn kldivergence_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    
    let sum_loss = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let tr = y_true_slice[i].max(1e-7);
            let pr = y_pred_slice[i].max(1e-7);
            sum += tr * (tr / pr).ln();
            out_grad_slice[i] = -tr / pr / n;
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((tr, pr), grad)| {
                let t = tr.max(1e-7);
                let p = pr.max(1e-7);
                *grad = -t / p / n;
                t * (t / p).ln()
            })
            .sum()
    };
    vec![(sum_loss / n) as f64]
}

#[napi]
pub fn poisson_native_into(y_true: Float32Array, y_pred: Float32Array, mut out_grad: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let out_grad_slice = &mut *out_grad;
    let n = y_true.len() as f32;
    
    let sum_loss = if y_true_slice.len() < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for i in 0..y_true_slice.len() {
            let tr = y_true_slice[i];
            let pr = y_pred_slice[i];
            let p_safe = pr.max(1e-7);
            sum += pr - tr * p_safe.ln();
            out_grad_slice[i] = (1.0 - tr / p_safe) / n;
        }
        sum
    } else {
        y_true_slice
            .par_iter()
            .zip(y_pred_slice.par_iter())
            .zip(out_grad_slice.par_iter_mut())
            .map(|((tr, pr), grad)| {
                let p_safe = pr.max(1e-7);
                *grad = (1.0 - *tr / p_safe) / n;
                *pr - *tr * p_safe.ln()
            })
            .sum()
    };
    vec![(sum_loss / n) as f64]
}
