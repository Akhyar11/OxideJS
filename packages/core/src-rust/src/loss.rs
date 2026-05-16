use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const MSE_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi]
pub fn mse_native(y_true: Float32Array, y_pred: Float32Array) -> Vec<f64> {
    let y_true_slice = &*y_true;
    let y_pred_slice = &*y_pred;
    let n = y_true.len() as f32;
    let sum_sq = if y_true_slice.len() < MSE_PARALLEL_THRESHOLD {
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
