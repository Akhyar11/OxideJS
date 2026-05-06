use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const ADAM_PARALLEL_THRESHOLD: usize = 8 * 1024;
const ELEMENTWISE_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi]
pub fn adam_update_native(
    grad: Float32Array,
    mut m: Float32Array,
    mut v: Float32Array,
    mut buffer: Float32Array,
    t: u32,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64
) {
    let alpha = alpha as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let epsilon = epsilon as f32;

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    let grad_slice = &*grad;
    let m_slice = &mut *m;
    let v_slice = &mut *v;
    let buffer_slice = &mut *buffer;

    if buffer_slice.len() < ADAM_PARALLEL_THRESHOLD {
        for i in 0..buffer_slice.len() {
            let g = grad_slice[i];
            let m_new = beta1 * m_slice[i] + one_minus_beta1 * g;
            let v_new = beta2 * v_slice[i] + one_minus_beta2 * g * g;
            m_slice[i] = m_new;
            v_slice[i] = v_new;

            let m_hat = m_new * bias_correction1;
            let v_hat = v_new * bias_correction2;
            buffer_slice[i] = alpha * m_hat / (v_hat.sqrt() + epsilon);
        }
    } else {
        buffer_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(m_slice.par_iter_mut())
            .zip(v_slice.par_iter_mut())
            .for_each(|(((b_val, &g), m_val), v_val)| {
                let m_new = beta1 * (*m_val) + one_minus_beta1 * g;
                let v_new = beta2 * (*v_val) + one_minus_beta2 * g * g;
                *m_val = m_new;
                *v_val = v_new;

                let m_hat = m_new * bias_correction1;
                let v_hat = v_new * bias_correction2;
                *b_val = alpha * m_hat / (v_hat.sqrt() + epsilon);
            });
    }
}

#[napi]
pub fn adam_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut m: Float32Array,
    mut v: Float32Array,
    t: u32,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let epsilon = epsilon as f32;

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();

    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let m_slice = &mut *m;
    let v_slice = &mut *v;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;

            let g = grad_slice[grad_idx];
            let m_new = beta1 * m_slice[full_idx] + one_minus_beta1 * g;
            let v_new = beta2 * v_slice[full_idx] + one_minus_beta2 * g * g;
            m_slice[full_idx] = m_new;
            v_slice[full_idx] = v_new;

            let m_hat = m_new * bias_correction1;
            let v_hat = v_new * bias_correction2;
            weight_slice[full_idx] -= alpha * m_hat / (v_hat.sqrt() + epsilon);
        }
    }
}

#[napi]
pub fn sgd_update_native(grad: Float32Array, mut out: Float32Array, alpha: f64) {
    let alpha = alpha as f32;
    let grad_slice = &*grad;
    let out_slice = &mut *out;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() { out_slice[i] = grad_slice[i] * alpha; }
    } else {
        out_slice.par_iter_mut().zip(grad_slice.par_iter()).for_each(|(o, &g)| {
            *o = g * alpha;
        });
    }
}

#[napi]
pub fn sgd_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    alpha: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            weight_slice[full_idx] -= alpha * grad_slice[grad_idx];
        }
    }
}

#[napi]
pub fn adagrad_update_native(
    grad: Float32Array,
    mut sum: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    epsilon: f64
) {
    let alpha = alpha as f32;
    let eps = epsilon as f32;
    let grad_slice = &*grad;
    let sum_slice = &mut *sum;
    let out_slice = &mut *out;

    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            let g = grad_slice[i];
            sum_slice[i] += g * g;
            out_slice[i] = alpha * g / (sum_slice[i].sqrt() + eps);
        }
    } else {
        out_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(sum_slice.par_iter_mut())
            .for_each(|((o, &g), s)| {
                *s += g * g;
                *o = alpha * g / (s.sqrt() + eps);
            });
    }
}

#[napi]
pub fn adagrad_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut sum: Float32Array,
    alpha: f64,
    epsilon: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let eps = epsilon as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let sum_slice = &mut *sum;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            let g = grad_slice[grad_idx];
            sum_slice[full_idx] += g * g;
            weight_slice[full_idx] -= alpha * g / (sum_slice[full_idx].sqrt() + eps);
        }
    }
}

#[napi]
pub fn momentum_update_native(
    grad: Float32Array,
    mut v: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    beta: f64
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let grad_slice = &*grad;
    let v_slice = &mut *v;
    let out_slice = &mut *out;

    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            v_slice[i] = beta * v_slice[i] + alpha * grad_slice[i];
            out_slice[i] = v_slice[i];
        }
    } else {
        out_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(v_slice.par_iter_mut())
            .for_each(|((o, &g), v_val)| {
                *v_val = beta * (*v_val) + alpha * g;
                *o = *v_val;
            });
    }
}

#[napi]
pub fn momentum_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut v: Float32Array,
    alpha: f64,
    beta: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let v_slice = &mut *v;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            v_slice[full_idx] = beta * v_slice[full_idx] + alpha * grad_slice[grad_idx];
            weight_slice[full_idx] -= v_slice[full_idx];
        }
    }
}

#[napi]
pub fn nag_update_native(
    grad: Float32Array,
    mut v: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    beta: f64
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let grad_slice = &*grad;
    let v_slice = &mut *v;
    let out_slice = &mut *out;

    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            let v_old = v_slice[i];
            let v_new = beta * v_old + alpha * (grad_slice[i] - beta * v_old);
            v_slice[i] = v_new;
            out_slice[i] = v_new;
        }
    } else {
        out_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(v_slice.par_iter_mut())
            .for_each(|((o, &g), v_val)| {
                let v_old = *v_val;
                let v_new = beta * v_old + alpha * (g - beta * v_old);
                *v_val = v_new;
                *o = v_new;
            });
    }
}

#[napi]
pub fn nag_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut v: Float32Array,
    alpha: f64,
    beta: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let v_slice = &mut *v;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            let v_old = v_slice[full_idx];
            let v_new = beta * v_old + alpha * (grad_slice[grad_idx] - beta * v_old);
            v_slice[full_idx] = v_new;
            weight_slice[full_idx] -= v_new;
        }
    }
}
