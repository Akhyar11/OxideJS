use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

#[napi(object)]
pub struct EmbeddingSparseBackwardResult {
    pub unique_indices: Int32Array,
    pub grad: Float32Array,
}

#[napi]
pub fn embedding_forward_native_into(
    indices: Vec<f64>,
    weight_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
    mut out: Float32Array
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    for i in 0..out.len() { out[i] = 0.0; }
    for j in 0..seq_len {
        let raw = indices[j];
        if !raw.is_finite() || raw < 0.0 || raw >= v_size as f64 { continue; }
        let token_idx = raw as usize;
        if let Some(pad_id) = pad_token_id { if pad_id >= 0 && token_idx == pad_id as usize { continue; } }
        for i in 0..dim { out[i * seq_len + j] = weight_data[i * v_size + token_idx]; }
    }
}

#[napi]
pub fn embedding_forward_native_int32_into(
    indices: Int32Array,
    weight_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
    mut out: Float32Array
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    for i in 0..out.len() { out[i] = 0.0; }
    for j in 0..seq_len {
        let raw = indices[j];
        if raw < 0 || raw >= vocab_size as i32 { continue; }
        let token_idx = raw as usize;
        if let Some(pad_id) = pad_token_id { if pad_id >= 0 && token_idx == pad_id as usize { continue; } }
        for i in 0..dim { out[i * seq_len + j] = weight_data[i * v_size + token_idx]; }
    }
}

#[napi]
pub fn embedding_backward_native(
    indices: Vec<f64>,
    err_data: Float32Array,
    mut grad_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    for i in 0..dim {
        for j in 0..seq_len {
            let raw = indices[j];
            if !raw.is_finite() || raw < 0.0 || raw >= v_size as f64 { continue; }
            let token_idx = raw as usize;
            if let Some(pad_id) = pad_token_id {
                if pad_id >= 0 && token_idx == pad_id as usize { continue; }
            }
            grad_data[i * v_size + token_idx] += err_data[i * seq_len + j];
        }
    }
}

#[napi]
pub fn embedding_backward_native_int32(
    indices: Int32Array,
    err_data: Float32Array,
    mut grad_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    for i in 0..dim {
        for j in 0..seq_len {
            let raw = indices[j];
            if raw < 0 || raw >= vocab_size as i32 { continue; }
            let token_idx = raw as usize;
            if let Some(pad_id) = pad_token_id {
                if pad_id >= 0 && token_idx == pad_id as usize { continue; }
            }
            grad_data[i * v_size + token_idx] += err_data[i * seq_len + j];
        }
    }
}

#[napi]
pub fn embedding_backward_sparse_native(
    indices: Int32Array,
    err_data: Float32Array,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
) -> EmbeddingSparseBackwardResult {
    let mut unique_map = HashMap::new();
    let mut unique_vec = Vec::new();
    let seq_len = indices.len();
    let mut pos_in_unique = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id {
                pos_in_unique.push(-1);
                continue;
            }
        }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    let dim = embedding_dim as usize;
    let mut grad = vec![0.0f32; num_unique * dim];

    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim {
            grad[i * num_unique + u_ptr] += err_data[i * seq_len + j];
        }
    }

    EmbeddingSparseBackwardResult {
        unique_indices: Int32Array::from(unique_vec),
        grad: Float32Array::from(grad),
    }
}

#[napi]
pub fn embedding_adam_backward_update_native(
    indices: Int32Array,
    err_data: Float32Array,
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
    pad_token_id: Option<i32>,
) {
    let alpha = alpha as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let epsilon = epsilon as f32;

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let mut unique_map: HashMap<i32, usize> = HashMap::new();
    let mut unique_vec: Vec<i32> = Vec::new();
    let mut pos_in_unique: Vec<i32> = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id { pos_in_unique.push(-1); continue; }
        }
        if idx < 0 || idx >= vocab_size as i32 { pos_in_unique.push(-1); continue; }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    if num_unique == 0 { return; }

    let mut grad: Vec<f32> = vec![0.0f32; dim * num_unique];
    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim { grad[i * num_unique + u_ptr] += err_data[i * seq_len + j]; }
    }

    let weight_slice = &mut *weight;
    let m_slice = &mut *m;
    let v_slice = &mut *v;

    for (j, &token_idx_i32) in unique_vec.iter().enumerate() {
        let token_idx = token_idx_i32 as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let g = grad[i * num_unique + j];
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
pub fn embedding_sgd_backward_update_native(
    indices: Int32Array,
    err_data: Float32Array,
    mut weight: Float32Array,
    alpha: f64,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
) {
    let alpha = alpha as f32;
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let mut unique_map: HashMap<i32, usize> = HashMap::new();
    let mut unique_vec: Vec<i32> = Vec::new();
    let mut pos_in_unique: Vec<i32> = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id { pos_in_unique.push(-1); continue; }
        }
        if idx < 0 || idx >= vocab_size as i32 { pos_in_unique.push(-1); continue; }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    if num_unique == 0 { return; }

    let mut grad = vec![0.0f32; dim * num_unique];
    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim { grad[i * num_unique + u_ptr] += err_data[i * seq_len + j]; }
    }

    let weight_slice = &mut *weight;
    for (j, &token_idx_i32) in unique_vec.iter().enumerate() {
        let token_idx = token_idx_i32 as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            weight_slice[full_idx] -= alpha * grad[i * num_unique + j];
        }
    }
}

#[napi]
pub fn embedding_adagrad_backward_update_native(
    indices: Int32Array,
    err_data: Float32Array,
    mut weight: Float32Array,
    mut sum_data: Float32Array,
    alpha: f64,
    epsilon: f64,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
) {
    let alpha = alpha as f32;
    let epsilon = epsilon as f32;
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let mut unique_map: HashMap<i32, usize> = HashMap::new();
    let mut unique_vec: Vec<i32> = Vec::new();
    let mut pos_in_unique: Vec<i32> = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id { pos_in_unique.push(-1); continue; }
        }
        if idx < 0 || idx >= vocab_size as i32 { pos_in_unique.push(-1); continue; }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    if num_unique == 0 { return; }

    let mut grad = vec![0.0f32; dim * num_unique];
    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim { grad[i * num_unique + u_ptr] += err_data[i * seq_len + j]; }
    }

    let weight_slice = &mut *weight;
    let sum_slice = &mut *sum_data;
    for (j, &token_idx_i32) in unique_vec.iter().enumerate() {
        let token_idx = token_idx_i32 as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let g = grad[i * num_unique + j];
            let accumulated = sum_slice[full_idx] + g * g;
            sum_slice[full_idx] = accumulated;
            weight_slice[full_idx] -= alpha * g / (accumulated + epsilon).sqrt();
        }
    }
}

#[napi]
pub fn embedding_momentum_backward_update_native(
    indices: Int32Array,
    err_data: Float32Array,
    mut weight: Float32Array,
    mut v_data: Float32Array,
    alpha: f64,
    beta: f64,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let mut unique_map: HashMap<i32, usize> = HashMap::new();
    let mut unique_vec: Vec<i32> = Vec::new();
    let mut pos_in_unique: Vec<i32> = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id { pos_in_unique.push(-1); continue; }
        }
        if idx < 0 || idx >= vocab_size as i32 { pos_in_unique.push(-1); continue; }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    if num_unique == 0 { return; }

    let mut grad = vec![0.0f32; dim * num_unique];
    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim { grad[i * num_unique + u_ptr] += err_data[i * seq_len + j]; }
    }

    let weight_slice = &mut *weight;
    let v_slice = &mut *v_data;
    for (j, &token_idx_i32) in unique_vec.iter().enumerate() {
        let token_idx = token_idx_i32 as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let g = grad[i * num_unique + j];
            let v_new = beta * v_slice[full_idx] + alpha * g;
            v_slice[full_idx] = v_new;
            weight_slice[full_idx] -= v_new;
        }
    }
}

#[napi]
pub fn embedding_nag_backward_update_native(
    indices: Int32Array,
    err_data: Float32Array,
    mut weight: Float32Array,
    mut v_data: Float32Array,
    alpha: f64,
    beta: f64,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let mut unique_map: HashMap<i32, usize> = HashMap::new();
    let mut unique_vec: Vec<i32> = Vec::new();
    let mut pos_in_unique: Vec<i32> = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id { pos_in_unique.push(-1); continue; }
        }
        if idx < 0 || idx >= vocab_size as i32 { pos_in_unique.push(-1); continue; }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    if num_unique == 0 { return; }

    let mut grad = vec![0.0f32; dim * num_unique];
    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim { grad[i * num_unique + u_ptr] += err_data[i * seq_len + j]; }
    }

    let weight_slice = &mut *weight;
    let v_slice = &mut *v_data;
    for (j, &token_idx_i32) in unique_vec.iter().enumerate() {
        let token_idx = token_idx_i32 as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let g = grad[i * num_unique + j];
            let v_old = v_slice[full_idx];
            let v_new = beta * v_old + alpha * (g - beta * v_old);
            v_slice[full_idx] = v_new;
            weight_slice[full_idx] -= v_new;
        }
    }
}
