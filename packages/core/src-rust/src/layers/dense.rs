use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use crate::math::{SafeRawPtr, SafeRawPtrMut};

const DENSE_LINEAR_BACKWARD_PARALLEL_THRESHOLD: usize = 64 * 64;

#[napi]
pub fn dense_linear_backward_native_into(
    err_activation: Float32Array,
    input: Float32Array,
    weight: Float32Array,
    output_units: u32,
    units: u32,
    seq_len: u32,
    clip_limit: f64,
    mut grad_weight_out: Float32Array,
    mut grad_bias_out: Float32Array,
    mut prev_err_out: Float32Array,
) {
    let output_units = output_units as usize;
    let units = units as usize;
    let seq_len = seq_len as usize;
    let clip_limit = clip_limit as f32;

    if err_activation.len() != output_units * seq_len {
        panic!(
            "dense_linear_backward_native_into: err_activation length mismatch {} != {}",
            err_activation.len(),
            output_units * seq_len
        );
    }
    if input.len() != units * seq_len {
        panic!(
            "dense_linear_backward_native_into: input length mismatch {} != {}",
            input.len(),
            units * seq_len
        );
    }
    if weight.len() != output_units * units {
        panic!(
            "dense_linear_backward_native_into: weight length mismatch {} != {}",
            weight.len(),
            output_units * units
        );
    }
    if grad_weight_out.len() != output_units * units {
        panic!(
            "dense_linear_backward_native_into: grad_weight_out length mismatch {} != {}",
            grad_weight_out.len(),
            output_units * units
        );
    }
    if grad_bias_out.len() != output_units {
        panic!(
            "dense_linear_backward_native_into: grad_bias_out length mismatch {} != {}",
            grad_bias_out.len(),
            output_units
        );
    }
    if prev_err_out.len() != units * seq_len {
        panic!(
            "dense_linear_backward_native_into: prev_err_out length mismatch {} != {}",
            prev_err_out.len(),
            units * seq_len
        );
    }

    let err_slice = &*err_activation;
    let input_slice = &*input;
    let weight_slice = &*weight;
    let grad_weight_slice = &mut *grad_weight_out;
    let grad_bias_slice = &mut *grad_bias_out;
    let prev_err_slice = &mut *prev_err_out;

    let err_len = err_activation.len();
    let input_len = input.len();
    let weight_len = weight.len();
    let g_weight_len = grad_weight_out.len();
    let g_bias_len = grad_bias_out.len();
    let p_err_len = prev_err_out.len();

    let err_p = SafeRawPtr(err_activation.as_ptr() as usize);
    let input_p = SafeRawPtr(input.as_ptr() as usize);
    let weight_p = SafeRawPtr(weight.as_ptr() as usize);

    let grad_weight_p = SafeRawPtrMut(grad_weight_out.as_ptr() as usize);
    let grad_bias_p = SafeRawPtrMut(grad_bias_out.as_ptr() as usize);
    let prev_err_p = SafeRawPtrMut(prev_err_out.as_ptr() as usize);

    let should_clip = clip_limit >= 0.0;

    // Parallel Gradient Computation (Weight & Bias)
    (0..output_units).into_par_iter().for_each(|out_idx| {
        unsafe {
            let err_ptr = std::slice::from_raw_parts(err_p.0 as *const f32, err_len);
            let input_ptr = std::slice::from_raw_parts(input_p.0 as *const f32, input_len);
            let g_weight_ptr = std::slice::from_raw_parts_mut(grad_weight_p.0 as *mut f32, g_weight_len);
            let g_bias_ptr = std::slice::from_raw_parts_mut(grad_bias_p.0 as *mut f32, g_bias_len);

            let err_row = &err_ptr[out_idx * seq_len..(out_idx + 1) * seq_len];
            let grad_weight_row = &mut g_weight_ptr[out_idx * units..(out_idx + 1) * units];

            let mut bias_sum = 0.0f32;
            for token_idx in 0..seq_len {
                bias_sum += err_row[token_idx];
            }
            g_bias_ptr[out_idx] = if should_clip { bias_sum.clamp(-clip_limit, clip_limit) } else { bias_sum };

            // Loop interchange for cache efficiency: dW = dY * X^T
            for unit_idx in 0..units {
                let input_row = &input_ptr[unit_idx * seq_len..(unit_idx + 1) * seq_len];
                let mut sum = 0.0f32;
                // LLVM can auto-vectorize this loop because both err_row and input_row are contiguous
                for token_idx in 0..seq_len {
                    sum += err_row[token_idx] * input_row[token_idx];
                }
                grad_weight_row[unit_idx] = if should_clip { sum.clamp(-clip_limit, clip_limit) } else { sum };
            }
        }
    });

    // Parallel Error Backpropagation: dX = W^T * dY
    (0..units).into_par_iter().for_each(|unit_idx| {
        unsafe {
            let weight_ptr = std::slice::from_raw_parts(weight_p.0 as *const f32, weight_len);
            let err_ptr = std::slice::from_raw_parts(err_p.0 as *const f32, err_len);
            let p_err_ptr = std::slice::from_raw_parts_mut(prev_err_p.0 as *mut f32, p_err_len);

            let prev_err_row = &mut p_err_ptr[unit_idx * seq_len..(unit_idx + 1) * seq_len];
            for token_idx in 0..seq_len {
                let mut sum = 0.0f32;
                for out_idx in 0..output_units {
                    sum += weight_ptr[out_idx * units + unit_idx] * err_ptr[out_idx * seq_len + token_idx];
                }
                prev_err_row[token_idx] = sum;
            }
        }
    });
}

#[napi]
pub fn project_last_token_logits_native_into(
    hidden: Float32Array,
    weight: Float32Array,
    bias: Float32Array,
    units: u32,
    seq_len: u32,
    batch_size: u32,
    vocab_size: u32,
    mut out: Float32Array,
) {
    let units = units as usize;
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;
    let total_cols = seq_len * batch_size;

    if hidden.len() != units * total_cols {
        panic!(
            "project_last_token_logits_native_into: hidden length mismatch {} != {}",
            hidden.len(),
            units * total_cols
        );
    }
    if weight.len() != vocab_size * units {
        panic!(
            "project_last_token_logits_native_into: weight length mismatch {} != {}",
            weight.len(),
            vocab_size * units
        );
    }
    if bias.len() != vocab_size {
        panic!(
            "project_last_token_logits_native_into: bias length mismatch {} != {}",
            bias.len(),
            vocab_size
        );
    }
    if out.len() != vocab_size * batch_size {
        panic!(
            "project_last_token_logits_native_into: out length mismatch {} != {}",
            out.len(),
            vocab_size * batch_size
        );
    }

    let weight_len = weight.len();
    let bias_len = bias.len();
    let out_len = out.len();
    let hidden_p = SafeRawPtr(hidden.as_ptr() as usize);
    let weight_p = SafeRawPtr(weight.as_ptr() as usize);
    let bias_p = SafeRawPtr(bias.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    // Optimized gathered hidden states (only last token per batch)
    // tile size: [units][batch_size]
    let mut gathered_hidden = vec![0.0f32; units * batch_size];
    for u in 0..units {
        for b in 0..batch_size {
            let token_col = (b + 1) * seq_len - 1;
            gathered_hidden[u * batch_size + b] = hidden[u * total_cols + token_col];
        }
    }

    (0..vocab_size).into_par_iter().for_each(|vocab_idx| {
        unsafe {
            let w_ptr = std::slice::from_raw_parts(weight_p.0 as *const f32, weight_len);
            let b_ptr = std::slice::from_raw_parts(bias_p.0 as *const f32, bias_len);
            let out_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, out_len);

            let weight_offset = vocab_idx * units;
            let bias_val = b_ptr[vocab_idx];
            
            for batch_idx in 0..batch_size {
                let mut sum = bias_val;
                for unit_idx in 0..units {
                    // Access gathered_hidden (u, b) and w (v, u)
                    // gathered_hidden is [units][batch_size], weight is [vocab_size][units]
                    sum += w_ptr[weight_offset + unit_idx] * gathered_hidden[unit_idx * batch_size + batch_idx];
                }
                out_ptr[vocab_idx * batch_size + batch_idx] = sum;
            }
        }
    });
}
