use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

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

    let should_clip = clip_limit >= 0.0;

    if output_units * units >= DENSE_LINEAR_BACKWARD_PARALLEL_THRESHOLD {
        grad_weight_slice
            .par_chunks_mut(units)
            .zip(grad_bias_slice.par_iter_mut())
            .enumerate()
            .for_each(|(out_idx, (grad_weight_row, grad_bias_ref))| {
                let err_row = &err_slice[out_idx * seq_len..(out_idx + 1) * seq_len];
                let mut bias_sum = 0.0f32;
                for unit_idx in 0..units {
                    let input_row = &input_slice[unit_idx * seq_len..(unit_idx + 1) * seq_len];
                    let mut sum = 0.0f32;
                    for token_idx in 0..seq_len {
                        let e = err_row[token_idx];
                        sum += e * input_row[token_idx];
                    }
                    grad_weight_row[unit_idx] = if should_clip {
                        sum.clamp(-clip_limit, clip_limit)
                    } else {
                        sum
                    };
                }
                for &e in err_row {
                    bias_sum += e;
                }
                *grad_bias_ref = if should_clip {
                    bias_sum.clamp(-clip_limit, clip_limit)
                } else {
                    bias_sum
                };
            });
    } else {
        for out_idx in 0..output_units {
            let err_row = &err_slice[out_idx * seq_len..(out_idx + 1) * seq_len];
            let grad_weight_row = &mut grad_weight_slice[out_idx * units..(out_idx + 1) * units];
            let mut bias_sum = 0.0f32;
            for unit_idx in 0..units {
                let input_row = &input_slice[unit_idx * seq_len..(unit_idx + 1) * seq_len];
                let mut sum = 0.0f32;
                for token_idx in 0..seq_len {
                    let e = err_row[token_idx];
                    sum += e * input_row[token_idx];
                }
                grad_weight_row[unit_idx] = if should_clip {
                    sum.clamp(-clip_limit, clip_limit)
                } else {
                    sum
                };
            }
            for &e in err_row {
                bias_sum += e;
            }
            grad_bias_slice[out_idx] = if should_clip {
                bias_sum.clamp(-clip_limit, clip_limit)
            } else {
                bias_sum
            };
        }
    }

    if units * seq_len >= DENSE_LINEAR_BACKWARD_PARALLEL_THRESHOLD {
        prev_err_slice
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(unit_idx, prev_err_row)| {
                for token_idx in 0..seq_len {
                    let mut sum = 0.0f32;
                    for out_idx in 0..output_units {
                        sum += weight_slice[out_idx * units + unit_idx] * err_slice[out_idx * seq_len + token_idx];
                    }
                    prev_err_row[token_idx] = sum;
                }
            });
    } else {
        for unit_idx in 0..units {
            let prev_err_row = &mut prev_err_slice[unit_idx * seq_len..(unit_idx + 1) * seq_len];
            for token_idx in 0..seq_len {
                let mut sum = 0.0f32;
                for out_idx in 0..output_units {
                    sum += weight_slice[out_idx * units + unit_idx] * err_slice[out_idx * seq_len + token_idx];
                }
                prev_err_row[token_idx] = sum;
            }
        }
    }
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

    let hidden_slice = &*hidden;
    let weight_slice = &*weight;
    let bias_slice = &*bias;
    let out_slice = &mut *out;

    out_slice
        .par_chunks_mut(batch_size)
        .enumerate()
        .for_each(|(vocab_idx, out_row)| {
            let weight_offset = vocab_idx * units;
            let bias_value = bias_slice[vocab_idx];
            for batch_idx in 0..batch_size {
                let token_col = (batch_idx + 1) * seq_len - 1;
                let mut sum = bias_value;
                for unit_idx in 0..units {
                    sum += weight_slice[weight_offset + unit_idx] * hidden_slice[unit_idx * total_cols + token_col];
                }
                out_row[batch_idx] = sum;
            }
        });
}
