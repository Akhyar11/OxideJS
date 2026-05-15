use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const MASKED_SPARSE_SOFTMAX_PARALLEL_THRESHOLD: usize = 2048;
const MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK: usize = 16;
const MSE_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi(object)]
pub struct MaskedSparseSoftmaxCrossEntropyResult {
    pub loss: f64,
    pub valid_tokens: u32,
}

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

#[napi]
pub fn masked_sparse_softmax_cross_entropy_into(
    logits: Float32Array,
    input_tokens: Float32Array,
    targets: Float32Array,
    seq_len: u32,
    batch_size: u32,
    vocab_size: u32,
    pad_token_id: Option<i32>,
    mut out_grad: Float32Array,
) -> MaskedSparseSoftmaxCrossEntropyResult {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;
    let total_tokens = seq_len * batch_size;
    let epsilon = 1e-15f32;
    let pad_token_id = pad_token_id.unwrap_or(-1);

    if logits.len() != vocab_size * total_tokens {
        panic!(
            "masked_sparse_softmax_cross_entropy_into: logits length mismatch {} != {}",
            logits.len(),
            vocab_size * total_tokens
        );
    }
    if input_tokens.len() != total_tokens || targets.len() != total_tokens {
        panic!(
            "masked_sparse_softmax_cross_entropy_into: token length mismatch input={} targets={} total_tokens={}",
            input_tokens.len(),
            targets.len(),
            total_tokens
        );
    }
    if out_grad.len() != logits.len() {
        panic!(
            "masked_sparse_softmax_cross_entropy_into: grad length mismatch {} != {}",
            out_grad.len(),
            logits.len()
        );
    }

    let logits_slice = &*logits;
    let input_slice = &*input_tokens;
    let target_slice = &*targets;
    let grad_slice = &mut *out_grad;

    let process_block = |block_index: usize,
                         grad_ptr_addr: usize,
                         logits_ptr_addr: usize|
     -> (f64, usize) {
        let start_token = block_index * MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK;
        let end_token = (start_token + MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK).min(total_tokens);
        let block_len = end_token - start_token;
        let grad_ptr = grad_ptr_addr as *mut f32;
        let logits_ptr = logits_ptr_addr as *const f32;

        let mut max_logits = [f32::NEG_INFINITY; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut sum_exps = [0.0f32; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut inv_sum_exps = [0.0f32; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut target_ids = [0usize; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut valid_mask = [false; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];

        let mut valid_tokens = 0usize;
        for local_idx in 0..block_len {
            let token_index = start_token + local_idx;
            let batch_idx = token_index / seq_len;
            let pos = token_index % seq_len;
            let source_index = pos * batch_size + batch_idx;
            let source_token = input_slice[source_index] as i32;
            let target_token = target_slice[source_index] as i32;
            let is_valid_position = pos < seq_len - 1
                && (pad_token_id < 0
                    || (source_token != pad_token_id && target_token != pad_token_id));

            if !is_valid_position {
                continue;
            }
            if target_token < 0 || target_token as usize >= vocab_size {
                panic!(
                    "masked_sparse_softmax_cross_entropy_into: target token {} out of range 0..{} at batch {} pos {}",
                    target_token,
                    vocab_size.saturating_sub(1),
                    batch_idx,
                    pos
                );
            }

            valid_mask[local_idx] = true;
            target_ids[local_idx] = target_token as usize;
            valid_tokens += 1;
        }

        if valid_tokens == 0 {
            for vocab_idx in 0..vocab_size {
                let row_offset = vocab_idx * total_tokens + start_token;
                unsafe {
                    std::ptr::write_bytes(grad_ptr.add(row_offset), 0, block_len);
                }
            }
            return (0.0, 0);
        }

        for vocab_idx in 0..vocab_size {
            let row_offset = vocab_idx * total_tokens + start_token;
            for local_idx in 0..block_len {
                if !valid_mask[local_idx] {
                    continue;
                }
                let value = unsafe { *logits_ptr.add(row_offset + local_idx) };
                if value > max_logits[local_idx] {
                    max_logits[local_idx] = value;
                }
            }
        }

        for vocab_idx in 0..vocab_size {
            let row_offset = vocab_idx * total_tokens + start_token;
            for local_idx in 0..block_len {
                let grad_ref = unsafe { &mut *grad_ptr.add(row_offset + local_idx) };
                if !valid_mask[local_idx] {
                    *grad_ref = 0.0;
                    continue;
                }
                let exp_val = (unsafe { *logits_ptr.add(row_offset + local_idx) }
                    - max_logits[local_idx])
                    .exp();
                *grad_ref = exp_val;
                sum_exps[local_idx] += exp_val;
            }
        }

        let mut total_loss = 0.0f64;
        for local_idx in 0..block_len {
            if !valid_mask[local_idx] {
                continue;
            }
            if !sum_exps[local_idx].is_finite() || sum_exps[local_idx] <= 0.0 {
                inv_sum_exps[local_idx] = 0.0;
                total_loss -= (1.0f32 / vocab_size as f32).max(epsilon).ln() as f64;
            } else {
                inv_sum_exps[local_idx] = 1.0 / sum_exps[local_idx];
            }
        }

        for vocab_idx in 0..vocab_size {
            let row_offset = vocab_idx * total_tokens + start_token;
            for local_idx in 0..block_len {
                let grad_ref = unsafe { &mut *grad_ptr.add(row_offset + local_idx) };
                if !valid_mask[local_idx] {
                    continue;
                }
                if inv_sum_exps[local_idx] == 0.0 {
                    *grad_ref = 1.0 / vocab_size as f32;
                } else {
                    *grad_ref *= inv_sum_exps[local_idx];
                }
            }
        }

        for local_idx in 0..block_len {
            if !valid_mask[local_idx] {
                continue;
            }
            let token_index = start_token + local_idx;
            let target_offset = target_ids[local_idx] * total_tokens + token_index;
            let grad_ref = unsafe { &mut *grad_ptr.add(target_offset) };
            if inv_sum_exps[local_idx] != 0.0 {
                total_loss -= (*grad_ref).max(epsilon).ln() as f64;
            }
            *grad_ref -= 1.0;
        }

        (total_loss, valid_tokens)
    };

    let total_blocks = total_tokens.div_ceil(MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK);
    let (total_loss, valid_tokens) = if total_tokens >= MASKED_SPARSE_SOFTMAX_PARALLEL_THRESHOLD {
        let grad_ptr_addr = grad_slice.as_mut_ptr() as usize;
        let logits_ptr_addr = logits_slice.as_ptr() as usize;
        (0..total_blocks)
            .into_par_iter()
            .map(|block_index| process_block(block_index, grad_ptr_addr, logits_ptr_addr))
            .reduce(
                || (0.0f64, 0usize),
                |(loss_a, valid_a), (loss_b, valid_b)| (loss_a + loss_b, valid_a + valid_b),
            )
    } else {
        let grad_ptr_addr = grad_slice.as_mut_ptr() as usize;
        let logits_ptr_addr = logits_slice.as_ptr() as usize;
        let mut total_loss = 0.0f64;
        let mut valid_tokens = 0usize;
        for block_index in 0..total_blocks {
            let (loss, valid) = process_block(block_index, grad_ptr_addr, logits_ptr_addr);
            total_loss += loss;
            valid_tokens += valid;
        }
        (total_loss, valid_tokens)
    };

    if valid_tokens == 0 {
        panic!("masked_sparse_softmax_cross_entropy_into: no valid tokens");
    }

    let inv_valid_tokens = 1.0f32 / valid_tokens as f32;
    if grad_slice.len() >= MASKED_SPARSE_SOFTMAX_PARALLEL_THRESHOLD {
        grad_slice.par_iter_mut().for_each(|value| {
            *value *= inv_valid_tokens;
        });
    } else {
        for value in grad_slice.iter_mut() {
            *value *= inv_valid_tokens;
        }
    }

    MaskedSparseSoftmaxCrossEntropyResult {
        loss: total_loss / valid_tokens as f64,
        valid_tokens: valid_tokens as u32,
    }
}
