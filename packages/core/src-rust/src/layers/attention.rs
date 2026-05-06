use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

#[napi]
pub fn apply_attention_mask_native(
    mut data: Float32Array,
    pad_mask: Vec<bool>,
    rows: u32,
    cols: u32,
    scale: f64
) {
    let r = rows as usize;
    let c = cols as usize;
    let masked_value = -1e9 as f32;
    let scale_f32 = scale as f32;

    for query in 0..c {
        if pad_mask[query] {
            for key in 0..r {
                data[key * c + query] = masked_value;
            }
            data[query * c + query] = 0.0;
            continue;
        }

        for key in 0..r {
            if pad_mask[key] || key > query {
                data[key * c + query] = masked_value;
            } else {
                data[key * c + query] *= scale_f32;
            }
        }
    }
}

fn mha_forward_head_into(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    pad_mask: &[bool],
    out_head: &mut [f32],
    attention_head: &mut [f32],
    total_cols: usize,
    head_idx: usize,
    head_units: usize,
    seq_len: usize,
    batch_size: usize,
    scale: f32,
) {
    let head_row_start = head_idx * head_units;

    for sample_idx in 0..batch_size {
        let sample_offset = sample_idx * seq_len;
        let attn_offset = sample_idx * seq_len * seq_len;
        let attn_block = &mut attention_head[attn_offset..attn_offset + seq_len * seq_len];

        for q_pos in 0..seq_len {
            let q_col = sample_offset + q_pos;
            if pad_mask[q_col] {
                for k_pos in 0..seq_len {
                    attn_block[k_pos * seq_len + q_pos] = 0.0;
                }
                for i in 0..head_units {
                    out_head[i * total_cols + q_col] = 0.0;
                }
                continue;
            }

            let mut max_score = f32::NEG_INFINITY;
            for k_pos in 0..seq_len {
                let k_col = sample_offset + k_pos;
                let idx = k_pos * seq_len + q_pos;
                if pad_mask[k_col] || k_pos > q_pos {
                    attn_block[idx] = f32::NEG_INFINITY;
                    continue;
                }

                let mut score = 0.0f32;
                for i in 0..head_units {
                    let row = head_row_start + i;
                    score += k_data[row * total_cols + k_col] * q_data[row * total_cols + q_col];
                }
                score *= scale;
                attn_block[idx] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            if !max_score.is_finite() {
                continue;
            }

            let mut sum_exp = 0.0f32;
            for k_pos in 0..seq_len {
                let idx = k_pos * seq_len + q_pos;
                let score = attn_block[idx];
                if !score.is_finite() {
                    attn_block[idx] = 0.0;
                    continue;
                }
                let exp_val = (score - max_score).exp();
                attn_block[idx] = exp_val;
                sum_exp += exp_val;
            }

            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                for k_pos in 0..seq_len {
                    attn_block[k_pos * seq_len + q_pos] = 0.0;
                }
                continue;
            }

            let inv_sum = 1.0f32 / sum_exp;
            for k_pos in 0..seq_len {
                let idx = k_pos * seq_len + q_pos;
                attn_block[idx] *= inv_sum;
            }

            for i in 0..head_units {
                let row = head_row_start + i;
                let out_idx = i * total_cols + q_col;
                let mut sum = 0.0f32;
                for k_pos in 0..seq_len {
                    let k_col = sample_offset + k_pos;
                    sum += v_data[row * total_cols + k_col] * attn_block[k_pos * seq_len + q_pos];
                }
                out_head[out_idx] = sum;
            }
        }
    }
}

fn mha_backward_head_into(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    attention_head: &[f32],
    d_out_data: &[f32],
    pad_mask: &[bool],
    d_q_head: &mut [f32],
    d_k_head: &mut [f32],
    d_v_head: &mut [f32],
    total_cols: usize,
    head_idx: usize,
    head_units: usize,
    seq_len: usize,
    batch_size: usize,
    scale: f32,
) {
    let head_row_start = head_idx * head_units;
    let mut err_attention = vec![0.0f32; seq_len];

    for sample_idx in 0..batch_size {
        let sample_offset = sample_idx * seq_len;
        let attn_offset = sample_idx * seq_len * seq_len;
        let attn_block = &attention_head[attn_offset..attn_offset + seq_len * seq_len];

        for q_pos in 0..seq_len {
            let q_col = sample_offset + q_pos;
            if pad_mask[q_col] {
                for i in 0..head_units {
                    d_q_head[i * total_cols + q_col] = 0.0;
                }
                continue;
            }

            err_attention.fill(0.0);

            for i in 0..head_units {
                let row = head_row_start + i;
                let d_out_val = d_out_data[row * total_cols + q_col];
                for k_pos in 0..seq_len {
                    let attn_idx = k_pos * seq_len + q_pos;
                    let k_col = sample_offset + k_pos;
                    d_v_head[i * total_cols + k_col] += d_out_val * attn_block[attn_idx];
                    err_attention[k_pos] += v_data[row * total_cols + k_col] * d_out_val;
                }
            }

            let mut dot = 0.0f32;
            for k_pos in 0..seq_len {
                let attn_idx = k_pos * seq_len + q_pos;
                dot += attn_block[attn_idx] * err_attention[k_pos];
            }

            for i in 0..head_units {
                let row = head_row_start + i;
                let q_val = q_data[row * total_cols + q_col];
                let mut dq_sum = 0.0f32;
                for k_pos in 0..seq_len {
                    let k_col = sample_offset + k_pos;
                    let attn_idx = k_pos * seq_len + q_pos;
                    let score_grad = attn_block[attn_idx] * (err_attention[k_pos] - dot) * scale;
                    dq_sum += k_data[row * total_cols + k_col] * score_grad;
                    d_k_head[i * total_cols + k_col] += q_val * score_grad;
                }
                d_q_head[i * total_cols + q_col] = dq_sum;
            }
        }
    }
}

#[napi]
pub fn multi_head_attention_forward_native_into(
    q_data: Float32Array,
    k_data: Float32Array,
    v_data: Float32Array,
    pad_mask: Vec<bool>,
    heads: u32,
    head_units: u32,
    seq_len: u32,
    batch_size: u32,
    scale: f64,
    mut out_data: Float32Array,
    mut attention_data: Float32Array,
) {
    let _h = heads as usize;
    let hu = head_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let scale_f32 = scale as f32;

    let q_slice = &*q_data;
    let k_slice = &*k_data;
    let v_slice = &*v_data;

    let out_slice = &mut *out_data;
    let attention_slice = &mut *attention_data;

    out_slice
        .par_chunks_mut(hu * total_cols)
        .zip(attention_slice.par_chunks_mut(bs * sl * sl))
        .enumerate()
        .for_each(|(head_idx, (out_head, attention_head))| {
            mha_forward_head_into(
                q_slice,
                k_slice,
                v_slice,
                &pad_mask,
                out_head,
                attention_head,
                total_cols,
                head_idx,
                hu,
                sl,
                bs,
                scale_f32,
            );
        });
}

#[napi]
pub fn multi_head_attention_backward_native_into(
    q_data: Float32Array,
    k_data: Float32Array,
    v_data: Float32Array,
    attention_data: Float32Array,
    d_out_data: Float32Array,
    pad_mask: Vec<bool>,
    heads: u32,
    head_units: u32,
    seq_len: u32,
    batch_size: u32,
    scale: f64,
    mut d_q_out: Float32Array,
    mut d_k_out: Float32Array,
    mut d_v_out: Float32Array,
) {
    let _h = heads as usize;
    let hu = head_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let scale_f32 = scale as f32;

    let q_slice = &*q_data;
    let k_slice = &*k_data;
    let v_slice = &*v_data;
    let d_out_slice = &*d_out_data;
    let attn_slice = &*attention_data;

    let d_q_slice = &mut *d_q_out;
    let d_k_slice = &mut *d_k_out;
    let d_v_slice = &mut *d_v_out;

    d_q_slice
        .par_chunks_mut(hu * total_cols)
        .zip(d_k_slice.par_chunks_mut(hu * total_cols))
        .zip(d_v_slice.par_chunks_mut(hu * total_cols))
        .zip(attn_slice.par_chunks(bs * sl * sl))
        .enumerate()
        .for_each(|(head_idx, (((d_q_head, d_k_head), d_v_head), attention_head))| {
            d_q_head.fill(0.0);
            d_k_head.fill(0.0);
            d_v_head.fill(0.0);
            mha_backward_head_into(
                q_slice,
                k_slice,
                v_slice,
                attention_head,
                d_out_slice,
                &pad_mask,
                d_q_head,
                d_k_head,
                d_v_head,
                total_cols,
                head_idx,
                hu,
                sl,
                bs,
                scale_f32,
            );
        });
}
