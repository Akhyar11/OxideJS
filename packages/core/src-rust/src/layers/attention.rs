use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use crate::math::{SafeRawPtr, SafeRawPtrMut};

const ATTENTION_MASK_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[napi]
pub fn apply_attention_mask_native(
    mut data: Float32Array,
    pad_mask: Vec<bool>,
    rows: u32,
    cols: u32,
    scale: f64,
) {
    let r = rows as usize;
    let c = cols as usize;
    let masked_value = -1e9 as f32;
    let scale_f32 = scale as f32;
    let data_slice = &mut *data;

    if data_slice.len() < ATTENTION_MASK_PARALLEL_THRESHOLD {
        for key in 0..r {
            let row = &mut data_slice[key * c..(key + 1) * c];
            let key_is_masked = pad_mask[key];
            for query in 0..c {
                if pad_mask[query] {
                    row[query] = if key == query { 0.0 } else { masked_value };
                } else if key_is_masked || key > query {
                    row[query] = masked_value;
                } else {
                    row[query] *= scale_f32;
                }
            }
        }
    } else {
        data_slice
            .par_chunks_mut(c)
            .enumerate()
            .for_each(|(key, row)| {
                let key_is_masked = pad_mask[key];
                for query in 0..c {
                    if pad_mask[query] {
                        row[query] = if key == query { 0.0 } else { masked_value };
                    } else if key_is_masked || key > query {
                        row[query] = masked_value;
                    } else {
                        row[query] *= scale_f32;
                    }
                }
            });
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

    // Pre-allocate tile buffers for K and V data (contiguous per-sample layout)
    // tile layout: [head_units][seq_len] — row-major per head_unit
    let tile_size = head_units * seq_len;
    let mut k_tile = vec![0.0f32; tile_size];
    let mut v_tile = vec![0.0f32; tile_size];

    for sample_idx in 0..batch_size {
        let sample_offset = sample_idx * seq_len;
        let attn_offset = sample_idx * seq_len * seq_len;
        let attn_block = &mut attention_head[attn_offset..attn_offset + seq_len * seq_len];

        // Pre-gather K and V for this sample into contiguous tiles
        for i in 0..head_units {
            let row = head_row_start + i;
            let src_base = row * total_cols + sample_offset;
            let dst_base = i * seq_len;
            k_tile[dst_base..dst_base + seq_len]
                .copy_from_slice(&k_data[src_base..src_base + seq_len]);
            v_tile[dst_base..dst_base + seq_len]
                .copy_from_slice(&v_data[src_base..src_base + seq_len]);
        }

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
                let k_col_local = k_pos; // local index into tile
                let idx = k_pos * seq_len + q_pos;
                if pad_mask[sample_offset + k_pos] || k_pos > q_pos {
                    attn_block[idx] = f32::NEG_INFINITY;
                    continue;
                }

                let mut score = 0.0f32;
                for i in 0..head_units {
                    let row = head_row_start + i;
                    // K from tile (contiguous), Q from original (strided but same q_col)
                    score += k_tile[i * seq_len + k_col_local] * q_data[row * total_cols + q_col];
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

            // Compute weighted sum of V using tile (contiguous reads)
            for i in 0..head_units {
                let out_idx = i * total_cols + q_col;
                let mut sum = 0.0f32;
                let v_row = &v_tile[i * seq_len..i * seq_len + seq_len];
                for k_pos in 0..seq_len {
                    sum += v_row[k_pos] * attn_block[k_pos * seq_len + q_pos];
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

    // Pre-allocate tiles for cache-friendly access
    let tile_size = head_units * seq_len;
    let mut k_tile = vec![0.0f32; tile_size];
    let mut v_tile = vec![0.0f32; tile_size];
    let mut dout_tile = vec![0.0f32; tile_size];
    let mut q_tile = vec![0.0f32; tile_size];

    for sample_idx in 0..batch_size {
        let sample_offset = sample_idx * seq_len;
        let attn_offset = sample_idx * seq_len * seq_len;
        let attn_block = &attention_head[attn_offset..attn_offset + seq_len * seq_len];

        // Pre-gather K, V, Q and dOut for this sample into contiguous tiles
        for i in 0..head_units {
            let row = head_row_start + i;
            let src_base = row * total_cols + sample_offset;
            let dst_base = i * seq_len;
            k_tile[dst_base..dst_base + seq_len].copy_from_slice(&k_data[src_base..src_base + seq_len]);
            v_tile[dst_base..dst_base + seq_len].copy_from_slice(&v_data[src_base..src_base + seq_len]);
            dout_tile[dst_base..dst_base + seq_len].copy_from_slice(&d_out_data[src_base..src_base + seq_len]);
            q_tile[dst_base..dst_base + seq_len].copy_from_slice(&q_data[src_base..src_base + seq_len]);
        }

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
                let d_out_val = dout_tile[i * seq_len + q_pos];
                let v_row = &v_tile[i * seq_len..i * seq_len + seq_len];
                let dv_row_base = i * total_cols + sample_offset;
                
                for k_pos in 0..seq_len {
                    let attn_idx = k_pos * seq_len + q_pos;
                    let attn_val = attn_block[attn_idx];
                    d_v_head[dv_row_base + k_pos] += d_out_val * attn_val;
                    err_attention[k_pos] += v_row[k_pos] * d_out_val;
                }
            }

            let mut dot = 0.0f32;
            for k_pos in 0..seq_len {
                dot += attn_block[k_pos * seq_len + q_pos] * err_attention[k_pos];
            }

            for i in 0..head_units {
                let q_val = q_tile[i * seq_len + q_pos];
                let k_row = &k_tile[i * seq_len..i * seq_len + seq_len];
                let dk_row_base = i * total_cols + sample_offset;
                let mut dq_sum = 0.0f32;

                for k_pos in 0..seq_len {
                    let attn_idx = k_pos * seq_len + q_pos;
                    let score_grad = attn_block[attn_idx] * (err_attention[k_pos] - dot) * scale;
                    dq_sum += k_row[k_pos] * score_grad;
                    d_k_head[dk_row_base + k_pos] += q_val * score_grad;
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

    let q_len = q_data.len();
    let k_len = k_data.len();
    let v_len = v_data.len();
    let out_len = out_data.len();
    let attn_len = attention_data.len();

    let q_p = SafeRawPtr(q_data.as_ptr() as usize);
    let k_p = SafeRawPtr(k_data.as_ptr() as usize);
    let v_p = SafeRawPtr(v_data.as_ptr() as usize);

    let out_p = SafeRawPtrMut(out_data.as_ptr() as usize);
    let attn_p = SafeRawPtrMut(attention_data.as_ptr() as usize);

    (0.._h).into_par_iter().for_each(|head_idx| {
        unsafe {
            let q_ptr = std::slice::from_raw_parts(q_p.0 as *const f32, q_len);
            let k_ptr = std::slice::from_raw_parts(k_p.0 as *const f32, k_len);
            let v_ptr = std::slice::from_raw_parts(v_p.0 as *const f32, v_len);
            
            let out_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, out_len);
            let attn_ptr = std::slice::from_raw_parts_mut(attn_p.0 as *mut f32, attn_len);

            let head_out_offset = head_idx * hu * total_cols;
            let head_attn_offset = head_idx * bs * sl * sl;
            
            mha_forward_head_into(
                q_ptr,
                k_ptr,
                v_ptr,
                &pad_mask,
                &mut out_ptr[head_out_offset..head_out_offset + hu * total_cols],
                &mut attn_ptr[head_attn_offset..head_attn_offset + bs * sl * sl],
                total_cols,
                head_idx,
                hu,
                sl,
                bs,
                scale_f32,
            );
        }
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

    let q_len = q_data.len();
    let k_len = k_data.len();
    let v_len = v_data.len();
    let dout_len = d_out_data.len();
    let attn_len = attention_data.len();
    let dq_len = d_q_out.len();
    let dk_len = d_k_out.len();
    let dv_len = d_v_out.len();

    let q_p = SafeRawPtr(q_data.as_ptr() as usize);
    let k_p = SafeRawPtr(k_data.as_ptr() as usize);
    let v_p = SafeRawPtr(v_data.as_ptr() as usize);
    let d_out_p = SafeRawPtr(d_out_data.as_ptr() as usize);
    let attn_p = SafeRawPtr(attention_data.as_ptr() as usize);

    let dq_p = SafeRawPtrMut(d_q_out.as_ptr() as usize);
    let dk_p = SafeRawPtrMut(d_k_out.as_ptr() as usize);
    let dv_p = SafeRawPtrMut(d_v_out.as_ptr() as usize);

    (0.._h).into_par_iter().for_each(|head_idx| {
        unsafe {
            let q_ptr = std::slice::from_raw_parts(q_p.0 as *const f32, q_len);
            let k_ptr = std::slice::from_raw_parts(k_p.0 as *const f32, k_len);
            let v_ptr = std::slice::from_raw_parts(v_p.0 as *const f32, v_len);
            let dout_ptr = std::slice::from_raw_parts(d_out_p.0 as *const f32, dout_len);
            let attn_ptr = std::slice::from_raw_parts(attn_p.0 as *const f32, attn_len);

            let dq_ptr = std::slice::from_raw_parts_mut(dq_p.0 as *mut f32, dq_len);
            let dk_ptr = std::slice::from_raw_parts_mut(dk_p.0 as *mut f32, dk_len);
            let dv_ptr = std::slice::from_raw_parts_mut(dv_p.0 as *mut f32, dv_len);

            let head_offset = head_idx * hu * total_cols;
            let head_attn_offset = head_idx * bs * sl * sl;

            let d_q_head = &mut dq_ptr[head_offset..head_offset + hu * total_cols];
            let d_k_head = &mut dk_ptr[head_offset..head_offset + hu * total_cols];
            let d_v_head = &mut dv_ptr[head_offset..head_offset + hu * total_cols];

            d_q_head.fill(0.0);
            d_k_head.fill(0.0);
            d_v_head.fill(0.0);

            mha_backward_head_into(
                q_ptr,
                k_ptr,
                v_ptr,
                &attn_ptr[head_attn_offset..head_attn_offset + bs * sl * sl],
                dout_ptr,
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
        }
    });
}
