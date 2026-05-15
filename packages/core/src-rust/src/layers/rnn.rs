use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use crate::math::{SafeRawPtr, SafeRawPtrMut};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers shared by AdaptiveMemoryRNN
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn stable_softmax(scores: &[f32], attention: &mut [f32]) {
    let n = scores.len();
    let mut max_v = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max_v {
            max_v = s;
        }
    }
    let mut denom = 0.0f32;
    for i in 0..n {
        let v = (scores[i] - max_v).exp();
        attention[i] = v;
        denom += v;
    }
    if denom == 0.0 || !denom.is_finite() {
        let u = 1.0 / n as f32;
        for i in 0..n {
            attention[i] = u;
        }
    } else {
        for i in 0..n {
            attention[i] /= denom;
        }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[napi]
pub fn lstm_forward_native_into(
    wxi: Float32Array,
    wxf: Float32Array,
    wxo: Float32Array,
    wxg: Float32Array,
    whi: Float32Array,
    whf: Float32Array,
    who: Float32Array,
    whg: Float32Array,
    bi: Float32Array,
    bf: Float32Array,
    bo: Float32Array,
    bg: Float32Array,
    x_seq: Float32Array,
    h0: Float32Array,
    c0: Float32Array,
    hidden_units: u32,
    input_units: u32,
    seq_len: u32,
    batch_size: u32,
    mut h_seq: Float32Array,
    mut c_seq: Float32Array,
    mut gi_seq: Float32Array,
    mut gf_seq: Float32Array,
    mut go_seq: Float32Array,
    mut gg_seq: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;

    let mut proj = vec![0.0f32; 4 * hu * total_cols];
    
    // Parallel projection: Wx * x + b
    let wxi_p = SafeRawPtr(wxi.as_ptr() as usize);
    let wxf_p = SafeRawPtr(wxf.as_ptr() as usize);
    let wxo_p = SafeRawPtr(wxo.as_ptr() as usize);
    let wxg_p = SafeRawPtr(wxg.as_ptr() as usize);
    let bi_p = SafeRawPtr(bi.as_ptr() as usize);
    let bf_p = SafeRawPtr(bf.as_ptr() as usize);
    let bo_p = SafeRawPtr(bo.as_ptr() as usize);
    let bg_p = SafeRawPtr(bg.as_ptr() as usize);
    let x_p = SafeRawPtr(x_seq.as_ptr() as usize);

    proj.par_chunks_mut(hu * total_cols).enumerate().for_each(move |(g, p_chunk)| {
        unsafe {
            let wx_addr = match g {
                0 => wxi_p.0, 1 => wxf_p.0, 2 => wxo_p.0, _ => wxg_p.0
            } as *const f32;
            let b_addr = match g {
                0 => bi_p.0, 1 => bf_p.0, 2 => bo_p.0, _ => bg_p.0
            } as *const f32;
            let x_seq_s = std::slice::from_raw_parts(x_p.0 as *const f32, iu * total_cols);
            
            for r in 0..hu {
                let wx_off = r * iu;
                let b_val = *b_addr.add(r);
                let p_row_start = r * total_cols;
                let p_row_end = p_row_start + total_cols;
                
                // Initialize with bias
                for i in p_row_start..p_row_end {
                    p_chunk[i] = b_val;
                }

                for c in 0..iu {
                    let wx_v = *wx_addr.add(wx_off + c);
                    let x_off = c * total_cols;
                    for t_bs in 0..total_cols {
                        p_chunk[p_row_start + t_bs] += wx_v * x_seq_s[x_off + t_bs];
                    }
                }
            }
        }
    });

    // Initial state
    for b in 0..bs {
        for i in 0..hu {
            h_seq[i * bs + b] = h0[i * bs + b];
            c_seq[i * bs + b] = c0[i * bs + b];
        }
    }

    // Sequence recurrence (batched)
    let h_p = SafeRawPtrMut(h_seq.as_mut_ptr() as usize);
    let c_p = SafeRawPtrMut(c_seq.as_mut_ptr() as usize);
    let gi_p = SafeRawPtrMut(gi_seq.as_mut_ptr() as usize);
    let gf_p = SafeRawPtrMut(gf_seq.as_mut_ptr() as usize);
    let go_p = SafeRawPtrMut(go_seq.as_mut_ptr() as usize);
    let gg_p = SafeRawPtrMut(gg_seq.as_mut_ptr() as usize);
    
    let whi_p = SafeRawPtr(whi.as_ptr() as usize);
    let whf_p = SafeRawPtr(whf.as_ptr() as usize);
    let who_p = SafeRawPtr(who.as_ptr() as usize);
    let whg_p = SafeRawPtr(whg.as_ptr() as usize);
    let proj_p = SafeRawPtr(proj.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(move |b_idx| unsafe {
        let h_ptr = h_p.0 as *mut f32;
        let c_ptr = c_p.0 as *mut f32;
        let gi_ptr = gi_p.0 as *mut f32;
        let gf_ptr = gf_p.0 as *mut f32;
        let go_ptr = go_p.0 as *mut f32;
        let gg_ptr = gg_p.0 as *mut f32;
        
        let whi_s = std::slice::from_raw_parts(whi_p.0 as *const f32, hu * hu);
        let whf_s = std::slice::from_raw_parts(whf_p.0 as *const f32, hu * hu);
        let who_s = std::slice::from_raw_parts(who_p.0 as *const f32, hu * hu);
        let whg_s = std::slice::from_raw_parts(whg_p.0 as *const f32, hu * hu);
        let proj_s = std::slice::from_raw_parts(proj_p.0 as *const f32, 4 * hu * total_cols);

        for t in 0..sl {
            let prev = t * bs * hu + b_idx;
            let curr = (t + 1) * bs * hu + b_idx;
            let gate = t * bs * hu + b_idx;
            let t_bs = t * bs + b_idx;

            for r in 0..hu {
                let mut ip = proj_s[0 * hu * total_cols + r * total_cols + t_bs];
                let mut fp = proj_s[1 * hu * total_cols + r * total_cols + t_bs];
                let mut op = proj_s[2 * hu * total_cols + r * total_cols + t_bs];
                let mut gp = proj_s[3 * hu * total_cols + r * total_cols + t_bs];
                
                let r_hu = r * hu;
                for c in 0..hu {
                    let hv = *h_ptr.add(prev + c * bs);
                    ip += whi_s[r_hu + c] * hv;
                    fp += whf_s[r_hu + c] * hv;
                    op += who_s[r_hu + c] * hv;
                    gp += whg_s[r_hu + c] * hv;
                }

                let iv = sigmoid(ip);
                let fv = sigmoid(fp);
                let ov = sigmoid(op);
                let gv = gp.tanh();

                let cp = *c_ptr.add(prev + r * bs);
                let cc = fv * cp + iv * gv;
                let hc = ov * cc.tanh();

                *gi_ptr.add(gate + r * bs) = iv;
                *gf_ptr.add(gate + r * bs) = fv;
                *go_ptr.add(gate + r * bs) = ov;
                *gg_ptr.add(gate + r * bs) = gv;
                *c_ptr.add(curr + r * bs) = cc;
                *h_ptr.add(curr + r * bs) = hc;
            }
        }
    });
}

#[napi]
pub fn rnn_forward_native_into(
    wxh: Float32Array,
    whh: Float32Array,
    bias: Float32Array,
    x_seq: Float32Array,
    h0: Float32Array,
    hidden_units: u32,
    input_units: u32,
    seq_len: u32,
    batch_size: u32,
    mut h_seq: Float32Array,
    mut act_grad: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;

    let mut proj = vec![0.0f32; hu * total_cols];
    
    let wx_p = SafeRawPtr(wxh.as_ptr() as usize);
    let b_p = SafeRawPtr(bias.as_ptr() as usize);
    let x_p = SafeRawPtr(x_seq.as_ptr() as usize);

    proj.par_chunks_mut(total_cols).enumerate().for_each(move |(r, p_row)| {
        unsafe {
            let wx_addr = wx_p.0 as *const f32;
            let b_addr = b_p.0 as *const f32;
            let x_addr = x_p.0 as *const f32;
            
            let b_val = *b_addr.add(r);
            let wx_off = r * iu;

            // Initialize with bias
            for val in p_row.iter_mut() {
                *val = b_val;
            }

            // Reordered loop for SIMD friendliness: 
            // Wx[r, c] * x[c, t_bs] -> sequential access on t_bs
            for c in 0..iu {
                let wx_v = *wx_addr.add(wx_off + c);
                let x_row = c * total_cols;
                
                // This inner loop is now perfectly sequential and auto-vectorizable
                for t_bs in 0..total_cols {
                    p_row[t_bs] += wx_v * *x_addr.add(x_row + t_bs);
                }
            }
        }
    });

    // Initial state
    for i in 0..(hu * bs) {
        h_seq[i] = h0[i];
    }

    let h_p = SafeRawPtrMut(h_seq.as_mut_ptr() as usize);
    let d_p = SafeRawPtrMut(act_grad.as_mut_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let proj_p = SafeRawPtr(proj.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(move |b_idx| unsafe {
        let h_ptr = h_p.0 as *mut f32;
        let d_ptr = d_p.0 as *mut f32;
        let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
        let proj_s = std::slice::from_raw_parts(proj_p.0 as *const f32, hu * total_cols);
        for t in 0..sl {
            let prev = t * bs * hu + b_idx;
            let curr = (t + 1) * bs * hu + b_idx;
            let t_bs = t * bs + b_idx;
            for r in 0..hu {
                let mut s = proj_s[r * total_cols + t_bs];
                let wh_off = r * hu;
                for c in 0..hu {
                    s += whh_s[wh_off + c] * *h_ptr.add(prev + c * bs);
                }
                let hv = s.tanh();
                *h_ptr.add(curr + r * bs) = hv;
                *d_ptr.add(prev + r * bs) = 1.0 - hv * hv;
            }
        }
    });
}

#[napi]
pub fn rnn_backward_native_into(
    wxh: Float32Array,
    whh: Float32Array,
    x_seq: Float32Array,
    h_seq: Float32Array,
    d_act: Float32Array,
    err_h: Float32Array,
    hidden_units: u32,
    input_units: u32,
    seq_len: u32,
    batch_size: u32,
    mut dwxh: Float32Array,
    mut dwhh: Float32Array,
    mut dbh: Float32Array,
    mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;

    let mut dz_all = vec![0.0f32; sl * bs * hu];
    
    // 1. Calculate dz and dx (parallel across batch)
    let dx_p = SafeRawPtrMut(dx_out.as_mut_ptr() as usize);
    let dz_p = SafeRawPtrMut(dz_all.as_mut_ptr() as usize);
    let err_h_p = SafeRawPtr(err_h.as_ptr() as usize);
    let d_act_p = SafeRawPtr(d_act.as_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let wxh_p = SafeRawPtr(wxh.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(move |b_idx| unsafe {
        let dxb = dx_p.0 as *mut f32;
        let dzb = dz_p.0 as *mut f32;
        let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
        let wxh_s = std::slice::from_raw_parts(wxh_p.0 as *const f32, hu * iu);
        let err_h_s = std::slice::from_raw_parts(err_h_p.0 as *const f32, sl * bs * hu);
        let d_act_s = std::slice::from_raw_parts(d_act_p.0 as *const f32, sl * bs * hu);

        let mut dhn = vec![0.0f32; hu];
        let mut ndh = vec![0.0f32; hu];

        for ti in 0..sl {
            let t = sl - 1 - ti;
            let gate = t * bs * hu + b_idx;
            for r in 0..hu {
                let dz = (err_h_s[gate + r * bs] + dhn[r]) * d_act_s[gate + r * bs];
                *dzb.add(gate + r * bs) = dz;
            }
            for r in 0..hu {
                let mut s = 0.0f32;
                for k in 0..hu {
                    s += whh_s[k * hu + r] * *dzb.add(gate + k * bs);
                }
                ndh[r] = s;
            }
            std::mem::swap(&mut dhn, &mut ndh);
            for j in 0..iu {
                let mut s = 0.0f32;
                for k in 0..hu {
                    s += wxh_s[k * iu + j] * *dzb.add(gate + k * bs);
                }
                *dxb.add(j * total_cols + t * bs + b_idx) = s;
            }
        }
    });

    // 2. Accumulate weight gradients (Thread-Local + Reduce)
    let x_p = SafeRawPtr(x_seq.as_ptr() as usize);
    let h_p = SafeRawPtr(h_seq.as_ptr() as usize);
    let dz_all_p = SafeRawPtr(dz_all.as_ptr() as usize);

    let grads = (0..bs).into_par_iter().map(move |b_idx| unsafe {
        let mut local_dwx = vec![0.0f32; hu * iu];
        let mut local_dwh = vec![0.0f32; hu * hu];
        let mut local_dbh = vec![0.0f32; hu];
        
        let x_s = std::slice::from_raw_parts(x_p.0 as *const f32, iu * total_cols);
        let h_s = std::slice::from_raw_parts(h_p.0 as *const f32, (sl + 1) * bs * hu);
        let dz_s = std::slice::from_raw_parts(dz_all_p.0 as *const f32, sl * bs * hu);

        for t in 0..sl {
            let gate_off = t * bs * hu + b_idx;
            let h_off = t * bs * hu + b_idx;
            let x_off = t * bs + b_idx;

            for r in 0..hu {
                let dv = dz_s[gate_off + r * bs];
                local_dbh[r] += dv;
                let r_iu = r * iu;
                for c in 0..iu {
                    local_dwx[r_iu + c] += dv * x_s[c * total_cols + x_off];
                }
                let r_hu = r * hu;
                for c in 0..hu {
                    local_dwh[r_hu + c] += dv * h_s[h_off + c * bs];
                }
            }
        }
        (local_dwx, local_dwh, local_dbh)
    }).reduce(|| (vec![0.0f32; hu * iu], vec![0.0f32; hu * hu], vec![0.0f32; hu]), 
    |mut a, b| {
        for i in 0..a.0.len() { a.0[i] += b.0[i]; }
        for i in 0..a.1.len() { a.1[i] += b.1[i]; }
        for i in 0..a.2.len() { a.2[i] += b.2[i]; }
        a
    });

    dwxh.copy_from_slice(&grads.0);
    dwhh.copy_from_slice(&grads.1);
    dbh.copy_from_slice(&grads.2);
}

#[napi]
pub fn gru_forward_native_into(
    wxr: Float32Array,
    whr: Float32Array,
    br: Float32Array,
    wxz: Float32Array,
    whz: Float32Array,
    bz: Float32Array,
    wxh: Float32Array,
    whh: Float32Array,
    bh: Float32Array,
    x_seq: Float32Array,
    h0: Float32Array,
    hidden_units: u32,
    input_units: u32,
    seq_len: u32,
    batch_size: u32,
    mut h_seq: Float32Array,
    mut r_seq: Float32Array,
    mut z_seq: Float32Array,
    mut n_seq: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let mut proj = vec![0.0f32; 3 * hu * total_cols];
    let wxr_p = SafeRawPtr(wxr.as_ptr() as usize);
    let wxz_p = SafeRawPtr(wxz.as_ptr() as usize);
    let wxh_p = SafeRawPtr(wxh.as_ptr() as usize);
    let br_p = SafeRawPtr(br.as_ptr() as usize);
    let bz_p = SafeRawPtr(bz.as_ptr() as usize);
    let bh_p = SafeRawPtr(bh.as_ptr() as usize);
    let x_p = SafeRawPtr(x_seq.as_ptr() as usize);

    proj.par_chunks_mut(hu * total_cols).enumerate().for_each(move |(g, p_chunk)| {
        unsafe {
            let wx_addr = match g { 0 => wxr_p.0, 1 => wxz_p.0, _ => wxh_p.0 } as *const f32;
            let b_addr = match g { 0 => br_p.0, 1 => bz_p.0, _ => bh_p.0 } as *const f32;
            let x_addr = x_p.0 as *const f32;
            
            for r in 0..hu {
                let wx_off = r * iu;
                let b_val = *b_addr.add(r);
                let p_row_start = r * total_cols;
                let p_row_end = p_row_start + total_cols;

                // Initialize with bias
                for i in p_row_start..p_row_end {
                    p_chunk[i] = b_val;
                }

                for c in 0..iu {
                    let wx_v = *wx_addr.add(wx_off + c);
                    let x_off = c * total_cols;
                    for t_bs in 0..total_cols {
                        p_chunk[p_row_start + t_bs] += wx_v * *x_addr.add(x_off + t_bs);
                    }
                }
            }
        }
    });

    for i in 0..(hu * bs) { h_seq[i] = h0[i]; }

    let h_p = SafeRawPtrMut(h_seq.as_mut_ptr() as usize);
    let r_p = SafeRawPtrMut(r_seq.as_mut_ptr() as usize);
    let z_p = SafeRawPtrMut(z_seq.as_mut_ptr() as usize);
    let n_p = SafeRawPtrMut(n_seq.as_mut_ptr() as usize);
    
    let whr_p = SafeRawPtr(whr.as_ptr() as usize);
    let whz_p = SafeRawPtr(whz.as_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let proj_p = SafeRawPtr(proj.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(move |b_idx| unsafe {
        let hb = h_p.0 as *mut f32;
        let rb = r_p.0 as *mut f32;
        let zb = z_p.0 as *mut f32;
        let nb = n_p.0 as *mut f32;
        
        let whr_s = std::slice::from_raw_parts(whr_p.0 as *const f32, hu * hu);
        let whz_s = std::slice::from_raw_parts(whz_p.0 as *const f32, hu * hu);
        let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
        let proj_s = std::slice::from_raw_parts(proj_p.0 as *const f32, 3 * hu * total_cols);

        for t in 0..sl {
            let pre = t * bs * hu + b_idx;
            let cur = (t + 1) * bs * hu + b_idx;
            let t_bs = t * bs + b_idx;
            let gate = t * bs * hu + b_idx;

            for r in 0..hu {
                let mut rp = proj_s[0 * hu * total_cols + r * total_cols + t_bs];
                let mut zp = proj_s[1 * hu * total_cols + r * total_cols + t_bs];
                let r_hu = r * hu;
                for c in 0..hu {
                    let hv = *hb.add(pre + c * bs);
                    rp += whr_s[r_hu + c] * hv;
                    zp += whz_s[r_hu + c] * hv;
                }
                let rv = sigmoid(rp);
                let zv = sigmoid(zp);
                *rb.add(gate + r * bs) = rv;
                *zb.add(gate + r * bs) = zv;

                let mut np = proj_s[2 * hu * total_cols + r * total_cols + t_bs];
                for c in 0..hu {
                    np += whh_s[r_hu + c] * (*rb.add(gate + c * bs)) * *hb.add(pre + c * bs);
                }
                let nv = np.tanh();
                *nb.add(gate + r * bs) = nv;
                let hpv = *hb.add(pre + r * bs);
                *hb.add(cur + r * bs) = (1.0 - zv) * nv + zv * hpv;
            }
        }
    });
}

#[napi]
pub fn gru_backward_native_into(
    wxr: Float32Array,
    whr: Float32Array,
    wxz: Float32Array,
    whz: Float32Array,
    wxh: Float32Array,
    whh: Float32Array,
    x_seq: Float32Array,
    h_seq: Float32Array,
    r_seq: Float32Array,
    z_seq: Float32Array,
    n_seq: Float32Array,
    err_h: Float32Array,
    hidden_units: u32,
    input_units: u32,
    seq_len: u32,
    batch_size: u32,
    mut dwxr: Float32Array,
    mut dwhr: Float32Array,
    mut dbr: Float32Array,
    mut dwxz: Float32Array,
    mut dwhz: Float32Array,
    mut dbz: Float32Array,
    mut dwxh: Float32Array,
    mut dwhh: Float32Array,
    mut dbh: Float32Array,
    mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let mut dz_all = vec![0.0f32; sl * bs * 3 * hu];
    let dz_p = SafeRawPtrMut(dz_all.as_mut_ptr() as usize);
    let dx_p = SafeRawPtrMut(dx_out.as_mut_ptr() as usize);
    let h_p = SafeRawPtr(h_seq.as_ptr() as usize);
    let r_p = SafeRawPtr(r_seq.as_ptr() as usize);
    let z_p = SafeRawPtr(z_seq.as_ptr() as usize);
    let n_p = SafeRawPtr(n_seq.as_ptr() as usize);
    let err_h_p = SafeRawPtr(err_h.as_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let whr_p = SafeRawPtr(whr.as_ptr() as usize);
    let whz_p = SafeRawPtr(whz.as_ptr() as usize);
    let wxr_p = SafeRawPtr(wxr.as_ptr() as usize);
    let wxz_p = SafeRawPtr(wxz.as_ptr() as usize);
    let wxh_p = SafeRawPtr(wxh.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(move |b_idx| unsafe {
        let dzb = dz_p.0 as *mut f32;
        let dxb = dx_p.0 as *mut f32;
        
        let h_s = std::slice::from_raw_parts(h_p.0 as *const f32, (sl + 1) * bs * hu);
        let r_s = std::slice::from_raw_parts(r_p.0 as *const f32, sl * bs * hu);
        let z_s = std::slice::from_raw_parts(z_p.0 as *const f32, sl * bs * hu);
        let n_s = std::slice::from_raw_parts(n_p.0 as *const f32, sl * bs * hu);
        let err_h_s = std::slice::from_raw_parts(err_h_p.0 as *const f32, sl * bs * hu);
        
        let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
        let whr_s = std::slice::from_raw_parts(whr_p.0 as *const f32, hu * hu);
        let whz_s = std::slice::from_raw_parts(whz_p.0 as *const f32, hu * hu);
        let wxr_s = std::slice::from_raw_parts(wxr_p.0 as *const f32, hu * iu);
        let wxz_s = std::slice::from_raw_parts(wxz_p.0 as *const f32, hu * iu);
        let wxh_s = std::slice::from_raw_parts(wxh_p.0 as *const f32, hu * iu);

        let mut dhn = vec![0.0f32; hu];
        let mut dhp = vec![0.0f32; hu];

        for ti in 0..sl {
            let t = sl - 1 - ti;
            let gate = t * bs * hu + b_idx;
            let pre = t * bs * hu + b_idx;
            let dzts = t * 3 * hu * bs;

            for row in 0..hu {
                let dh = err_h_s[gate + row * bs] + dhn[row];
                let zv = z_s[gate + row * bs];
                let nv = n_s[gate + row * bs];
                let hpv = h_s[pre + row * bs];
                let dn = dh * (1.0 - zv);
                let dze = dh * (hpv - nv);
                let dnp = dn * (1.0 - nv * nv);
                let dzp = dze * zv * (1.0 - zv);
                *dzb.add(dzts + 2 * hu * bs + row * bs + b_idx) = dnp;
                *dzb.add(dzts + 1 * hu * bs + row * bs + b_idx) = dzp;
            }

            for j in 0..hu {
                let mut dr_aux = 0.0f32;
                let dzts_cand = dzts + 2 * hu * bs;
                for i in 0..hu {
                    dr_aux += whh_s[i * hu + j] * *dzb.add(dzts_cand + i * bs + b_idx);
                }
                let dr_val = r_s[gate + j * bs];
                *dzb.add(dzts + 0 * hu * bs + j * bs + b_idx) = dr_aux * h_s[pre + j * bs] * dr_val * (1.0 - dr_val);
            }

            for row in 0..hu {
                let mut s = dhn[row] * z_s[gate + row * bs];
                for k in 0..hu {
                    s += whr_s[k * hu + row] * *dzb.add(dzts + 0 * hu * bs + k * bs + b_idx);
                    s += whz_s[k * hu + row] * *dzb.add(dzts + 1 * hu * bs + k * bs + b_idx);
                    s += whh_s[k * hu + row] * *dzb.add(dzts + 2 * hu * bs + k * bs + b_idx) * r_s[gate + row * bs];
                }
                dhp[row] = s;
            }
            std::mem::swap(&mut dhn, &mut dhp);

            for j in 0..iu {
                let mut s = 0.0f32;
                for k in 0..hu {
                    s += wxr_s[k * iu + j] * *dzb.add(dzts + 0 * hu * bs + k * bs + b_idx);
                    s += wxz_s[k * iu + j] * *dzb.add(dzts + 1 * hu * bs + k * bs + b_idx);
                    s += wxh_s[k * iu + j] * *dzb.add(dzts + 2 * hu * bs + k * bs + b_idx);
                }
                *dxb.add(j * total_cols + t * bs + b_idx) = s;
            }
        }
    });

    // 2. Accumulate weight gradients (Thread-Local + Reduce)
    let dz_p = SafeRawPtr(dz_all.as_ptr() as usize);
    let x_p = SafeRawPtr(x_seq.as_ptr() as usize);
    let h_p = SafeRawPtr(h_seq.as_ptr() as usize);
    let r_p = SafeRawPtr(r_seq.as_ptr() as usize);

    let grads = (0..bs).into_par_iter().map(move |b_idx| unsafe {
        let mut l_dwxr = vec![0.0f32; hu * iu];
        let mut l_dwhr = vec![0.0f32; hu * hu];
        let mut l_dbr = vec![0.0f32; hu];
        let mut l_dwxz = vec![0.0f32; hu * iu];
        let mut l_dwhz = vec![0.0f32; hu * hu];
        let mut l_dbz = vec![0.0f32; hu];
        let mut l_dwxh = vec![0.0f32; hu * iu];
        let mut l_dwhh = vec![0.0f32; hu * hu];
        let mut l_dbh = vec![0.0f32; hu];
        
        let dz_s = std::slice::from_raw_parts(dz_p.0 as *const f32, sl * bs * 3 * hu);
        let x_s = std::slice::from_raw_parts(x_p.0 as *const f32, iu * total_cols);
        let h_s = std::slice::from_raw_parts(h_p.0 as *const f32, (sl + 1) * bs * hu);
        let r_s = std::slice::from_raw_parts(r_p.0 as *const f32, sl * bs * hu);

        for t in 0..sl {
            let dzts = t * 3 * hu * bs;
            let hoff = t * bs * hu + b_idx;
            let xoff = t * bs + b_idx;
            for r in 0..hu {
                let dzr = dz_s[dzts + 0 * hu * bs + r * bs + b_idx];
                let dzz = dz_s[dzts + 1 * hu * bs + r * bs + b_idx];
                let dzh = dz_s[dzts + 2 * hu * bs + r * bs + b_idx];

                l_dbr[r] += dzr;
                l_dbz[r] += dzz;
                l_dbh[r] += dzh;

                let r_iu = r * iu;
                let r_hu = r * hu;
                for c in 0..iu {
                    let xv = x_s[c * total_cols + xoff];
                    l_dwxr[r_iu + c] += dzr * xv;
                    l_dwxz[r_iu + c] += dzz * xv;
                    l_dwxh[r_iu + c] += dzh * xv;
                }
                for c in 0..hu {
                    let hv = h_s[hoff + c * bs];
                    l_dwhr[r_hu + c] += dzr * hv;
                    l_dwhz[r_hu + c] += dzz * hv;
                    l_dwhh[r_hu + c] += dzh * hv * r_s[t * bs * hu + r * bs + b_idx];
                }
            }
        }
        (l_dwxr, l_dwhr, l_dbr, l_dwxz, l_dwhz, l_dbz, l_dwxh, l_dwhh, l_dbh)
    }).reduce(|| (vec![0.0f32; hu * iu], vec![0.0f32; hu * hu], vec![0.0f32; hu],
                  vec![0.0f32; hu * iu], vec![0.0f32; hu * hu], vec![0.0f32; hu],
                  vec![0.0f32; hu * iu], vec![0.0f32; hu * hu], vec![0.0f32; hu]),
    |mut a, b| {
        for i in 0..a.0.len() { a.0[i] += b.0[i]; a.3[i] += b.3[i]; a.6[i] += b.6[i]; }
        for i in 0..a.1.len() { a.1[i] += b.1[i]; a.4[i] += b.4[i]; a.7[i] += b.7[i]; }
        for i in 0..a.2.len() { a.2[i] += b.2[i]; a.5[i] += b.5[i]; a.8[i] += b.8[i]; }
        a
    });

    dwxr.copy_from_slice(&grads.0); dwhr.copy_from_slice(&grads.1); dbr.copy_from_slice(&grads.2);
    dwxz.copy_from_slice(&grads.3); dwhz.copy_from_slice(&grads.4); dbz.copy_from_slice(&grads.5);
    dwxh.copy_from_slice(&grads.6); dwhh.copy_from_slice(&grads.7); dbh.copy_from_slice(&grads.8);
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_memory_rnn_forward_native_into(
    wq: Float32Array,
    wm: Float32Array,
    wxh: Float32Array,
    whh: Float32Array,
    bh: Float32Array,
    wg: Float32Array,
    bg: Float32Array,
    x_seq: Float32Array,
    h0: Float32Array,
    hidden_units: u32,
    input_units: u32,
    memory_dim: u32,
    memory_slots: u32,
    seq_len: u32,
    batch_size: u32,
    use_relu: bool,
    mut h_seq_out: Float32Array,
    mut act_grad: Float32Array,
    mut mem_keys: Float32Array,
    mut mem_values: Float32Array,
    mut mem_usage: Float32Array,
    mut combined_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let md = memory_dim as usize;
    let ms = memory_slots as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let combined_width = iu + md;

    // Initialise h_seq from h0
    for i in 0..(hu * bs) {
        h_seq_out[i] = h0[i];
    }

    let h_p = SafeRawPtrMut(h_seq_out.as_mut_ptr() as usize);
    let da_p = SafeRawPtrMut(act_grad.as_mut_ptr() as usize);
    let mk_p = SafeRawPtrMut(mem_keys.as_mut_ptr() as usize);
    let mv_p = SafeRawPtrMut(mem_values.as_mut_ptr() as usize);
    let mu_p = SafeRawPtrMut(mem_usage.as_mut_ptr() as usize);
    let co_p = SafeRawPtrMut(combined_out.as_mut_ptr() as usize);
    
    let wxh_p = SafeRawPtr(wxh.as_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let bh_p = SafeRawPtr(bh.as_ptr() as usize);
    let x_p = SafeRawPtr(x_seq.as_ptr() as usize);
    let wq_p = SafeRawPtr(wq.as_ptr() as usize);
    let wm_p = SafeRawPtr(wm.as_ptr() as usize);
    let wg_p = SafeRawPtr(wg.as_ptr() as usize);
    let bg_p = SafeRawPtr(bg.as_ptr() as usize);
    let h0_p = SafeRawPtr(h0.as_ptr() as usize);
    let use_relu_val = use_relu;

    (0..bs).into_par_iter().for_each(move |b| {
        unsafe {
            let hb = h_p.0 as *mut f32;
            let dab = da_p.0 as *mut f32;
            let mkb = mk_p.0 as *mut f32;
            let mvb = mv_p.0 as *mut f32;
            let mub = mu_p.0 as *mut f32;
            let cob = co_p.0 as *mut f32;
            
            let wxh_s = std::slice::from_raw_parts(wxh_p.0 as *const f32, hu * combined_width);
            let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
            let bh_s = std::slice::from_raw_parts(bh_p.0 as *const f32, hu);
            let x_seq_s = std::slice::from_raw_parts(x_p.0 as *const f32, iu * total_cols);
            let wq_s = std::slice::from_raw_parts(wq_p.0 as *const f32, md * (iu + hu));
            let wm_s = std::slice::from_raw_parts(wm_p.0 as *const f32, md * hu);
            let wg_s = std::slice::from_raw_parts(wg_p.0 as *const f32, md * (iu + hu + md));
            let bg_s = std::slice::from_raw_parts(bg_p.0 as *const f32, md);
            let h0_ptr = h0_p.0 as *const f32;

            let mem_off = b * md * ms;
            let usage_off = b * ms;
            let score_scale = 1.0 / (md as f32).sqrt();

            let mut query = vec![0.0f32; md];
            let mut scores = vec![0.0f32; ms];
            let mut attn = vec![0.0f32; ms];
            let mut read = vec![0.0f32; md];
            let mut gate = vec![0.0f32; md];
            let mut cand = vec![0.0f32; md];
            let mut x_t = vec![0.0f32; iu];
            let mut h_prev_t = vec![0.0f32; hu];
            let mut h_curr_t = vec![0.0f32; hu];
            let mut local_h_seq = vec![0.0f32; (sl + 1) * hu];
            let mut local_act_grad = vec![0.0f32; sl * hu];
            let mut memory_keys_t = vec![0.0f32; md * ms];
            let mut memory_values_t = vec![0.0f32; md * ms];
            let mut memory_usage_t = vec![0.0f32; ms];

            for i in 0..hu {
                let h0_v = *h0_ptr.add(i * bs + b);
                h_prev_t[i] = h0_v;
                local_h_seq[i] = h0_v;
            }
            for i in 0..(md * ms) {
                memory_keys_t[i] = *mkb.add(mem_off + i);
                memory_values_t[i] = *mvb.add(mem_off + i);
            }
            for i in 0..ms {
                memory_usage_t[i] = *mub.add(usage_off + i);
            }

            for t in 0..sl {
                let t_bs = t * bs + b;
                for i in 0..iu {
                    x_t[i] = x_seq_s[i * total_cols + t_bs];
                }

                // 1. Query projection
                let qi_len = iu + hu;
                for i in 0..md {
                    let mut s = 0.0f32;
                    let row = i * qi_len;
                    for j in 0..iu {
                        s += wq_s[row + j] * x_t[j];
                    }
                    for j in 0..hu {
                        s += wq_s[row + iu + j] * h_prev_t[j];
                    }
                    query[i] = s;
                }

                // 2. Attention retrieval
                let mut best_slot = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for slot in 0..ms {
                    let mut sc = 0.0f32;
                    for i in 0..md {
                        sc += query[i] * memory_keys_t[i * ms + slot];
                    }
                    sc *= score_scale;
                    scores[slot] = sc;
                    if sc > best_score {
                        best_score = sc;
                        best_slot = slot;
                    }
                }
                stable_softmax(&scores, &mut attn);
                for i in 0..md {
                    let mut s = 0.0f32;
                    let base = i * ms;
                    for slot in 0..ms {
                        s += memory_values_t[base + slot] * attn[slot];
                    }
                    read[i] = s;
                }

                // 3. Combined output
                let co_t_base = t * combined_width * bs;
                for j in 0..iu {
                    *cob.add(co_t_base + j * bs + b) = x_t[j];
                }
                for j in 0..md {
                    *cob.add(co_t_base + (iu + j) * bs + b) = read[j];
                }

                // 4. RNN cell
                let local_h_curr_base = (t + 1) * hu;
                let local_da_base = t * hu;
                for i in 0..hu {
                    let mut s = bh_s[i];
                    let wx_row = i * combined_width;
                    for j in 0..iu {
                        s += wxh_s[wx_row + j] * x_t[j];
                    }
                    for j in 0..md {
                        s += wxh_s[wx_row + iu + j] * read[j];
                    }
                    let wh_row = i * hu;
                    for j in 0..hu {
                        s += whh_s[wh_row + j] * h_prev_t[j];
                    }
                    h_curr_t[i] = if use_relu_val {
                        if s > 0.0 {
                            local_act_grad[local_da_base + i] = 1.0;
                            s
                        } else {
                            local_act_grad[local_da_base + i] = 0.0;
                            0.0
                        }
                    } else {
                        let tv = s.tanh();
                        local_act_grad[local_da_base + i] = 1.0 - tv * tv;
                        tv
                    };
                    local_h_seq[local_h_curr_base + i] = h_curr_t[i];
                }

                // 5. Write gate
                let gi_len = iu + hu + md;
                for i in 0..md {
                    let mut s = bg_s[i];
                    let row = i * gi_len;
                    for j in 0..iu {
                        s += wg_s[row + j] * x_t[j];
                    }
                    for j in 0..hu {
                        s += wg_s[row + iu + j] * h_curr_t[j];
                    }
                    for j in 0..md {
                        s += wg_s[row + iu + hu + j] * read[i];
                    }
                    gate[i] = sigmoid(s);
                }

                // 6. Candidate memory
                for i in 0..md {
                    let mut s = 0.0f32;
                    let row = i * hu;
                    for j in 0..hu {
                        s += wm_s[row + j] * h_curr_t[j];
                    }
                    cand[i] = s;
                }

                // 7. Select write slot
                let mut write_slot = ms;
                for slot in 0..ms {
                    if memory_usage_t[slot] == 0.0 {
                        write_slot = slot;
                        break;
                    }
                }
                if write_slot == ms {
                    write_slot = best_slot;
                    let mut min_u = memory_usage_t[best_slot];
                    for slot in 0..ms {
                        let u = memory_usage_t[slot];
                        if u < min_u {
                            min_u = u;
                            write_slot = slot;
                        }
                    }
                }

                // 8. Gated memory update
                for i in 0..md {
                    let idx = i * ms + write_slot;
                    let g = gate[i];
                    memory_keys_t[idx] = (1.0 - g) * memory_keys_t[idx] + g * query[i];
                    memory_values_t[idx] = (1.0 - g) * memory_values_t[idx] + g * cand[i];
                }
                memory_usage_t[write_slot] += 1.0;
                std::mem::swap(&mut h_prev_t, &mut h_curr_t);
            }
            for t in 0..=sl {
                let global_h_base = t * bs * hu + b;
                let local_h_base = t * hu;
                for i in 0..hu {
                    *hb.add(global_h_base + i * bs) = local_h_seq[local_h_base + i];
                }
            }
            for t in 0..sl {
                let global_da_base = t * bs * hu + b;
                let local_da_base = t * hu;
                for i in 0..hu {
                    *dab.add(global_da_base + i * bs) = local_act_grad[local_da_base + i];
                }
            }
            for i in 0..(md * ms) {
                *mkb.add(mem_off + i) = memory_keys_t[i];
                *mvb.add(mem_off + i) = memory_values_t[i];
            }
            for i in 0..ms {
                *mub.add(usage_off + i) = memory_usage_t[i];
            }
        }
    });
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_memory_rnn_backward_native_into(
    wxh: Float32Array,
    whh: Float32Array,
    combined: Float32Array,
    h_seq: Float32Array,
    act_grad: Float32Array,
    err_h: Float32Array,
    hidden_units: u32,
    input_units: u32,
    memory_dim: u32,
    seq_len: u32,
    batch_size: u32,
    mut dwxh: Float32Array,
    mut dwhh: Float32Array,
    mut dbh: Float32Array,
    mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let md = memory_dim as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let combined_width = iu + md;

    let mut dz_all = vec![0.0f32; sl * bs * hu];
    
    let dz_p = SafeRawPtrMut(dz_all.as_mut_ptr() as usize);
    let dx_p = SafeRawPtrMut(dx_out.as_mut_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let wxh_p = SafeRawPtr(wxh.as_ptr() as usize);
    let err_h_p = SafeRawPtr(err_h.as_ptr() as usize);
    let act_grad_p = SafeRawPtr(act_grad.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(move |b| unsafe {
        let dzb = dz_p.0 as *mut f32;
        let dxb = dx_p.0 as *mut f32;
        let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
        let wxh_s = std::slice::from_raw_parts(wxh_p.0 as *const f32, hu * combined_width);
        let err_h_s = std::slice::from_raw_parts(err_h_p.0 as *const f32, sl * bs * hu);
        let act_grad_s = std::slice::from_raw_parts(act_grad_p.0 as *const f32, sl * bs * hu);

        let mut dhn = vec![0.0f32; hu];
        let mut ndh = vec![0.0f32; hu];
        for ti in 0..sl {
            let t = sl - 1 - ti;
            let gate = t * bs * hu + b;
            for i in 0..hu {
                let dz = (err_h_s[gate + i * bs] + dhn[i]) * act_grad_s[gate + i * bs];
                *dzb.add(gate + i * bs) = dz;
            }
            for j in 0..hu {
                let mut s = 0.0f32;
                for k in 0..hu {
                    s += whh_s[k * hu + j] * *dzb.add(gate + k * bs);
                }
                ndh[j] = s;
            }
            std::mem::swap(&mut dhn, &mut ndh);
            for j in 0..iu {
                let mut s = 0.0f32;
                for k in 0..hu {
                    s += wxh_s[k * combined_width + j] * *dzb.add(gate + k * bs);
                }
                *dxb.add(j * total_cols + t * bs + b) = s;
            }
        }
    });

    let dwx_p = SafeRawPtrMut(dwxh.as_mut_ptr() as usize);
    let dwh_p = SafeRawPtrMut(dwhh.as_mut_ptr() as usize);
    let co_p = SafeRawPtr(combined.as_ptr() as usize);
    let hs_p = SafeRawPtr(h_seq.as_ptr() as usize);
    let dz_p_accum = SafeRawPtr(dz_all.as_ptr() as usize);

    (0..hu).into_par_iter().for_each(move |r| unsafe {
        let dwx = dwx_p.0 as *mut f32;
        let dwh = dwh_p.0 as *mut f32;
        let combined_s = std::slice::from_raw_parts(co_p.0 as *const f32, sl * bs * combined_width);
        let h_seq_s = std::slice::from_raw_parts(hs_p.0 as *const f32, (sl + 1) * bs * hu);
        let dz_s = std::slice::from_raw_parts(dz_p_accum.0 as *const f32, sl * bs * hu);
        
        for t in 0..sl {
            let dz_off = t * bs * hu;
            let co_off = t * combined_width * bs;
            let hs_off = t * bs * hu;
            for b in 0..bs {
                let dv = dz_s[dz_off + r * bs + b];
                for c in 0..combined_width {
                    *dwx.add(r * combined_width + c) += dv * combined_s[co_off + c * bs + b];
                }
                for c in 0..hu {
                    *dwh.add(r * hu + c) += dv * h_seq_s[hs_off + c * bs + b];
                }
            }
        }
    });

    let dbh_p = SafeRawPtrMut(dbh.as_mut_ptr() as usize);
    let dz_p_bias = SafeRawPtr(dz_all.as_ptr() as usize);
    (0..hu).into_par_iter().for_each(move |r| unsafe {
        let dbh_ptr = dbh_p.0 as *mut f32;
        let dz_s = std::slice::from_raw_parts(dz_p_bias.0 as *const f32, sl * bs * hu);
        let mut s = 0.0f32;
        for t in 0..sl {
            let off = t * bs * hu + r * bs;
            for b in 0..bs {
                s += dz_s[off + b];
            }
        }
        *dbh_ptr.add(r) = s;
    });
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_memory_rnn_backward_full_native_into(
    wxh: Float32Array,
    whh: Float32Array,
    wq: Float32Array,
    wm: Float32Array,
    wg: Float32Array,
    h_prev: Float32Array,
    h: Float32Array,
    d_act: Float32Array,
    combined: Float32Array,
    _read: Float32Array,
    query_input: Float32Array,
    query: Float32Array,
    attention: Float32Array,
    gate_input: Float32Array,
    gate: Float32Array,
    candidate: Float32Array,
    memory_keys_before: Float32Array,
    memory_values_before: Float32Array,
    write_slots: Int32Array,
    err_h: Float32Array,
    hidden_units: u32,
    input_units: u32,
    memory_dim: u32,
    memory_slots: u32,
    seq_len: u32,
    batch_size: u32,
    mut dwxh: Float32Array,
    mut dwhh: Float32Array,
    mut dbh: Float32Array,
    mut dwq: Float32Array,
    mut dwm: Float32Array,
    mut dwg: Float32Array,
    mut dbg: Float32Array,
    mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let md = memory_dim as usize;
    let ms = memory_slots as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let combined_width = iu + md;
    let query_input_width = iu + hu;
    let gate_input_width = iu + hu + md;
    let memory_state_size = md * ms;
    let score_scale = 1.0 / (md as f32).sqrt();

    let wxh_p = SafeRawPtr(wxh.as_ptr() as usize);
    let whh_p = SafeRawPtr(whh.as_ptr() as usize);
    let wq_p = SafeRawPtr(wq.as_ptr() as usize);
    let wm_p = SafeRawPtr(wm.as_ptr() as usize);
    let wg_p = SafeRawPtr(wg.as_ptr() as usize);
    let h_prev_p = SafeRawPtr(h_prev.as_ptr() as usize);
    let h_p = SafeRawPtr(h.as_ptr() as usize);
    let d_act_p = SafeRawPtr(d_act.as_ptr() as usize);
    let combined_p = SafeRawPtr(combined.as_ptr() as usize);
    let query_input_p = SafeRawPtr(query_input.as_ptr() as usize);
    let query_p = SafeRawPtr(query.as_ptr() as usize);
    let attention_p = SafeRawPtr(attention.as_ptr() as usize);
    let gate_input_p = SafeRawPtr(gate_input.as_ptr() as usize);
    let gate_p = SafeRawPtr(gate.as_ptr() as usize);
    let candidate_p = SafeRawPtr(candidate.as_ptr() as usize);
    let memory_keys_before_p = SafeRawPtr(memory_keys_before.as_ptr() as usize);
    let memory_values_before_p = SafeRawPtr(memory_values_before.as_ptr() as usize);
    let write_slots_p = SafeRawPtr(write_slots.as_ptr() as usize);
    let err_h_p = SafeRawPtr(err_h.as_ptr() as usize);

    let batch_results: Vec<_> = (0..bs)
        .into_par_iter()
        .map(move |b| unsafe {
            let mut local_dwxh = vec![0.0f32; hu * combined_width];
            let mut local_dwhh = vec![0.0f32; hu * hu];
            let mut local_dbh = vec![0.0f32; hu];
            let mut local_dwq = vec![0.0f32; md * query_input_width];
            let mut local_dwm = vec![0.0f32; md * hu];
            let mut local_dwg = vec![0.0f32; md * gate_input_width];
            let mut local_dbg = vec![0.0f32; md];
            let mut local_dx = vec![0.0f32; iu * total_cols];

            let mut dh_next = vec![0.0f32; hu];
            let mut dh_prev_buf = vec![0.0f32; hu];
            let mut d_memory_keys_next = vec![0.0f32; memory_state_size];
            let mut d_memory_values_next = vec![0.0f32; memory_state_size];
            let mut d_memory_keys_before = vec![0.0f32; memory_state_size];
            let mut d_memory_values_before = vec![0.0f32; memory_state_size];
            let mut d_query = vec![0.0f32; md];
            let mut d_candidate = vec![0.0f32; md];
            let mut d_gate = vec![0.0f32; md];
            let mut d_read = vec![0.0f32; md];
            let mut dh = vec![0.0f32; hu];
            let mut d_gate_pre = vec![0.0f32; md];
            let mut dz = vec![0.0f32; hu];
            let mut d_attention = vec![0.0f32; ms];
            let mut d_scores = vec![0.0f32; ms];

            let wxh_s = std::slice::from_raw_parts(wxh_p.0 as *const f32, hu * combined_width);
            let whh_s = std::slice::from_raw_parts(whh_p.0 as *const f32, hu * hu);
            let wq_s = std::slice::from_raw_parts(wq_p.0 as *const f32, md * query_input_width);
            let wm_s = std::slice::from_raw_parts(wm_p.0 as *const f32, md * hu);
            let wg_s = std::slice::from_raw_parts(wg_p.0 as *const f32, md * gate_input_width);
            let h_prev_s = std::slice::from_raw_parts(h_prev_p.0 as *const f32, sl * bs * hu);
            let h_s = std::slice::from_raw_parts(h_p.0 as *const f32, sl * bs * hu);
            let d_act_s = std::slice::from_raw_parts(d_act_p.0 as *const f32, sl * bs * hu);
            let combined_s = std::slice::from_raw_parts(combined_p.0 as *const f32, sl * bs * combined_width);
            let query_input_s = std::slice::from_raw_parts(query_input_p.0 as *const f32, sl * bs * query_input_width);
            let query_s = std::slice::from_raw_parts(query_p.0 as *const f32, sl * bs * md);
            let attention_s = std::slice::from_raw_parts(attention_p.0 as *const f32, sl * bs * ms);
            let gate_input_s = std::slice::from_raw_parts(gate_input_p.0 as *const f32, sl * bs * gate_input_width);
            let gate_s = std::slice::from_raw_parts(gate_p.0 as *const f32, sl * bs * md);
            let candidate_s = std::slice::from_raw_parts(candidate_p.0 as *const f32, sl * bs * md);
            let memory_keys_before_s = std::slice::from_raw_parts(memory_keys_before_p.0 as *const f32, sl * bs * memory_state_size);
            let memory_values_before_s = std::slice::from_raw_parts(memory_values_before_p.0 as *const f32, sl * bs * memory_state_size);
            let write_slots_s = std::slice::from_raw_parts(write_slots_p.0 as *const i32, sl * bs);
            let err_h_s = std::slice::from_raw_parts(err_h_p.0 as *const f32, sl * bs * hu);

            for ti in 0..sl {
                let t = sl - 1 - ti;
                let step_index = b * sl + t;
                let h_prev_base = step_index * hu;
                let h_base = step_index * hu;
                let d_act_base = step_index * hu;
                let combined_base = step_index * combined_width;
                let query_input_base = step_index * query_input_width;
                let query_base = step_index * md;
                let attention_base = step_index * ms;
                let gate_input_base = step_index * gate_input_width;
                let gate_base = step_index * md;
                let candidate_base = step_index * md;
                let memory_base = step_index * memory_state_size;
                let err_base = step_index * hu;
                let write_slot = write_slots_s[step_index] as usize;

                d_query.fill(0.0);
                d_candidate.fill(0.0);
                d_gate.fill(0.0);
                d_read.fill(0.0);
                d_memory_keys_before.copy_from_slice(&d_memory_keys_next);
                d_memory_values_before.copy_from_slice(&d_memory_values_next);

                for i in 0..md {
                    let idx = i * ms + write_slot;
                    let gate_val = gate_s[gate_base + i];
                    let old_key = memory_keys_before_s[memory_base + idx];
                    let old_value = memory_values_before_s[memory_base + idx];
                    let d_key_after = d_memory_keys_next[idx];
                    let d_value_after = d_memory_values_next[idx];
                    d_memory_keys_before[idx] = d_key_after * (1.0 - gate_val);
                    d_memory_values_before[idx] = d_value_after * (1.0 - gate_val);
                    d_query[i] += d_key_after * gate_val;
                    d_candidate[i] += d_value_after * gate_val;
                    d_gate[i] += d_key_after * (query_s[query_base + i] - old_key)
                        + d_value_after * (candidate_s[candidate_base + i] - old_value);
                }

                for i in 0..hu {
                    dh[i] = err_h_s[err_base + i] + dh_next[i];
                }

                for i in 0..md {
                    let dc = d_candidate[i];
                    let row_off = i * hu;
                    for j in 0..hu {
                        local_dwm[row_off + j] += dc * h_s[h_base + j];
                    }
                }
                for j in 0..hu {
                    let mut sum = 0.0f32;
                    for i in 0..md {
                        sum += wm_s[i * hu + j] * d_candidate[i];
                    }
                    dh[j] += sum;
                }

                for i in 0..md {
                    let gate_val = gate_s[gate_base + i];
                    let dgp = d_gate[i] * gate_val * (1.0 - gate_val);
                    d_gate_pre[i] = dgp;
                    local_dbg[i] += dgp;
                }

                for i in 0..md {
                    let dgp = d_gate_pre[i];
                    let row_off = i * gate_input_width;
                    for j in 0..gate_input_width {
                        local_dwg[row_off + j] += dgp * gate_input_s[gate_input_base + j];
                    }
                }

                let col = t * bs + b;
                for j in 0..iu {
                    let mut sum = 0.0f32;
                    for i in 0..md {
                        sum += wg_s[i * gate_input_width + j] * d_gate_pre[i];
                    }
                    local_dx[j * total_cols + col] += sum;
                }
                for j in 0..hu {
                    let mut sum = 0.0f32;
                    for i in 0..md {
                        sum += wg_s[i * gate_input_width + iu + j] * d_gate_pre[i];
                    }
                    dh[j] += sum;
                }
                for j in 0..md {
                    let mut sum = 0.0f32;
                    for i in 0..md {
                        sum += wg_s[i * gate_input_width + iu + hu + j] * d_gate_pre[i];
                    }
                    d_read[j] += sum;
                }

                for i in 0..hu {
                    let dzv = dh[i] * d_act_s[d_act_base + i];
                    dz[i] = dzv;
                    local_dbh[i] += dzv;
                }

                for i in 0..hu {
                    let dzv = dz[i];
                    let wxh_off = i * combined_width;
                    let whh_off = i * hu;
                    for j in 0..combined_width {
                        local_dwxh[wxh_off + j] += dzv * combined_s[combined_base + j];
                    }
                    for j in 0..hu {
                        local_dwhh[whh_off + j] += dzv * h_prev_s[h_prev_base + j];
                    }
                }

                for j in 0..iu {
                    let mut sum = 0.0f32;
                    for i in 0..hu {
                        sum += wxh_s[i * combined_width + j] * dz[i];
                    }
                    local_dx[j * total_cols + col] += sum;
                }
                for j in 0..md {
                    let mut sum = 0.0f32;
                    for i in 0..hu {
                        sum += wxh_s[i * combined_width + iu + j] * dz[i];
                    }
                    d_read[j] += sum;
                }

                for j in 0..hu {
                    let mut sum = 0.0f32;
                    for i in 0..hu {
                        sum += whh_s[i * hu + j] * dz[i];
                    }
                    dh_prev_buf[j] = sum;
                }

                for slot in 0..ms {
                    let mut attn_grad = 0.0f32;
                    for i in 0..md {
                        let idx = i * ms + slot;
                        d_memory_values_before[idx] +=
                            d_read[i] * attention_s[attention_base + slot];
                        attn_grad += memory_values_before_s[memory_base + idx] * d_read[i];
                    }
                    d_attention[slot] = attn_grad;
                }

                let mut softmax_inner = 0.0f32;
                for slot in 0..ms {
                    softmax_inner += d_attention[slot] * attention_s[attention_base + slot];
                }
                for slot in 0..ms {
                    d_scores[slot] =
                        attention_s[attention_base + slot] * (d_attention[slot] - softmax_inner);
                }

                for slot in 0..ms {
                    let score_grad = d_scores[slot] * score_scale;
                    for i in 0..md {
                        let idx = i * ms + slot;
                        d_memory_keys_before[idx] += query_s[query_base + i] * score_grad;
                        d_query[i] += memory_keys_before_s[memory_base + idx] * score_grad;
                    }
                }

                for i in 0..md {
                    let dq = d_query[i];
                    let row_off = i * query_input_width;
                    for j in 0..query_input_width {
                        local_dwq[row_off + j] += dq * query_input_s[query_input_base + j];
                    }
                }

                for j in 0..iu {
                    let mut sum = 0.0f32;
                    for i in 0..md {
                        sum += wq_s[i * query_input_width + j] * d_query[i];
                    }
                    local_dx[j * total_cols + col] += sum;
                }
                for j in 0..hu {
                    let mut sum = 0.0f32;
                    for i in 0..md {
                        sum += wq_s[i * query_input_width + iu + j] * d_query[i];
                    }
                    dh_prev_buf[j] += sum;
                }

                std::mem::swap(&mut dh_next, &mut dh_prev_buf);
                std::mem::swap(&mut d_memory_keys_next, &mut d_memory_keys_before);
                std::mem::swap(&mut d_memory_values_next, &mut d_memory_values_before);
            }

            (
                local_dwxh, local_dwhh, local_dbh, local_dwq, local_dwm, local_dwg, local_dbg,
                local_dx,
            )
        })
        .collect();

    for (local_dwxh, local_dwhh, local_dbh, local_dwq, local_dwm, local_dwg, local_dbg, local_dx) in
        batch_results
    {
        for i in 0..local_dwxh.len() { dwxh[i] += local_dwxh[i]; }
        for i in 0..local_dwhh.len() { dwhh[i] += local_dwhh[i]; }
        for i in 0..local_dbh.len() { dbh[i] += local_dbh[i]; }
        for i in 0..local_dwq.len() { dwq[i] += local_dwq[i]; }
        for i in 0..local_dwm.len() { dwm[i] += local_dwm[i]; }
        for i in 0..local_dwg.len() { dwg[i] += local_dwg[i]; }
        for i in 0..local_dbg.len() { dbg[i] += local_dbg[i]; }
        for i in 0..local_dx.len() { dx_out[i] += local_dx[i]; }
    }
}
