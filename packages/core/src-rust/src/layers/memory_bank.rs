use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

/// A simple wrapper to pass raw pointers across threads safely.
struct SafeRawPtr(usize);
unsafe impl Send for SafeRawPtr {}
unsafe impl Sync for SafeRawPtr {}

struct SafeRawPtrMut(usize);
unsafe impl Send for SafeRawPtrMut {}
unsafe impl Sync for SafeRawPtrMut {}

#[napi]
pub fn memory_bank_similarity_scores_native(
    query: Float32Array,
    keys: Float32Array,
    units: u32,
    slots: u32,
    similarity_type: String, // "cosine" or "dot"
    mut scores_out: Float32Array,
) {
    let u = units as usize;
    let s = slots as usize;
    let q_ptr = SafeRawPtr(query.as_ptr() as usize);
    let k_ptr = SafeRawPtr(keys.as_ptr() as usize);
    let out_ptr = SafeRawPtrMut(scores_out.as_mut_ptr() as usize);
    
    let is_cosine = similarity_type == "cosine";
    let score_scale = if !is_cosine { 1.0 / (u as f32).sqrt() } else { 1.0 };

    (0..s).into_par_iter().for_each(move |slot| {
        unsafe {
            let q = std::slice::from_raw_parts(q_ptr.0 as *const f32, u);
            let k_all = std::slice::from_raw_parts(k_ptr.0 as *const f32, u * s);
            let out = std::slice::from_raw_parts_mut(out_ptr.0 as *mut f32, s);
            
            let mut dot = 0.0f32;
            let mut norm_q = 0.0f32;
            let mut norm_k = 0.0f32;
            
            for i in 0..u {
                let kv = k_all[i * s + slot];
                let qv = q[i];
                dot += qv * kv;
                if is_cosine {
                    norm_q += qv * qv;
                    norm_k += kv * kv;
                }
            }
            
            if is_cosine {
                let denom = (norm_q * norm_k).sqrt();
                if denom > 1e-12 {
                    out[slot] = dot / denom;
                } else {
                    out[slot] = 0.0;
                }
            } else {
                out[slot] = dot * score_scale;
            }
        }
    });
}

#[napi]
pub fn memory_bank_update_native(
    mut keys: Float32Array,
    mut values: Float32Array,
    new_key: Float32Array,
    new_value: Float32Array,
    slot: u32,
    gate: Float32Array, // size [units]
    units: u32,
    slots: u32,
) {
    let u = units as usize;
    let s = slots as usize;
    let sl = slot as usize;
    
    let k_ptr = SafeRawPtrMut(keys.as_mut_ptr() as usize);
    let v_ptr = SafeRawPtrMut(values.as_mut_ptr() as usize);
    let nk_ptr = SafeRawPtr(new_key.as_ptr() as usize);
    let nv_ptr = SafeRawPtr(new_value.as_ptr() as usize);
    let g_ptr = SafeRawPtr(gate.as_ptr() as usize);

    // Update is sequential for the specific slot but we can vectorize across units
    // although for a single slot, a simple loop is usually fine.
    // However, we ensure no data race by locking to the slot.
    unsafe {
        let k_all = std::slice::from_raw_parts_mut(k_ptr.0 as *mut f32, u * s);
        let v_all = std::slice::from_raw_parts_mut(v_ptr.0 as *mut f32, u * s);
        let nk = std::slice::from_raw_parts(nk_ptr.0 as *const f32, u);
        let nv = std::slice::from_raw_parts(nv_ptr.0 as *const f32, u);
        let g = std::slice::from_raw_parts(g_ptr.0 as *const f32, u);

        for i in 0..u {
            let gate_val = g[i];
            let idx = i * s + sl;
            k_all[idx] = (1.0 - gate_val) * k_all[idx] + gate_val * nk[i];
            v_all[idx] = (1.0 - gate_val) * v_all[idx] + gate_val * nv[i];
        }
    }
}
