use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const ACTIVATION_PARALLEL_THRESHOLD: usize = 16 * 1024;

/// Process a chunk of elements for relu with a pattern that encourages
/// auto-vectorization (no branches inside the inner loop body).
#[inline(always)]
fn relu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    debug_assert_eq!(input.len(), out_res.len());
    debug_assert_eq!(input.len(), out_grad.len());
    for i in 0..input.len() {
        let val = input[i];
        // branchless: select via comparison mask
        let mask = if val > 0.0 { 1.0f32 } else { 0.0f32 };
        out_res[i] = val * mask;
        out_grad[i] = mask;
    }
}

#[napi]
pub fn relu_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        relu_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        // Use chunks_exact for better auto-vectorization alignment
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                relu_chunk(inp, res_chunk, grad_chunk);
            });
        // remainder
        if full < n {
            relu_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}

#[inline(always)]
fn sigmoid_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let val = 1.0 / (1.0 + (-input[i]).exp());
        out_res[i] = val;
        out_grad[i] = val * (1.0 - val);
    }
}

#[napi]
pub fn sigmoid_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        sigmoid_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                sigmoid_chunk(inp, res_chunk, grad_chunk);
            });
        if full < n {
            sigmoid_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}

#[inline(always)]
fn tanh_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let val = input[i].tanh();
        out_res[i] = val;
        out_grad[i] = 1.0 - val * val;
    }
}

#[napi]
pub fn tanh_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        tanh_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                tanh_chunk(inp, res_chunk, grad_chunk);
            });
        if full < n {
            tanh_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}

#[inline(always)]
fn lrelu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let val = input[i];
        if val < 0.0 {
            out_res[i] = val * 1e-5;
            out_grad[i] = 1e-5;
        } else {
            out_res[i] = val;
            out_grad[i] = 1.0;
        }
    }
}

#[napi]
pub fn l_relu_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        lrelu_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                lrelu_chunk(inp, res_chunk, grad_chunk);
            });
        if full < n {
            lrelu_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}
