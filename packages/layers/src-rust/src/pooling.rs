use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

#[derive(Copy, Clone)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    pub fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Copy, Clone)]
struct SendPtrConst(*const f32);
unsafe impl Send for SendPtrConst {}
unsafe impl Sync for SendPtrConst {}

impl SendPtrConst {
    #[inline(always)]
    pub fn get(self) -> *const f32 {
        self.0
    }
}

#[derive(Copy, Clone)]
struct SendPtrInt32(*mut i32);
unsafe impl Send for SendPtrInt32 {}
unsafe impl Sync for SendPtrInt32 {}

impl SendPtrInt32 {
    #[inline(always)]
    pub fn get(self) -> *mut i32 {
        self.0
    }
}

#[derive(Copy, Clone)]
struct SendPtrInt32Const(*const i32);
unsafe impl Send for SendPtrInt32Const {}
unsafe impl Sync for SendPtrInt32Const {}

impl SendPtrInt32Const {
    #[inline(always)]
    pub fn get(self) -> *const i32 {
        self.0
    }
}

#[napi]
pub fn max_pooling_1d_forward_native(
    inputs: Float32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    pool_size: u32,
    strides: u32,
    pad_left: u32,
    mut out: Float32Array,
    mut max_indices: Int32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let p_size = pool_size as usize;
    let strd = strides as usize;
    let p_left = pad_left as usize;

    let l_out = out.len() / (b_size * in_dim);

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let max_indices_ptr = SendPtrInt32(max_indices.as_mut_ptr());
    let total_outputs = b_size * l_out;

    (0..total_outputs).into_par_iter().for_each(move |out_row_idx| {
        let b = out_row_idx / l_out;
        let i = out_row_idx % l_out;
        let t_start = (i * strd) as isize - p_left as isize;

        for c in 0..in_dim {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx: i32 = -1;

            for k in 0..p_size {
                let t = t_start + k as isize;
                if t >= 0 && t < seq_len as isize {
                    let src_idx = (b * seq_len + t as usize) * in_dim + c;
                    unsafe {
                        let val = *inputs_ptr.get().add(src_idx);
                        if val > max_val {
                            max_val = val;
                            max_idx = src_idx as i32;
                        }
                    }
                }
            }

            let dest_idx = out_row_idx * in_dim + c;
            unsafe {
                *out_ptr.get().add(dest_idx) = if max_val == f32::NEG_INFINITY { 0.0 } else { max_val };
                *max_indices_ptr.get().add(dest_idx) = max_idx;
            }
        }
    });
}

#[napi]
pub fn max_pooling_1d_backward_native(
    grad_out: Float32Array,
    max_indices: Int32Array,
    mut grad_in: Float32Array,
) {
    let grad_out_slice = &*grad_out;
    let max_indices_slice = &*max_indices;
    let grad_in_slice = &mut *grad_in;

    for idx in 0..max_indices_slice.len() {
        let src_idx = max_indices_slice[idx];
        if src_idx >= 0 {
            grad_in_slice[src_idx as usize] += grad_out_slice[idx];
        }
    }
}

#[napi]
pub fn max_pooling_2d_forward_native(
    inputs: Float32Array,
    batch_size: u32,
    height: u32,
    width: u32,
    channels: u32,
    pool_rows: u32,
    pool_cols: u32,
    stride_rows: u32,
    stride_cols: u32,
    pad_top: u32,
    pad_left: u32,
    h_out: u32,
    w_out: u32,
    mut out: Float32Array,
    mut max_indices: Int32Array,
) {
    let b_size = batch_size as usize;
    let h = height as usize;
    let w = width as usize;
    let c = channels as usize;
    let pr_size = pool_rows as usize;
    let pc_size = pool_cols as usize;
    let sr = stride_rows as usize;
    let sc = stride_cols as usize;
    let p_top = pad_top as usize;
    let p_left = pad_left as usize;
    let ho = h_out as usize;
    let wo = w_out as usize;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let max_indices_ptr = SendPtrInt32(max_indices.as_mut_ptr());
    let total_outputs = b_size * ho * wo;

    (0..total_outputs).into_par_iter().for_each(move |out_row_idx| {
        let b = out_row_idx / (ho * wo);
        let rem = out_row_idx % (ho * wo);
        let i = rem / wo;
        let j = rem % wo;

        let h_start = (i * sr) as isize - p_top as isize;
        let w_start = (j * sc) as isize - p_left as isize;

        for c_idx in 0..c {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx: i32 = -1;

            for pr in 0..pr_size {
                let h_idx = h_start + pr as isize;
                if h_idx >= 0 && h_idx < h as isize {
                    for pc in 0..pc_size {
                        let w_idx = w_start + pc as isize;
                        if w_idx >= 0 && w_idx < w as isize {
                            let src_idx = (b * h * w + h_idx as usize * w + w_idx as usize) * c + c_idx;
                            unsafe {
                                let val = *inputs_ptr.get().add(src_idx);
                                if val > max_val {
                                    max_val = val;
                                    max_idx = src_idx as i32;
                                }
                            }
                        }
                    }
                }
            }

            let dest_idx = out_row_idx * c + c_idx;
            unsafe {
                *out_ptr.get().add(dest_idx) = if max_val == f32::NEG_INFINITY { 0.0 } else { max_val };
                *max_indices_ptr.get().add(dest_idx) = max_idx;
            }
        }
    });
}

#[napi]
pub fn max_pooling_2d_backward_native(
    grad_out: Float32Array,
    max_indices: Int32Array,
    mut grad_in: Float32Array,
) {
    let grad_out_slice = &*grad_out;
    let max_indices_slice = &*max_indices;
    let grad_in_slice = &mut *grad_in;

    for idx in 0..max_indices_slice.len() {
        let src_idx = max_indices_slice[idx];
        if src_idx >= 0 {
            grad_in_slice[src_idx as usize] += grad_out_slice[idx];
        }
    }
}

#[napi]
pub fn average_pooling_1d_forward_native(
    inputs: Float32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    pool_size: u32,
    strides: u32,
    pad_left: u32,
    mut out: Float32Array,
    mut window_counts: Int32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let p_size = pool_size as usize;
    let strd = strides as usize;
    let p_left = pad_left as usize;

    let l_out = out.len() / (b_size * in_dim);

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let window_counts_ptr = SendPtrInt32(window_counts.as_mut_ptr());
    let total_outputs = b_size * l_out;

    (0..total_outputs).into_par_iter().for_each(move |out_row_idx| {
        let b = out_row_idx / l_out;
        let i = out_row_idx % l_out;
        let t_start = (i * strd) as isize - p_left as isize;

        let mut count = 0;
        for k in 0..p_size {
            let t = t_start + k as isize;
            if t >= 0 && t < seq_len as isize {
                count += 1;
            }
        }
        unsafe {
            *window_counts_ptr.get().add(out_row_idx) = count;
        }

        for c in 0..in_dim {
            let mut sum = 0.0;
            for k in 0..p_size {
                let t = t_start + k as isize;
                if t >= 0 && t < seq_len as isize {
                    unsafe {
                        sum += *inputs_ptr.get().add((b * seq_len + t as usize) * in_dim + c);
                    }
                }
            }
            unsafe {
                *out_ptr.get().add(out_row_idx * in_dim + c) = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
    });
}

#[napi]
pub fn average_pooling_1d_backward_native(
    grad_out: Float32Array,
    window_counts: Int32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    pool_size: u32,
    strides: u32,
    pad_left: u32,
    mut grad_in: Float32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let p_size = pool_size as usize;
    let strd = strides as usize;
    let p_left = pad_left as usize;

    let l_out = grad_out.len() / (b_size * in_dim);

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let window_counts_ptr = SendPtrInt32Const(window_counts.as_ptr());
    let grad_in_ptr = SendPtr(grad_in.as_mut_ptr());

    (0..b_size).into_par_iter().for_each(move |b| {
        for i in 0..l_out {
            let out_row_idx = b * l_out + i;
            let t_start = (i * strd) as isize - p_left as isize;
            unsafe {
                let count = *window_counts_ptr.get().add(out_row_idx);
                if count > 0 {
                    let count_f = count as f32;
                    for c in 0..in_dim {
                        let val = *grad_out_ptr.get().add(out_row_idx * in_dim + c) / count_f;
                        for k in 0..p_size {
                            let t = t_start + k as isize;
                            if t >= 0 && t < seq_len as isize {
                                *grad_in_ptr.get().add((b * seq_len + t as usize) * in_dim + c) += val;
                            }
                        }
                    }
                }
            }
        }
    });
}

#[napi]
pub fn average_pooling_2d_forward_native(
    inputs: Float32Array,
    batch_size: u32,
    height: u32,
    width: u32,
    channels: u32,
    pool_rows: u32,
    pool_cols: u32,
    stride_rows: u32,
    stride_cols: u32,
    pad_top: u32,
    pad_left: u32,
    h_out: u32,
    w_out: u32,
    mut out: Float32Array,
    mut window_counts: Int32Array,
) {
    let b_size = batch_size as usize;
    let h = height as usize;
    let w = width as usize;
    let c = channels as usize;
    let pr_size = pool_rows as usize;
    let pc_size = pool_cols as usize;
    let sr = stride_rows as usize;
    let sc = stride_cols as usize;
    let p_top = pad_top as usize;
    let p_left = pad_left as usize;
    let ho = h_out as usize;
    let wo = w_out as usize;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let window_counts_ptr = SendPtrInt32(window_counts.as_mut_ptr());
    let total_outputs = b_size * ho * wo;

    (0..total_outputs).into_par_iter().for_each(move |out_row_idx| {
        let b = out_row_idx / (ho * wo);
        let rem = out_row_idx % (ho * wo);
        let i = rem / wo;
        let j = rem % wo;

        let h_start = (i * sr) as isize - p_top as isize;
        let w_start = (j * sc) as isize - p_left as isize;

        let mut count = 0;
        for pr in 0..pr_size {
            let h_idx = h_start + pr as isize;
            if h_idx >= 0 && h_idx < h as isize {
                for pc in 0..pc_size {
                    let w_idx = w_start + pc as isize;
                    if w_idx >= 0 && w_idx < w as isize {
                        count += 1;
                    }
                }
            }
        }
        unsafe {
            *window_counts_ptr.get().add(out_row_idx) = count;
        }

        for c_idx in 0..c {
            let mut sum = 0.0;
            for pr in 0..pr_size {
                let h_idx = h_start + pr as isize;
                if h_idx >= 0 && h_idx < h as isize {
                    for pc in 0..pc_size {
                        let w_idx = w_start + pc as isize;
                        if w_idx >= 0 && w_idx < w as isize {
                            unsafe {
                                sum += *inputs_ptr.get().add((b * h * w + h_idx as usize * w + w_idx as usize) * c + c_idx);
                            }
                        }
                    }
                }
            }

            unsafe {
                *out_ptr.get().add(out_row_idx * c + c_idx) = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
    });
}

#[napi]
pub fn average_pooling_2d_backward_native(
    grad_out: Float32Array,
    window_counts: Int32Array,
    batch_size: u32,
    height: u32,
    width: u32,
    channels: u32,
    pool_rows: u32,
    pool_cols: u32,
    stride_rows: u32,
    stride_cols: u32,
    pad_top: u32,
    pad_left: u32,
    h_out: u32,
    w_out: u32,
    mut grad_in: Float32Array,
) {
    let b_size = batch_size as usize;
    let h = height as usize;
    let w = width as usize;
    let c = channels as usize;
    let pr_size = pool_rows as usize;
    let pc_size = pool_cols as usize;
    let sr = stride_rows as usize;
    let sc = stride_cols as usize;
    let p_top = pad_top as usize;
    let p_left = pad_left as usize;
    let ho = h_out as usize;
    let wo = w_out as usize;

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let window_counts_ptr = SendPtrInt32Const(window_counts.as_ptr());
    let grad_in_ptr = SendPtr(grad_in.as_mut_ptr());

    (0..b_size).into_par_iter().for_each(move |b| {
        for i in 0..ho {
            for j in 0..wo {
                let out_row_idx = b * ho * wo + i * wo + j;
                let h_start = (i * sr) as isize - p_top as isize;
                let w_start = (j * sc) as isize - p_left as isize;
                unsafe {
                    let count = *window_counts_ptr.get().add(out_row_idx);
                    if count > 0 {
                        let count_f = count as f32;
                        for c_idx in 0..c {
                            let val = *grad_out_ptr.get().add(out_row_idx * c + c_idx) / count_f;

                            for pr in 0..pr_size {
                                let h_idx = h_start + pr as isize;
                                if h_idx >= 0 && h_idx < h as isize {
                                    for pc in 0..pc_size {
                                        let w_idx = w_start + pc as isize;
                                        if w_idx >= 0 && w_idx < w as isize {
                                            *grad_in_ptr.get().add((b * h * w + h_idx as usize * w + w_idx as usize) * c + c_idx) += val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}
