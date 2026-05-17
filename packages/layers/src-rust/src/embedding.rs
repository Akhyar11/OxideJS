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

#[napi]
pub fn embedding_forward_native(
    inputs: Float32Array,
    embeddings: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    mut out: Float32Array,
) -> napi::Result<()> {
    let total_tokens = inputs.len();
    let embed_dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let embeddings_ptr = SendPtrConst(embeddings.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());

    (0..total_tokens).into_par_iter().try_for_each(move |i| -> napi::Result<()> {
        let idx = unsafe { (*inputs_ptr.get().add(i)).floor() as isize };
        if idx < 0 || idx >= v_size as isize {
            return Err(napi::Error::from_reason(format!(
                "[Embedding] Token index {} is out of vocabulary bounds [0, {}].",
                idx,
                v_size - 1
            )));
        }
        let dest_offset = i * embed_dim;
        let src_offset = idx as usize * embed_dim;
        unsafe {
            for j in 0..embed_dim {
                *out_ptr.get().add(dest_offset + j) = *embeddings_ptr.get().add(src_offset + j);
            }
        }
        Ok(())
    })
}

#[napi]
pub fn embedding_backward_native(
    grad_out: Float32Array,
    inputs: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    mut grad_embed: Float32Array,
) -> napi::Result<()> {
    let total_tokens = inputs.len();
    let embed_dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());

    // Validate indices in parallel first for rapid short-circuiting
    (0..total_tokens).into_par_iter().try_for_each(move |i| -> napi::Result<()> {
        let idx = unsafe { (*inputs_ptr.get().add(i)).floor() as isize };
        if idx < 0 || idx >= v_size as isize {
            return Err(napi::Error::from_reason(format!(
                "[Embedding] Token index {} is out of vocabulary bounds [0, {}].",
                idx,
                v_size - 1
            )));
        }
        Ok(())
    })?;

    let grad_out_slice = &*grad_out;
    let inputs_slice = &*inputs;
    let grad_embed_slice = &mut *grad_embed;

    // Accumulate sequentially to prevent thread data races on overlapping token indices
    for i in 0..total_tokens {
        let idx = inputs_slice[i].floor() as usize;
        let src_offset = i * embed_dim;
        let dest_offset = idx * embed_dim;
        for j in 0..embed_dim {
            grad_embed_slice[dest_offset + j] += grad_out_slice[src_offset + j];
        }
    }

    Ok(())
}
