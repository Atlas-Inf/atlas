// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas-owned H32/D128 KDA prefill. CUDA/CUTLASS are build dependencies;
//! no model-specific external library, allocator or global state is used.
//! FP32 persistent storage does NOT imply FP32 internal recurrence: the
//! tensor-core kernel rounds recurrent shared state to BF16 at chunk boundaries.

#[path = "glm_kda/layout.rs"]
mod layout;
#[path = "glm_kda/sparkglm.rs"]
pub mod sparkglm;
pub use layout::{CHUNK, DIM, HEADS, MAX_ROWS, MIN_ROWS, ScratchLayout};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Unavailable,
    Launch(i32),
    Preload(i32),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Unavailable => f.write_str("Atlas native KDA was not compiled into this build"),
            Self::Preload(code) => write!(f, "Atlas native KDA CUDA preload failed: {code}"),
            Self::Launch(code) => write!(
                f,
                "Atlas native KDA CUDA error {code}; do not retry mutated state"
            ),
        }
    }
}
impl std::error::Error for Error {}

/// Internal native ABI, paired with cuda/glm_kda/params.h. Constructing this
/// value checks arithmetic/aliasing, not ownership of the device allocations.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Params {
    qkv: u64,
    gate: u64,
    beta: u64,
    a_log: u64,
    dt_bias: u64,
    state: u64,
    output: u64,
    workspace: u64,
    layout: ScratchLayout,
    tokens: u32,
    reserved: u32,
    stream: u64,
}

impl Params {
    /// All allocations must remain live on `stream` until its work completes.
    /// `state` is the same canonical key-major FP32 allocation on input/output.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qkv: u64,
        gate: u64,
        beta: u64,
        a_log: u64,
        dt_bias: u64,
        state: u64,
        output: u64,
        workspace: u64,
        workspace_bytes: usize,
        tokens: usize,
        stream: u64,
    ) -> Result<Self, Error> {
        let layout = ScratchLayout::new(tokens).map_err(Error::Invalid)?;
        if layout.bytes > workspace_bytes as u64 {
            return Err(Error::Invalid("native KDA scratch capacity exceeded"));
        }
        if !workspace.is_multiple_of(128) {
            return Err(Error::Invalid(
                "native KDA workspace must be 128-byte aligned",
            ));
        }
        let plane = (tokens * HEADS * DIM * 2) as u64;
        let ranges = [
            (qkv, 3 * plane),
            (gate, plane),
            (beta, (tokens * HEADS * 2) as u64),
            (a_log, (HEADS * 4) as u64),
            (dt_bias, (HEADS * DIM * 4) as u64),
            (state, (HEADS * DIM * DIM * 4) as u64),
            (output, plane),
            (workspace, layout.bytes),
        ];
        for (i, &(ptr, bytes)) in ranges.iter().enumerate() {
            if ptr == 0 || !ptr.is_multiple_of(16) {
                return Err(Error::Invalid(
                    "native KDA operands must be nonnull and 16-byte aligned",
                ));
            }
            let end = ptr
                .checked_add(bytes)
                .ok_or(Error::Invalid("native KDA address overflow"))?;
            for &(other, count) in &ranges[..i] {
                let other_end = other
                    .checked_add(count)
                    .ok_or(Error::Invalid("native KDA address overflow"))?;
                if ptr < other_end && other < end {
                    return Err(Error::Invalid("native KDA operands overlap"));
                }
            }
        }
        Ok(Self {
            qkv,
            gate,
            beta,
            a_log,
            dt_bias,
            state,
            output,
            workspace,
            layout,
            tokens: tokens as u32,
            reserved: 0,
            stream,
        })
    }
}

pub const fn available() -> bool {
    cfg!(all(feature = "cuda", atlas_glm_kda))
}

/// Resolve native code and validate shared-memory requirements on the current
/// CUDA device before taking the engine's free-memory snapshot for KV sizing.
/// This allocates no operand/scratch buffers and launches no kernels.
pub fn preload() -> Result<(), Error> {
    #[cfg(all(feature = "cuda", atlas_glm_kda))]
    {
        unsafe extern "C" {
            fn atlas_glm_kda_preload() -> i32;
        }
        // SAFETY: no pointers cross the ABI; CUDA validates the current device.
        let status = unsafe { atlas_glm_kda_preload() };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Preload(status))
        }
    }
    #[cfg(not(all(feature = "cuda", atlas_glm_kda)))]
    {
        Err(Error::Unavailable)
    }
}

/// Enqueue native prefill without allocation, synchronization or external APIs.
///
/// # Safety
/// Every operand in `params` must be a live CUDA allocation of the promised
/// size on the current device, exclusively owned or otherwise stream-ordered
/// for this operation. Output, scratch and recurrent state are mutated.
pub unsafe fn launch(params: &Params) -> Result<(), Error> {
    #[cfg(all(feature = "cuda", atlas_glm_kda))]
    {
        unsafe extern "C" {
            fn atlas_glm_kda_prefill(params: *const Params) -> i32;
        }
        // SAFETY: caller guarantees device allocation/stream ownership; the
        // constructor checks the complete bounded layout and disjoint ranges.
        let status = unsafe { atlas_glm_kda_prefill(params) };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Launch(status))
        }
    }
    #[cfg(not(all(feature = "cuda", atlas_glm_kda)))]
    {
        let _ = params;
        Err(Error::Unavailable)
    }
}

#[cfg(test)]
#[path = "glm_kda/tests.rs"]
mod tests;
