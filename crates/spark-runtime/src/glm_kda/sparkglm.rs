// SPDX-License-Identifier: AGPL-3.0-only

//! SparkGLM 0.2.0's KDA prefill (`ATLAS_KDA_FLASH_LIBRARY`): MoonshotAI's
//! FlashKDA with SparkGLM's FP32-recurrence patches and its C bridge
//! (`libatlas_mango_flash.so` there), compiled into this crate from
//! `cuda/glm_kda_fp32/` with SparkGLM's flags. Unlike [`super::launch`], the
//! recurrent state stays FP32 for the whole call. The bridge transposes the
//! canonical key-major state into an aux buffer, runs the library over packed
//! (or, [`prefill_planes`], planar) q / k / v and transposes it back.

use super::Error;

/// Local KDA heads the bridge serves: TP4's 16 and TP2's 32.
pub const HEADS: [usize; 2] = [16, 32];
/// Rows one call accepts (the bridge rejects any other count).
pub const ROWS: std::ops::RangeInclusive<usize> = super::MIN_ROWS..=super::MAX_ROWS;
const DIM: usize = super::DIM;

fn check(tokens: usize, heads: usize) -> Result<(), Error> {
    if !ROWS.contains(&tokens) {
        return Err(Error::Invalid("SparkGLM KDA requires 2048..=4100 rows"));
    }
    if !HEADS.contains(&heads) {
        return Err(Error::Invalid("SparkGLM KDA requires 16 or 32 heads"));
    }
    Ok(())
}

/// Bytes of one `[tokens, heads, 128]` BF16 plane.
pub fn plane_bytes(tokens: usize, heads: usize) -> usize {
    tokens * heads * DIM * 2
}

/// Bytes of one layer's FP32 recurrent state.
pub fn state_bytes(heads: usize) -> usize {
    heads * DIM * DIM * 4
}

/// The bridge's aux scratch: the transposed state, the head-major beta and
/// its one-sequence metadata (`cu_seqlens[2]`, `slot[1]`), 128-aligned.
pub fn aux_bytes(tokens: usize, heads: usize) -> usize {
    (state_bytes(heads) + tokens * heads * 2).next_multiple_of(128) + 20
}

/// The library's workspace for one call, from the compiled library itself.
pub fn workspace_bytes(tokens: usize, heads: usize) -> Result<usize, Error> {
    check(tokens, heads)?;
    #[cfg(all(feature = "cuda", atlas_glm_kda))]
    {
        unsafe extern "C" {
            fn atlas_flash_kda_workspace_size(tokens: i32, heads: i32, sequences: i32) -> i64;
        }
        // SAFETY: plain arithmetic over validated sizes; no pointers.
        let bytes = unsafe { atlas_flash_kda_workspace_size(tokens as i32, heads as i32, 1) };
        usize::try_from(bytes)
            .ok()
            .filter(|&b| b > 0)
            .ok_or(Error::Invalid("SparkGLM KDA workspace size"))
    }
    #[cfg(not(all(feature = "cuda", atlas_glm_kda)))]
    {
        Err(Error::Unavailable)
    }
}

/// Device operands of one call, by address.
#[derive(Clone, Copy, Debug)]
pub struct Operands {
    pub gate: u64,
    pub beta: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub state: u64,
    pub output: u64,
    /// `(address, capacity)` of the library workspace and the bridge's aux.
    pub workspace: (u64, usize),
    pub aux: (u64, usize),
}

/// FlashKDA's scale and the KDA gate's lower bound, as SparkGLM passes them.
#[cfg(all(feature = "cuda", atlas_glm_kda))]
fn scale() -> f32 {
    1.0 / (DIM as f32).sqrt()
}
#[cfg(all(feature = "cuda", atlas_glm_kda))]
const LOWER_BOUND: f32 = -5.0;

#[cfg(all(feature = "cuda", atlas_glm_kda))]
fn status(code: i32) -> Result<(), Error> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Launch(code))
    }
}

/// One call over interleaved `[tokens, 3, heads, 128]` `qkv`, copied into
/// `packed` (`(address, capacity)`, three planes) first.
///
/// # Safety
/// Every address must be a live device allocation of the promised size on the
/// current device, stream-ordered on `stream`, and the ranges disjoint. The
/// state, output and scratch are mutated; never retry a failed call.
pub unsafe fn prefill(
    qkv: u64,
    packed: (u64, usize),
    o: &Operands,
    tokens: usize,
    heads: usize,
    stream: u64,
) -> Result<(), Error> {
    check(tokens, heads)?;
    #[cfg(all(feature = "cuda", atlas_glm_kda))]
    {
        use std::ffi::c_void;
        unsafe extern "C" {
            fn atlas_mango_flash_prefill(
                qkv: *const c_void,
                gate: *const c_void,
                rawbeta: *const c_void,
                a: *const c_void,
                bias: *const c_void,
                state: *mut c_void,
                out: *mut c_void,
                packed: *mut c_void,
                workspace: *mut c_void,
                aux: *mut c_void,
                packed_bytes: u64,
                workspace_bytes: u64,
                aux_bytes: u64,
                tokens: i32,
                heads: i32,
                scale: f32,
                lower: f32,
                stream: *mut c_void,
            ) -> i32;
        }
        let p = |x: u64| x as *mut c_void;
        // SAFETY: the caller's contract above; the bridge validates shapes and
        // capacities before any launch.
        status(unsafe {
            atlas_mango_flash_prefill(
                p(qkv),
                p(o.gate),
                p(o.beta),
                p(o.a_log),
                p(o.dt_bias),
                p(o.state),
                p(o.output),
                p(packed.0),
                p(o.workspace.0),
                p(o.aux.0),
                packed.1 as u64,
                o.workspace.1 as u64,
                o.aux.1 as u64,
                tokens as i32,
                heads as i32,
                scale(),
                LOWER_BOUND,
                p(stream),
            )
        })
    }
    #[cfg(not(all(feature = "cuda", atlas_glm_kda)))]
    {
        let _ = (qkv, packed, o, stream);
        Err(Error::Unavailable)
    }
}

/// [`prefill`] over q / k / v already in their own `[tokens, heads, 128]`
/// planes: the same library call on the same values, without the copy.
///
/// # Safety
/// As [`prefill`].
pub unsafe fn prefill_planes(
    [q, k, v]: [u64; 3],
    o: &Operands,
    tokens: usize,
    heads: usize,
    stream: u64,
) -> Result<(), Error> {
    check(tokens, heads)?;
    #[cfg(all(feature = "cuda", atlas_glm_kda))]
    {
        use std::ffi::c_void;
        unsafe extern "C" {
            fn atlas_mango_flash_prefill_planes(
                q: *const c_void,
                k: *const c_void,
                v: *const c_void,
                gate: *const c_void,
                rawbeta: *const c_void,
                a: *const c_void,
                bias: *const c_void,
                state: *mut c_void,
                out: *mut c_void,
                workspace: *mut c_void,
                aux: *mut c_void,
                workspace_bytes: u64,
                aux_bytes: u64,
                tokens: i32,
                heads: i32,
                scale: f32,
                lower: f32,
                stream: *mut c_void,
            ) -> i32;
        }
        let p = |x: u64| x as *mut c_void;
        // SAFETY: as `prefill`.
        status(unsafe {
            atlas_mango_flash_prefill_planes(
                p(q),
                p(k),
                p(v),
                p(o.gate),
                p(o.beta),
                p(o.a_log),
                p(o.dt_bias),
                p(o.state),
                p(o.output),
                p(o.workspace.0),
                p(o.aux.0),
                o.workspace.1 as u64,
                o.aux.1 as u64,
                tokens as i32,
                heads as i32,
                scale(),
                LOWER_BOUND,
                p(stream),
            )
        })
    }
    #[cfg(not(all(feature = "cuda", atlas_glm_kda)))]
    {
        let _ = (q, k, v, o, stream);
        Err(Error::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_sparkglm_bridge_arithmetic() {
        // bridge_check: state + beta, 128-aligned, then int64 cu[2] + int32 slot.
        assert_eq!(state_bytes(32), 2 * 1024 * 1024);
        assert_eq!(aux_bytes(4096, 32), 2 * 1024 * 1024 + 4096 * 32 * 2 + 20);
        assert_eq!(aux_bytes(4100, 16), (1 << 20) + 131_200 + 20);
        assert_eq!(plane_bytes(4096, 16), 4096 * 16 * 128 * 2);
    }

    #[test]
    fn shapes_outside_the_bridge_fail_before_any_call() {
        for (tokens, heads) in [(2047, 32), (4101, 32), (4096, 8), (4096, 64), (0, 16)] {
            assert!(matches!(
                workspace_bytes(tokens, heads),
                Err(Error::Invalid(_))
            ));
        }
        assert!(check(2048, 16).is_ok() && check(4100, 32).is_ok());
    }
}
