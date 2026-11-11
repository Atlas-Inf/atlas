// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_QSA_KEY_ONLY=1` (default off): the staged decode ingest
//! (`qsa_staged.rs`) projects only the raw key.
//!
//! An inert row's ingest needs one thing from the indexer's qk projection:
//! the raw key, its last `hd` columns. The staged half still ran the full
//! `[1, hidden] x [(n_heads + 1) * hd, hidden]^T` projection (640 outputs on
//! Qwen3.8-Flash-Next) into scratch row 0 and copied the key out, once per
//! row and attention layer: at C8 x K=4, 32 cuBLASLt GEMVs of ~10 us plus 32
//! copies a layer, ~350 of each a step inside the verify graph (nsys, TP=EP=2
//! rank 0). Projecting the `hd` key rows of the weight straight into the
//! staging row reads a fifth of the weight and drops the copy: 32 rows in a
//! graph, 219 -> 104 us on GB10.
//!
//! Exact only if cuBLASLt computes each key output the same way at N = hd as
//! at N = 640. On the runtime image both shapes take the same algorithm
//! (algo 13, the GEMV, no split-K) and 1,048,576 key outputs over 8,192
//! adversarial rows came out byte-identical. That is a property of the
//! library and GPU, not of the code, so it is CHECKED, once per process,
//! before the first key-only launch: [`QsaIndexer::key_only`] runs both forms
//! on a probe and keeps the full projection unless every key byte agrees. A
//! batched (strided) call, by contrast, is NOT exact: 1-3 outputs in 20,480
//! differ at 2..32 rows, so it is not used.

use std::sync::OnceLock;

use super::*;

/// `ATLAS_QWEN4EXP_QSA_KEY_ONLY=1`, read once.
fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_QSA_KEY_ONLY").as_deref() == Ok("1"))
}

/// The probe's verdict, decided once (every indexer has the same shape).
static VERDICT: OnceLock<bool> = OnceLock::new();

/// Probe rows: activations from a fixed SplitMix64 stream, a wide range of
/// magnitudes mixed in (rounding differences surface in the BF16 outputs).
const PROBE_ROWS: usize = 64;

fn probe_activations(rows: usize, hidden: usize) -> Vec<u8> {
    let mut s = 0x5EED_0F0A_5A5Au64;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..rows * hidden)
        .flat_map(|_| {
            let r = next();
            let unit = ((r >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            let scale = if r % 13 == 0 { 64.0 } else { 1.0 };
            half::bf16::from_f32(unit * scale).to_le_bytes()
        })
        .collect()
}

impl QsaIndexer {
    /// The raw-key rows of the qk weight, `[hd, hidden]`: its last `hd`
    /// output rows.
    pub(super) fn key_proj_w(&self) -> DevicePtr {
        self.qk_proj_w
            .offset(self.n_heads as usize * self.hd as usize * self.hidden as usize * 2)
    }

    /// Whether the staged ingest projects only the raw key: the switch, and
    /// the probe's verdict. Undecided inside a capture (no probe can run
    /// there), which keeps the full projection for that capture; both forms
    /// stage the same bytes once the probe has passed, so the graphs may mix.
    pub(super) fn key_only(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<bool> {
        if !requested() {
            return Ok(false);
        }
        if let Some(&v) = VERDICT.get() {
            return Ok(v);
        }
        if gpu.stream_is_capturing(stream) {
            return Ok(false);
        }
        let differing = self.probe_key_only(gpu, stream)?;
        let ok = *VERDICT.get_or_init(|| differing == 0);
        if ok {
            tracing::info!(
                "QSA key-only staged ingest ON (ATLAS_QWEN4EXP_QSA_KEY_ONLY=1): probe of \
                 {PROBE_ROWS} rows byte-identical to the full projection's key columns"
            );
        } else {
            tracing::warn!(
                "QSA key-only staged ingest OFF: {differing} key bytes of the probe differ \
                 from the full projection's; keeping the full projection"
            );
        }
        Ok(ok)
    }

    /// Key bytes that differ between the full projection's key columns and
    /// the key-only projection, over [`PROBE_ROWS`] probe rows, one row a
    /// call as the staged ingest calls them. Synchronous; temporary buffers.
    fn probe_key_only(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<usize> {
        let (hd, hidden, qkw) = (self.hd as usize, self.hidden as usize, self.qk_width());
        let act = gpu.alloc(PROBE_ROWS * hidden * 2)?;
        let full = gpu.alloc(PROBE_ROWS * qkw * 2)?;
        let key = gpu.alloc(PROBE_ROWS * hd * 2)?;
        let run = || -> Result<usize> {
            gpu.copy_h2d(&probe_activations(PROBE_ROWS, hidden), act)?;
            for r in 0..PROBE_ROWS {
                let a = act.offset(r * hidden * 2);
                ops::cublas_bf16_proj_dense(
                    a,
                    self.qk_proj_w,
                    full.offset(r * qkw * 2),
                    1,
                    qkw as u32,
                    self.hidden,
                    stream,
                )?;
                ops::cublas_bf16_proj_dense(
                    a,
                    self.key_proj_w(),
                    key.offset(r * hd * 2),
                    1,
                    self.hd,
                    self.hidden,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            let mut f = vec![0u8; PROBE_ROWS * qkw * 2];
            let mut k = vec![0u8; PROBE_ROWS * hd * 2];
            gpu.copy_d2h(full, &mut f)?;
            gpu.copy_d2h(key, &mut k)?;
            Ok(key_bytes_differing(&f, &k, qkw * 2, hd * 2))
        };
        let out = run();
        for p in [act, full, key] {
            gpu.free(p)?;
        }
        out
    }
}

/// Bytes of `key` (rows `key_row` apart) that differ from the last
/// `key_row` bytes of each `full` row (rows `full_row` apart).
fn key_bytes_differing(full: &[u8], key: &[u8], full_row: usize, key_row: usize) -> usize {
    full.chunks(full_row)
        .zip(key.chunks(key_row))
        .map(|(f, k)| {
            f[full_row - key_row..]
                .iter()
                .zip(k)
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_comparison_reads_the_key_columns() {
        // Two rows of [q q q | k k]: equal keys, then one differing byte.
        let full = [1, 2, 3, 7, 8, 4, 5, 6, 9, 9];
        assert_eq!(key_bytes_differing(&full, &[7, 8, 9, 9], 5, 2), 0);
        assert_eq!(key_bytes_differing(&full, &[7, 8, 9, 0], 5, 2), 1);
    }

    #[test]
    fn the_probe_is_deterministic_and_wide() {
        let a = probe_activations(2, 64);
        assert_eq!(a, probe_activations(2, 64));
        let vals: Vec<f32> = a
            .chunks(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        assert!(vals.iter().any(|v| v.abs() > 2.0) && vals.iter().any(|v| v.abs() < 0.5));
    }
}
