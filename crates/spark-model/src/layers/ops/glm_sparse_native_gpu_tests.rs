// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit, ignored maintenance-window checks of actual Atlas sparse dispatch.
use super::{Plan, Span, dispatch, plan};
use anyhow::{Result, ensure};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "glm_sparse_native_gpu_fixture.rs"]
mod fixture;
use fixture::{BLOCKS, END, Host, ROWS, START, read_f32, read_u16, value};

struct Allocation<'a> {
    gpu: &'a dyn GpuBackend,
    base: DevicePtr,
    bytes: usize,
}
impl Allocation<'_> {
    fn ptr(&self) -> DevicePtr {
        self.base.offset(256)
    }
    fn read(&self) -> Result<Vec<u8>> {
        download(self.gpu, self.ptr(), self.bytes)
    }
    fn guards(&self) -> Result<()> {
        for ptr in [self.base, self.ptr().offset(self.bytes)] {
            ensure!(
                download(self.gpu, ptr, 256)?.iter().all(|&x| x == 0xa5),
                "sparse operand guard overwritten"
            );
        }
        Ok(())
    }
}
impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        if self.base.0 != 0 {
            let _ = self.gpu.free(self.base);
        }
    }
}
struct Stream<'a>(&'a dyn GpuBackend, u64);
impl Drop for Stream<'_> {
    fn drop(&mut self) {
        let _ = self.0.destroy_stream(self.1);
    }
}
fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut output = vec![0; bytes];
    gpu.copy_d2h(ptr, &mut output)?;
    Ok(output)
}

fn run(host: &Host) -> Result<(Vec<u8>, Vec<u8>)> {
    let target =
        atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4").ok_or_else(|| {
            anyhow::anyhow!("build the GLM CUDA target; ATLAS_SKIP_BUILD must be unset")
        })?;
    let gpu = AtlasCudaBackend::new(0, &target.modules)?;
    dispatch::resolve(&gpu)?;
    ensure!(
        gpu.free_memory()? > 4 * 1024 * 1024 * 1024,
        "GPU requires4GiB free; run only in an exclusive maintenance window"
    );
    let stream = Stream(&gpu, gpu.create_stream()?);
    let (required, metadata) =
        plan::required_bytes(ROWS, END, BLOCKS, END / 16).map_err(anyhow::Error::msg)?;
    ensure!(
        required.iter().sum::<usize>() + 512 * 10 < 1024 * 1024 * 1024,
        "fixture exceeds1GiB"
    );
    let mut buffers = Vec::new();
    for bytes in required {
        let base = gpu.alloc(bytes + 512)?;
        let allocation = Allocation {
            gpu: &gpu,
            base,
            bytes,
        };
        gpu.memset(base, 0xa5, bytes + 512)?;
        buffers.push(allocation);
    }
    let inputs = host.inputs();
    for (buffer, bytes) in buffers.iter().zip(&inputs) {
        ensure!(bytes.len() == buffer.bytes, "host operand size mismatch");
        gpu.copy_h2d(bytes, buffer.ptr())?;
    }
    let spans = std::array::from_fn(|i| Span {
        ptr: buffers[i].ptr().0,
        bytes: buffers[i].bytes,
    });
    let plan =
        Plan::new(ROWS, START, BLOCKS, END / 16, spans, stream.1).map_err(anyhow::Error::msg)?;
    // Exactly the serving implementation, including capability checks and
    // per-source typed KernelLaunch calls. No duplicated dispatch or host ABI.
    dispatch::execute(&plan, &gpu)?;
    gpu.synchronize(stream.1)?;
    for buffer in &buffers {
        buffer.guards()?;
    }
    for (buffer, expected) in buffers.iter().zip(&inputs) {
        ensure!(
            buffer.read()? == *expected,
            "native sparse mutated an input"
        );
    }
    fixture::check_prep(host, &buffers[4].read()?, &buffers[5].read()?)?;
    let meta = buffers[6].read()?;
    check_metadata(host, &meta, &metadata)?;
    let output = buffers[9].read()?;
    let lse = meta[metadata.offsets[7]..metadata.offsets[7] + ROWS * 32 * 4].to_vec();
    ensure!(
        output
            .chunks_exact(2)
            .all(|x| value(u16::from_le_bytes(x.try_into().unwrap())).is_finite()),
        "nonfinite output"
    );
    ensure!(
        lse.chunks_exact(4)
            .all(|x| f32::from_le_bytes(x.try_into().unwrap()).is_finite()),
        "nonfinite LSE"
    );
    println!(
        "{{\"fixture_device_bytes\":{},\"kernel_set_hash\":\"{}\",\"checks\":\"inputs/redzones/prep/metadata\"}}",
        required.iter().sum::<usize>() + 5120,
        atlas_kernels::KERNEL_SET_HASH
    );
    // Explicit successful cleanup reports failures; RAII still covers errors.
    for buffer in &mut buffers {
        gpu.free(buffer.base)?;
        buffer.base = DevicePtr::NULL;
    }
    drop(buffers);
    Ok((output, lse))
}

fn check_metadata(host: &Host, bytes: &[u8], layout: &plan::Metadata) -> Result<()> {
    let integer = |array: usize, index: usize| {
        let offset = layout.offsets[array] + index * 4;
        i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    };
    for row in 0..ROWS {
        let count = (START + row + 1) % 4;
        for col in 0..2048 {
            ensure!(
                integer(0, row * 2048 + col) == host.selected[row * 2051 + col],
                "main selection mismatch"
            );
            let tail = if col < count {
                host.selected[row * 2051 + 2048 + col]
            } else if col == 0 && count == 0 {
                0
            } else {
                -1
            };
            ensure!(
                integer(1, row * 2048 + col) == tail,
                "tail selection mismatch"
            );
        }
        ensure!(
            integer(2, row) == 2048
                && integer(3, row) == count.max(1) as i32
                && integer(4, row) == count as i32,
            "length/count mismatch"
        );
    }
    let sizes = [
        ROWS * 2048 * 4,
        ROWS * 2048 * 4,
        ROWS * 4,
        ROWS * 4,
        ROWS * 4,
        ROWS * 32 * 4,
        ROWS * 32 * 4,
        ROWS * 32 * 4,
    ];
    let mut end = 0;
    for (offset, size) in layout.offsets.into_iter().zip(sizes) {
        ensure!(
            bytes[end..offset].iter().all(|&x| x == 0xa5),
            "metadata padding overwritten"
        );
        end = offset + size;
    }
    Ok(())
}

#[test]
#[ignore = "requires exclusive GB10 GPU, real GLM kernel build and externally capped process"]
fn analytic_sparse_attention_through_atlas_dispatch() -> Result<()> {
    for kind in ["zero", "constant", "onehot"] {
        let mut host = Host::new(if kind == "constant" { 1.0 } else { 0.0 });
        if kind == "onehot" {
            host.token_mut(37)[0] = bf16::from_f32(1.0).to_bits();
        }
        let (output, lse) = run(&host)?;
        let mut max_lse_ulps = 0;
        for row in 0..ROWS {
            let length = 2048 + (START + row + 1) % 4;
            let expected_lse = (length as f64).log2() as f32;
            for head in 0..32 {
                let actual_lse = read_f32(&lse, (row * 32 + head) * 4);
                let ulps = actual_lse.to_bits().abs_diff(expected_lse.to_bits());
                max_lse_ulps = max_lse_ulps.max(ulps);
                // Fixed before GPU execution: analytic uniform logits, positive
                // FP32 LSE; allows rounding across fast log2 and main/tail merge.
                ensure!(
                    ulps <= 8,
                    "analytic {kind} log2-LSE error {ulps}ULP at row{row}/head{head}"
                );
                for dim in 0..512 {
                    let expected = match kind {
                        "constant" => 1.0,
                        "onehot" if dim == 0 => 1.0 / length as f32,
                        _ => 0.0,
                    };
                    ensure!(
                        read_u16(&output, ((row * 32 + head) * 512 + dim) * 2)
                            == bf16::from_f32(expected).to_bits(),
                        "analytic {kind} output mismatch at {row}/{head}/{dim}"
                    );
                }
            }
        }
        println!("{{\"analytic\":\"{kind}\",\"passed\":true,\"max_lse_ulps\":{max_lse_ulps}}}");
    }
    Ok(())
}

#[test]
#[ignore = "requires explicit sampled safetensors fixture and exclusive capped GB10 GPU"]
fn sampled_scaffold_comparison_through_atlas_dispatch() -> Result<()> {
    let (host, samples) = fixture::sampled_fixture()?;
    let (output, lse) = run(&host)?;
    for sample in samples {
        let mut max_error = 0f64;
        let mut different = 0;
        for dim in 0..512 {
            let actual = read_u16(&output, ((sample.row * 32 + sample.head) * 512 + dim) * 2);
            different += usize::from(actual != sample.output[dim]);
            max_error = max_error
                .max((f64::from(value(actual)) - f64::from(value(sample.output[dim]))).abs());
        }
        let lse_error = (read_f32(&lse, (sample.row * 32 + sample.head) * 4) - sample.lse).abs();
        // Stored source-built scaffold outputs are not an independent oracle.
        // Architecture/entry construction changed; report deviations for review,
        // never fit a new tolerance to these observations or claim full parity.
        println!(
            "{{\"scope\":\"sampled scaffold diagnostic\",\"qualified\":false,\"row\":{},\"head\":{},\"different_elements\":{},\"max_abs_error\":{},\"log2_lse_abs_error\":{}}}",
            sample.row, sample.head, different, max_error, lse_error
        );
    }
    Ok(())
}
