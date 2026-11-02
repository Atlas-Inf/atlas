// SPDX-License-Identifier: AGPL-3.0-only
//! Exact GLM M5 router ABI, distinct from the old BN16 launch geometry.
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn glm_router_bn4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0
            && !input.is_null()
            && !weight.is_null()
            && !output.is_null()
            && input.0.is_multiple_of(8)
            && weight.0.is_multiple_of(8)
            && output.0.is_multiple_of(2),
        "GLM BN4 router launch alignment/handle"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([72, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(5)
        .arg_u32(288)
        .arg_u32(4096)
        .launch(stream)
}
