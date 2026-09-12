// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{GpuBackend, KernelArg};
use spark_runtime::graph_runtime::{
    BoundedLoopExitReason, BoundedLoopState, CaptureFailurePolicy, GraphCost, GraphFingerprint,
    GraphMode, GraphPayload, GraphPolicies, GraphRuntime, GraphRuntimeConfig, GraphSegment,
    PhasePolicy, ShapeBucket, SpeculativeAlgorithm,
};
use std::sync::Arc;

fn main() -> Result<()> {
    let backend = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let gpu: Arc<dyn GpuBackend> = backend;
    let capabilities = gpu.graph_capabilities();
    ensure!(
        capabilities.conditional_nodes,
        "CUDA conditional graph nodes are unavailable on this driver"
    );
    let policy = PhasePolicy {
        max_entries: 4,
        max_estimated_bytes: 16 * 1024 * 1024,
        capture_enabled: true,
        replay_enabled: true,
        prewarm_enabled: false,
    };
    let environment = gpu
        .graph_environment()
        .ok_or_else(|| anyhow::anyhow!("CUDA graph environment query failed"))?;
    let runtime = GraphRuntime::new(
        gpu.clone(),
        capabilities,
        GraphRuntimeConfig {
            mode: GraphMode::Piecewise,
            speculative_algorithm: SpeculativeAlgorithm::None,
            shape_buckets: vec![ShapeBucket {
                token_limit: 8,
                request_limit: 1,
            }],
            policies: GraphPolicies::new(policy, policy, policy, policy, policy)
                .map_err(anyhow::Error::msg)?,
            max_cache_entries: 4,
            max_cache_bytes: 16 * 1024 * 1024,
            fingerprint: GraphFingerprint {
                runtime: env!("CARGO_PKG_VERSION").into(),
                model: "conditional-microtest".into(),
                kernel_build: "graph-control".into(),
                device: environment.device,
                cuda: environment.cuda,
                driver: environment.driver,
                memory_layout: "u32-predicate-output".into(),
            },
            resource_generation: 1,
            compatibility_rules: Vec::new(),
            export_dir: None,
            prewarm_profile: None,
        },
        Arc::new(Default::default()),
    )
    .map_err(anyhow::Error::msg)?;
    let stream = gpu.default_stream();
    let predicate = gpu.alloc(4)?;
    let output = gpu.alloc(4)?;
    let setter = gpu.kernel("graph_control", "graph_set_conditional")?;
    let writer = gpu.kernel("graph_control", "graph_write_value")?;
    let identity = runtime.identity(
        GraphSegment::Conditional,
        GraphPayload::Verify {
            request_count: 1,
            row_count: 2,
            slots: vec![0],
            depths: vec![1],
            layout: Vec::new(),
        },
    )?;
    let graph = runtime.capture_if_else(
        identity,
        stream,
        GraphCost {
            estimated_bytes: 1024 * 1024,
            node_count: 3,
            child_count: 2,
            staging_bytes: 8,
        },
        predicate,
        setter,
        CaptureFailurePolicy::Retry,
        |gpu| {
            let value = 11u32.to_le_bytes();
            gpu.launch_typed(
                writer,
                [1, 1, 1],
                [1, 1, 1],
                0,
                stream,
                &[KernelArg::Buffer(output), KernelArg::Bytes(&value)],
            )
        },
        |gpu| {
            let value = 22u32.to_le_bytes();
            gpu.launch_typed(
                writer,
                [1, 1, 1],
                [1, 1, 1],
                0,
                stream,
                &[KernelArg::Buffer(output), KernelArg::Bytes(&value)],
            )
        },
    )?;

    for (condition, expected) in [(1u32, 11u32), (0, 22)] {
        gpu.copy_h2d(&condition.to_le_bytes(), predicate)?;
        runtime.launch(&graph, stream)?;
        gpu.synchronize(stream)?;
        let mut bytes = [0u8; 4];
        gpu.copy_d2h(output, &mut bytes)?;
        ensure!(u32::from_le_bytes(bytes) == expected);
    }

    let loop_state = BoundedLoopState {
        continuation: gpu.alloc(4)?,
        iteration: gpu.alloc(4)?,
        cancellation: gpu.alloc(4)?,
        exit_reason: gpu.alloc(4)?,
        output: gpu.alloc(4 * 4)?,
        output_capacity: 4,
        max_iterations: 4,
    };
    let loop_update = gpu.kernel("graph_control", "graph_while_update")?;
    let loop_emit = gpu.kernel("graph_control", "graph_loop_emit")?;
    let capture_loop = |slot: u32, stop_after: u32| -> Result<_> {
        let identity = runtime.identity(
            GraphSegment::LoopBody,
            GraphPayload::DecodeLoop {
                request_count: 1,
                max_iterations: loop_state.max_iterations,
                output_capacity: loop_state.output_capacity,
                slots: vec![slot],
            },
        )?;
        Ok(runtime.capture_bounded_while(
            identity,
            stream,
            GraphCost {
                estimated_bytes: 1024 * 1024,
                node_count: 3,
                child_count: 1,
                staging_bytes: 6 * 4,
            },
            loop_state,
            setter,
            loop_update,
            CaptureFailurePolicy::Retry,
            |gpu, state| {
                let stop_after = stop_after.to_le_bytes();
                gpu.launch_typed(
                    loop_emit,
                    [1, 1, 1],
                    [1, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Buffer(state.output),
                        KernelArg::Buffer(state.iteration),
                        KernelArg::Buffer(state.continuation),
                        KernelArg::Bytes(&stop_after),
                    ],
                )
            },
        )?)
    };
    let capped = capture_loop(0, 0)?;
    let eos = capture_loop(1, 2)?;
    let read_u32 = |ptr| -> Result<u32> {
        let mut bytes = [0u8; 4];
        gpu.copy_d2h(ptr, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    };

    runtime.prepare_bounded_loop(loop_state, stream, true, true)?;
    runtime.launch(&capped, stream)?;
    gpu.synchronize(stream)?;
    ensure!(read_u32(loop_state.iteration)? == 4);
    ensure!(read_u32(loop_state.exit_reason)? == BoundedLoopExitReason::IterationCap as u32);
    let mut emitted = [0u8; 16];
    gpu.copy_d2h(loop_state.output, &mut emitted)?;
    ensure!(
        emitted
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>()
            == [1, 2, 3, 4]
    );

    runtime.prepare_bounded_loop(loop_state, stream, true, true)?;
    runtime.launch(&eos, stream)?;
    gpu.synchronize(stream)?;
    ensure!(read_u32(loop_state.iteration)? == 2);
    ensure!(read_u32(loop_state.exit_reason)? == BoundedLoopExitReason::Stop as u32);

    runtime.prepare_bounded_loop(loop_state, stream, true, true)?;
    gpu.copy_h2d(&1u32.to_le_bytes(), loop_state.cancellation)?;
    runtime.launch(&capped, stream)?;
    gpu.synchronize(stream)?;
    ensure!(read_u32(loop_state.iteration)? == 1);
    ensure!(read_u32(loop_state.exit_reason)? == BoundedLoopExitReason::Cancelled as u32);

    runtime.shutdown()?;
    gpu.free(predicate)?;
    gpu.free(output)?;
    for ptr in [
        loop_state.continuation,
        loop_state.iteration,
        loop_state.cancellation,
        loop_state.exit_reason,
        loop_state.output,
    ] {
        gpu.free(ptr)?;
    }
    Ok(())
}
