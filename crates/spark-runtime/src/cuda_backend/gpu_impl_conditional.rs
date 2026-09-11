// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use cudarc::driver::sys;
use std::ffi::c_void;

use super::{AtlasCudaBackend, cuDriverGetVersion};
use crate::gpu::{
    ConditionalGraphTemplate, ConditionalNodeKind, DevicePtr, GraphHandle, KernelHandle,
};

fn check(status: sys::CUresult, operation: &str) -> Result<()> {
    if status as u32 != 0 {
        bail!("{operation} failed: {status:?}");
    }
    Ok(())
}

fn graph(raw: u64) -> sys::CUgraph {
    raw as usize as sys::CUgraph
}

fn stream(raw: u64) -> sys::CUstream {
    raw as usize as sys::CUstream
}

impl AtlasCudaBackend {
    pub(super) fn conditional_nodes_supported_cu(&self) -> bool {
        if cfg!(atlas_scale) {
            return false;
        }
        let mut driver = 0;
        unsafe { cuDriverGetVersion(&mut driver) == 0 && driver >= 12_030 }
    }

    pub(super) fn create_conditional_graph_cu(
        &self,
        kind: ConditionalNodeKind,
        predicate: DevicePtr,
        setter: KernelHandle,
        body_count: usize,
    ) -> Result<ConditionalGraphTemplate> {
        if !self.conditional_nodes_supported_cu() {
            bail!("CUDA conditional nodes require a CUDA 12.3 or newer NVIDIA driver");
        }
        let valid_count = match kind {
            ConditionalNodeKind::If => matches!(body_count, 1 | 2),
            ConditionalNodeKind::While => body_count == 1,
        };
        if !valid_count || predicate.is_null() || setter.0 == 0 {
            bail!("invalid CUDA conditional graph predicate, setter, or body count");
        }

        let mut parent: sys::CUgraph = std::ptr::null_mut();
        check(
            unsafe { sys::cuGraphCreate(&mut parent, 0) },
            "cuGraphCreate",
        )?;
        let result = (|| {
            let context = self.cuda_ctx as usize as sys::CUcontext;
            let mut conditional_handle: sys::CUgraphConditionalHandle = 0;
            check(
                unsafe {
                    sys::cuGraphConditionalHandleCreate(
                        &mut conditional_handle,
                        parent,
                        context,
                        0,
                        0,
                    )
                },
                "cuGraphConditionalHandleCreate",
            )?;

            let mut handle_arg = conditional_handle;
            let mut predicate_arg = predicate.0;
            let mut kernel_args = [
                &mut handle_arg as *mut u64 as *mut c_void,
                &mut predicate_arg as *mut u64 as *mut c_void,
            ];
            let kernel = sys::CUDA_KERNEL_NODE_PARAMS {
                func: setter.0 as usize as sys::CUfunction,
                gridDimX: 1,
                gridDimY: 1,
                gridDimZ: 1,
                blockDimX: 1,
                blockDimY: 1,
                blockDimZ: 1,
                sharedMemBytes: 0,
                kernelParams: kernel_args.as_mut_ptr(),
                extra: std::ptr::null_mut(),
                kern: std::ptr::null_mut(),
                ctx: context,
            };
            let mut setter_node: sys::CUgraphNode = std::ptr::null_mut();
            check(
                unsafe {
                    sys::cuGraphAddKernelNode_v2(
                        &mut setter_node,
                        parent,
                        std::ptr::null(),
                        0,
                        &kernel,
                    )
                },
                "cuGraphAddKernelNode_v2(predicate)",
            )?;

            let conditional = sys::CUDA_CONDITIONAL_NODE_PARAMS {
                handle: conditional_handle,
                type_: match kind {
                    ConditionalNodeKind::If => {
                        sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF
                    }
                    ConditionalNodeKind::While => {
                        sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE
                    }
                },
                size: body_count as u32,
                // OUTPUT, not input. The driver overwrites this with a pointer to
                // its OWN CUDA-owned array of body graphs (valid for the node's
                // lifetime). Passing our own array here is ignored and yields
                // null bodies — the NVIDIA graphConditionalNodes sample reads
                // `cParams.conditional.phGraph_out[0]` after the add and never
                // assigns the field itself.
                phGraph_out: std::ptr::null_mut(),
                ctx: context,
            };
            let mut params: sys::CUgraphNodeParams = unsafe { std::mem::zeroed() };
            params.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
            params.__bindgen_anon_1.conditional = conditional;
            let dependencies = [setter_node];
            let mut conditional_node: sys::CUgraphNode = std::ptr::null_mut();
            check(
                unsafe {
                    sys::cuGraphAddNode_v2(
                        &mut conditional_node,
                        parent,
                        dependencies.as_ptr(),
                        std::ptr::null(),
                        dependencies.len(),
                        &mut params,
                    )
                },
                "cuGraphAddNode_v2(conditional)",
            )?;
            let bodies_ptr = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
            if bodies_ptr.is_null() {
                bail!(
                    "CUDA returned no conditional body array — conditional nodes are \
                     unsupported on this driver"
                );
            }
            let bodies = unsafe { std::slice::from_raw_parts(bodies_ptr, body_count) };
            if bodies.iter().any(|body| body.is_null()) {
                bail!("CUDA returned a null conditional body graph");
            }
            Ok(ConditionalGraphTemplate {
                graph: parent as usize as u64,
                conditional_handle,
                bodies: [
                    bodies[0] as usize as u64,
                    bodies.get(1).copied().unwrap_or(std::ptr::null_mut()) as usize as u64,
                ],
                body_count,
                kind,
            })
        })();
        if result.is_err() {
            unsafe { sys::cuGraphDestroy(parent) };
        }
        result
    }

    pub(super) fn begin_conditional_branch_cu(
        &self,
        template: &ConditionalGraphTemplate,
        branch: usize,
        capture_stream: u64,
    ) -> Result<()> {
        if branch >= template.body_count {
            bail!(
                "conditional branch {branch} is outside {} bodies",
                template.body_count
            );
        }
        check(
            unsafe {
                sys::cuStreamBeginCaptureToGraph(
                    stream(capture_stream),
                    graph(template.bodies[branch]),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
            },
            "cuStreamBeginCaptureToGraph(conditional body)",
        )
    }

    pub(super) fn end_conditional_branch_cu(
        &self,
        template: &ConditionalGraphTemplate,
        branch: usize,
        capture_stream: u64,
    ) -> Result<()> {
        if branch >= template.body_count {
            bail!(
                "conditional branch {branch} is outside {} bodies",
                template.body_count
            );
        }
        let mut captured: sys::CUgraph = std::ptr::null_mut();
        check(
            unsafe { sys::cuStreamEndCapture(stream(capture_stream), &mut captured) },
            "cuStreamEndCapture(conditional body)",
        )?;
        if captured != graph(template.bodies[branch]) {
            bail!("CUDA conditional body capture returned an unexpected graph");
        }
        Ok(())
    }

    pub(super) fn instantiate_conditional_graph_cu(
        &self,
        template: &ConditionalGraphTemplate,
        dot_path: Option<&std::path::Path>,
    ) -> Result<GraphHandle> {
        if let Some(path) = dot_path {
            let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
            check(
                unsafe { sys::cuGraphDebugDotPrint(graph(template.graph), path.as_ptr(), 0) },
                "cuGraphDebugDotPrint(conditional)",
            )?;
        }
        let mut executable: sys::CUgraphExec = std::ptr::null_mut();
        check(
            unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, graph(template.graph), 0) },
            "cuGraphInstantiateWithFlags(conditional)",
        )?;
        Ok(GraphHandle(executable as usize as u64))
    }

    pub(super) fn destroy_conditional_graph_template_cu(
        &self,
        template: &ConditionalGraphTemplate,
    ) {
        if template.graph != 0 {
            unsafe { sys::cuGraphDestroy(graph(template.graph)) };
        }
    }
}
