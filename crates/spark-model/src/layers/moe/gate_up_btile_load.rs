// SPDX-License-Identifier: AGPL-3.0-only
//! Load-time facade: owns the one scratch buffer and complete kernel family.
//! Private transaction authority; no loader/environment activation caller.
use super::{
    RepackWorkspace, binding::ValidatedTables, kernels::KernelFamily,
    native_source::NativeGateUpLayer, resident::Storage,
};
use crate::layers::moe::MoeLayer;
use crate::weight_loader::glm5::retirement::RetirementLog;
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

/// Minted only inside this exclusive load transaction. Public/native converter
/// guards remain strict; this permits only the existing internal shared/down
/// phases for the specific layer and backend currently being constructed.
pub(in crate::layers::moe) struct ConstructionToken {
    layer: usize,
    backend: usize,
}
impl ConstructionToken {
    pub(in crate::layers::moe) fn validate(
        &self,
        layer: &MoeLayer,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        anyhow::ensure!(
            self.layer == layer as *const MoeLayer as usize
                && self.backend == gpu as *const dyn GpuBackend as *const () as usize
                && matches!(layer.btile_storage, super::resident::Storage::Constructing),
            "B-tile internal phase construction owner mismatch"
        );
        Ok(())
    }
}

/// No serving caller or environment selection in the reader-closure partition.
/// Future loader activation must prevalidate every target before `prepare`.
pub(crate) struct BTileLoadSession<'g> {
    family: KernelFamily<'g>,
    workspace: RepackWorkspace<'g>,
}

impl<'g> BTileLoadSession<'g> {
    pub(crate) fn new(gpu: &'g dyn GpuBackend, config: &ModelConfig, stream: u64) -> Result<Self> {
        let family = KernelFamily::resolve(gpu, config, stream)?;
        let workspace = RepackWorkspace::new(gpu, stream)?;
        Ok(Self { family, workspace })
    }

    pub(crate) fn prepare(
        &mut self,
        layer: &mut MoeLayer,
        retirement: &RetirementLog<'_>,
        config: &ModelConfig,
        ordinal: usize,
    ) -> Result<()> {
        layer.btile_storage.require_legacy()?;
        super::preflight::layer(layer, config)?;
        let gpu = self.family.gpu;
        anyhow::ensure!(
            std::env::var("ATLAS_HOST_TRANSPOSE").as_deref() != Ok("1"),
            "B-tile requires native transpose"
        );
        anyhow::ensure!(
            gpu.kernel("transpose_u8", "transpose_u8")?.0 != 0,
            "missing shared transpose kernel"
        );
        layer.validate_btile_down(gpu, config, ordinal, retirement)?;
        let local: Vec<bool> = layer
            .weights
            .experts
            .iter()
            .map(|e| !e.gate_proj.is_null())
            .collect();
        let source = NativeGateUpLayer::from_live(
            retirement,
            config,
            ordinal,
            &local,
            gpu,
            self.workspace.stream,
        )?;
        super::preflight::sources(
            layer,
            &source,
            retirement,
            config,
            ordinal,
            gpu,
            self.workspace.stream,
        )?;
        let checked = ValidatedTables::read(&source, &self.family, layer)?;
        // Refuse mixed layouts/table aliases before transforming any payload.
        // The real shared transforms mint their own private allocation receipt.
        layer.btile_storage = Storage::Constructing;
        let token = ConstructionToken {
            layer: layer as *const MoeLayer as usize,
            backend: gpu as *const dyn GpuBackend as *const () as usize,
        };
        let result = (|| {
            layer.transpose_btile_shared(&token, gpu, config)?;
            let checked = checked.with_shared(&source, &self.family, layer)?;
            let scratch = self
                .workspace
                .scratch
                .ok_or_else(|| anyhow::anyhow!("closed repack workspace"))?;
            anyhow::ensure!(
                checked.tables.iter().all(|s| s.disjoint(scratch))
                    && checked
                        .shared
                        .as_ref()
                        .is_some_and(|(_, spans)| spans.iter().all(|s| s.disjoint(scratch))),
                "scratch aliases constructed owners"
            );
            let unpublished = self.workspace.repack(source, &self.family)?;
            let mut ready =
                super::resident::seal(unpublished, &self.family, layer, checked, config.ep_rank)?;
            let down = layer.transpose_btile_down(&token, gpu, config, ordinal, retirement)?;
            ready.complete_down(layer, down, config, retirement, gpu, scratch)?;
            layer.btile_storage = Storage::Published(ready);
            Ok(())
        })();
        if result.is_err() {
            layer.btile_storage = Storage::Failed;
        }
        result
    }

    pub(crate) fn close(self) -> Result<()> {
        self.workspace.close()
    }
}
