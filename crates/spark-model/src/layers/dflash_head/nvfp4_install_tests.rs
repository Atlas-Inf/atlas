// SPDX-License-Identifier: AGPL-3.0-only
//! Real installer eligibility; subprocesses isolate startup env and TC OnceLock.
use super::super::{DflashQuantization, free_state_tests::zero_head};
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use spark_runtime::gpu::{KernelHandle, mock::MockGpuBackend};

#[test]
fn integration_nvfp4_installs_for_the_actual_execution_route() {
    const CHILD: &str = "ATLAS_NVFP4_INSTALL_TEST_CHILD";
    let Ok(mode) = std::env::var(CHILD) else {
        let name = format!(
            "{}::integration_nvfp4_installs_for_the_actual_execution_route",
            module_path!().split_once("::").unwrap().1
        );
        for mode in ["scalar", "tc"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            for (key, _) in std::env::vars().filter(|(k, _)| k.starts_with("ATLAS_DFLASH_")) {
                child.env_remove(key);
            }
            let output = child
                .args(["--exact", &name, "--nocapture"])
                .env(CHILD, mode)
                .env(
                    "ATLAS_DFLASH_NVFP4_TC",
                    if mode == "tc" { "1" } else { "0" },
                )
                .env("ATLAS_W4A16_TC", if mode == "tc" { "1" } else { "0" })
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{mode}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let gpu = MockGpuBackend::new();
    let _tiers = W4a16BatchmTiers::resolve(&gpu, true);
    for (gamma, scalar_present) in [
        (0, false),
        (4, true),
        (8, true),
        (8, false),
        (16, true),
        (65, false),
    ] {
        let mut head = zero_head();
        head.gamma = gamma;
        if scalar_present {
            head.kernels.w4a16_gemv_batch4 = KernelHandle(1);
            head.kernels.w4a16_gemv_batch8 = KernelHandle(1);
        }
        // No numerical emulation: empty layers isolate the real install gate.
        head.install_twins(&gpu).unwrap();
        let expected = gamma > 0 && (mode == "tc" || (gamma <= 8 && scalar_present));
        assert_eq!(
            matches!(head.quant, DflashQuantization::Nvfp4Weights),
            expected,
            "mode={mode}, gamma={gamma}, scalar_present={scalar_present}"
        );
        assert_eq!(head.twins.nvfp4_tc, expected && mode == "tc");
    }
}
