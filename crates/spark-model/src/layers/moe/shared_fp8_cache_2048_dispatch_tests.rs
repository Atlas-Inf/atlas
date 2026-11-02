// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn explicit_2048_installed_dispatch_slabs_with_exact_row_offsets_and_no_oracle_claim() {
    const CHILD: &str = "ATLAS_TEST_SHARED_PREFILL_2048";
    if std::env::var_os(CHILD).is_none() {
        let full = concat!(
            module_path!(),
            "::explicit_2048_installed_dispatch_slabs_with_exact_row_offsets_and_no_oracle_claim"
        );
        let test = full.split_once("::").expect("crate-qualified test").1;
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test])
            .env(CHILD, "1")
            .env("ATLAS_GLM_PREFILL_2048", "1")
            .env("ATLAS_GLM_PREFILL_4096", "0")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("running 1 test"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    fixture(2052, |layer, ctx, gpu| {
        layer.shared_fp8_cache.verify = true;
        let before = gpu.effects.load(Ordering::Relaxed);
        for p in 0..3 {
            for rows in [1024u32, 1025, 2048, 2049, 2052] {
                gpu.launches.lock().unwrap().clear();
                let (cache, output, n, k, _) = projection(layer, ctx, p);
                let input = ctx.buffers.norm_output();
                layer
                    .run_shared_fp8_cache(p, input, cache, output, rows, n, k, ctx, 19)
                    .unwrap();
                let launches = gpu.launches.lock().unwrap();
                assert_eq!(launches.len(), rows.div_ceil(1024) as usize);
                for (i, launch) in launches.iter().enumerate() {
                    let start = i as u32 * 1024;
                    let count = (rows - start).min(1024);
                    assert_eq!(launch.kernel, 777);
                    assert_eq!(launch.stream, 19);
                    assert_eq!(launch.grid, [n.div_ceil(128), count.div_ceil(64), 1]);
                    assert_eq!(
                        launch.args,
                        vec![
                            Arg::Ptr(input.offset(start as usize * k as usize * 2)),
                            Arg::Ptr(cache),
                            Arg::Ptr(output.offset(start as usize * n as usize * 2)),
                            Arg::Bytes(count.to_ne_bytes().to_vec()),
                            Arg::Bytes(n.to_ne_bytes().to_vec()),
                            Arg::Bytes(k.to_ne_bytes().to_vec()),
                        ]
                    );
                }
                assert_eq!(layer.shared_fp8_cache.checked.load(Ordering::Relaxed), 0);
            }
        }
        assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
    });
    for capacity in [2048, 2053] {
        fixture(capacity, |layer, ctx, gpu| {
            let (cache, output, n, k, _) = projection(layer, ctx, 0);
            let rows = if capacity == 2048 { 2049 } else { 2053 };
            assert!(
                layer
                    .run_shared_fp8_cache(
                        0,
                        ctx.buffers.norm_output(),
                        cache,
                        output,
                        rows,
                        n,
                        k,
                        ctx,
                        19
                    )
                    .is_err()
            );
            assert!(gpu.launches.lock().unwrap().is_empty());
        });
    }
}
