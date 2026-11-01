// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;

use recording::{Arg, Event};

fn ptr(p: u64) -> Arg {
    Arg::Ptr(DevicePtr(p))
}

fn u32a(v: u32) -> Arg {
    Arg::Bytes(v.to_le_bytes().to_vec())
}

fn tiers() -> MxTiers {
    MxTiers {
        row_major: [KernelHandle(8), KernelHandle(16), KernelHandle(32)],
        tiled: [KernelHandle(108), KernelHandle(116), KernelHandle(132)],
        fp8b_scales: true,
    }
}

fn launches(gpu: &recording::Gpu) -> Vec<(u64, [u32; 3], Vec<Arg>)> {
    gpu.trace()
        .into_iter()
        .filter_map(|e| match e {
            Event::Launch(f, grid, block, _, _, args) => {
                assert_eq!(block, [256, 1, 1]);
                Some((f, grid, args))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn prefill_switch_parses_three_modes() {
    assert_eq!(Fp8bPrefill::parse(None).unwrap(), Fp8bPrefill::W8a8);
    assert_eq!(Fp8bPrefill::parse(Some("w8a8")).unwrap(), Fp8bPrefill::W8a8);
    assert_eq!(
        Fp8bPrefill::parse(Some("w8a16")).unwrap(),
        Fp8bPrefill::W8a16
    );
    assert_eq!(Fp8bPrefill::parse(Some("bf16")).unwrap(), Fp8bPrefill::Bf16);
    for v in ["1", "W8A8", "fp8"] {
        assert!(Fp8bPrefill::parse(Some(v)).is_err(), "{v}");
    }
    assert!(Fp8bPrefill::Bf16.needs_bf16() && !Fp8bPrefill::W8a8.needs_bf16());
}

#[test]
fn gemv_rows_chunk_at_32_on_the_layout_tier() {
    for (tiled, base) in [(false, 0u64), (true, 100)] {
        let gpu = recording::Gpu::new();
        let w = Fp8BlockWeight {
            data: DevicePtr(0x200),
            scales: DevicePtr(0x300),
            tiled,
        };
        // [n = 256, k = 128], 40 rows: a 32-row launch and an 8-row one.
        fp8b_gemv_rows(
            &gpu,
            &tiers(),
            &w,
            DevicePtr(0x10_0000),
            DevicePtr(0x20_0000),
            40,
            256,
            128,
            256,
            0,
        )
        .unwrap();
        let got = launches(&gpu);
        assert_eq!(got.len(), 2);
        for (i, (rows, tier)) in [(32u32, 32u64), (8, 8)].into_iter().enumerate() {
            let first = 32 * i as u64;
            assert_eq!(got[i].0, base + tier);
            assert_eq!(got[i].1, [16, 1, 1]);
            // (A, W, S, C, M, N, K, out_stride)
            assert_eq!(
                got[i].2,
                vec![
                    ptr(0x10_0000 + first * 128 * 2),
                    ptr(0x200),
                    ptr(0x300),
                    ptr(0x20_0000 + first * 256 * 2),
                    u32a(rows),
                    u32a(256),
                    u32a(128),
                    u32a(256),
                    // No touch (ATLAS_GLM_DECODE_GEMV_BATCH unset): plain PDL entry.
                    u32a(0),
                    u32a(0),
                ]
            );
        }
    }
}

#[test]
fn gemv_rows_refuse_off_grid_shapes_and_missing_tiers() {
    let gpu = recording::Gpu::new();
    let w = Fp8BlockWeight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    let x = DevicePtr(0x1000);
    // k off the 128 grid, n off 16, out_stride < n.
    for (n, k, os) in [(256, 96, 256), (250, 128, 256), (256, 128, 128)] {
        assert!(fp8b_gemv_rows(&gpu, &tiers(), &w, x, x, 4, n, k, os, 0).is_err());
    }
    assert!(fp8b_gemv_rows(&gpu, &MxTiers::default(), &w, x, x, 4, 256, 128, 256, 0).is_err());
    assert!(launches(&gpu).is_empty());
}

#[test]
fn multi_launch_packs_three_projections_into_one_grid() {
    let gpu = recording::Gpu::new();
    let parts = [
        MxPart {
            data: DevicePtr(0x1),
            scales: DevicePtr(0x2),
            out: DevicePtr(0x10_0000),
            n: 1536,
        },
        MxPart {
            data: DevicePtr(0x3),
            scales: DevicePtr(0x4),
            out: DevicePtr(0x20_0000),
            n: 512,
        },
    ];
    // 9 rows: the 16-row tier; grid = 1536/16 + 512/16 tiles; the third
    // projection slot is null with N = 0.
    mx_multi_rows(
        &gpu,
        &tiers(),
        false,
        DevicePtr(0x40_0000),
        &parts,
        9,
        4096,
        0,
    )
    .unwrap();
    let got = launches(&gpu);
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].0, got[0].1), (16, [96 + 32, 1, 1]));
    assert_eq!(
        got[0].2,
        vec![
            ptr(0x40_0000),
            ptr(0x1),
            ptr(0x2),
            ptr(0x10_0000),
            u32a(1536),
            ptr(0x3),
            ptr(0x4),
            ptr(0x20_0000),
            u32a(512),
            ptr(0),
            ptr(0),
            ptr(0),
            u32a(0),
            u32a(9),
            u32a(4096),
            u32a(0),
            u32a(0),
        ]
    );
}

#[test]
fn multi_launch_offsets_every_output_per_32_row_chunk() {
    let gpu = recording::Gpu::new();
    let part = |o: u64| MxPart {
        data: DevicePtr(0x1),
        scales: DevicePtr(0x2),
        out: DevicePtr(o),
        n: 64,
    };
    let parts = [part(0x10_0000), part(0x20_0000), part(0x30_0000)];
    mx_multi_rows(
        &gpu,
        &tiers(),
        true,
        DevicePtr(0x40_0000),
        &parts,
        33,
        128,
        0,
    )
    .unwrap();
    let got = launches(&gpu);
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].0, 108);
    assert_eq!(got[1].2[0], ptr(0x40_0000 + 32 * 128 * 2));
    for (i, base) in [0x10_0000u64, 0x20_0000, 0x30_0000].into_iter().enumerate() {
        assert_eq!(got[1].2[3 + 4 * i], ptr(base + 32 * 64 * 2));
    }
    assert!(mx_multi_rows(&gpu, &tiers(), false, DevicePtr(0), &[], 1, 128, 0).is_err());
}

#[test]
fn ld_rows_pass_the_slice_stride() {
    let gpu = recording::Gpu::new();
    let tiers = [KernelHandle(8), KernelHandle(16), KernelHandle(32)];
    // down's K-slice: [n = 4096, k = 1024] of a [4096, 2048] weight.
    fp8b_gemv_ld_rows(
        &gpu,
        &tiers,
        DevicePtr(0x200),
        DevicePtr(0x300),
        DevicePtr(0x1000),
        DevicePtr(0x2000),
        5,
        4096,
        1024,
        2048,
        0,
    )
    .unwrap();
    let got = launches(&gpu);
    assert_eq!((got[0].0, got[0].1), (8, [256, 1, 1]));
    assert_eq!(got[0].2[8], u32a(2048));
    let x = DevicePtr(0x1000);
    assert!(fp8b_gemv_ld_rows(&gpu, &tiers, x, x, x, x, 5, 4096, 1024, 1000, 0).is_err());
}

/// `src` without `//` and `/* */` comments (string literals in kernels never
/// hold either).
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("//") {
            rest = r.find('\n').map_or("", |i| &r[i..]);
        } else if let Some(r) = rest.strip_prefix("/*") {
            rest = r.find("*/").map_or("", |i| &r[i + 2..]);
        } else {
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

fn ident(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(s.len());
    (&s[..end], &s[end..])
}

/// The name a `__global__` declarator at `s` (just after `__global__`)
/// declares, stepping over `void` and `__launch_bounds__(...)`.
fn declarator(mut s: &str) -> Option<&str> {
    loop {
        s = s.trim_start_matches(|c: char| c.is_whitespace() || c == '\\');
        if let Some(r) = s.strip_prefix("void") {
            s = r;
        } else if let Some(r) = s.strip_prefix("__launch_bounds__") {
            let mut depth = 0;
            let end = r.char_indices().find_map(|(i, c)| {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                None
            })?;
            s = &r[end..];
        } else {
            let (name, _) = ident(s);
            return (!name.is_empty()).then_some(name);
        }
    }
}

/// Every kernel `src` declares: plain `__global__` functions and the
/// instantiations of a `#define M(name, ...)` whose body declares `name`.
fn declared_kernels(src: &str) -> std::collections::HashSet<String> {
    let src = strip_comments(src);
    let mut names = std::collections::HashSet::new();
    let mut macros: Vec<(String, usize)> = Vec::new();
    for (i, _) in src.match_indices("__global__") {
        // Inside a #define? Then it names a macro parameter, handled below.
        let line_start = src[..i].rfind('\n').map_or(0, |p| p + 1);
        let mut def = line_start;
        while def > 0 && src[..def - 1].ends_with('\\') {
            def = src[..def - 1].rfind('\n').map_or(0, |p| p + 1);
        }
        if let Some(d) = src[def..].trim_start().strip_prefix("#define")
            && let (mname, rest) = ident(d.trim_start())
            && let Some(params) = rest.strip_prefix('(')
            && let Some(close) = params.find(')')
            && let Some(name) = declarator(&src[i + "__global__".len()..])
        {
            if let Some(idx) = params[..close]
                .split(',')
                .map(str::trim)
                .position(|p| p == name)
            {
                macros.push((mname.to_string(), idx));
            }
            continue;
        }
        if let Some(name) = declarator(&src[i + "__global__".len()..]) {
            names.insert(name.to_string());
        }
    }
    for (mname, idx) in macros {
        for line in src.lines() {
            let line = line.trim_start();
            if let Some(args) = line
                .strip_prefix(mname.as_str())
                .and_then(|r| r.strip_prefix('('))
                && let Some(arg) = args.split([',', ')']).nth(idx)
            {
                names.insert(arg.trim().to_string());
            }
        }
    }
    names
}

#[test]
fn the_kernel_scanner_sees_plain_and_macro_declared_kernels() {
    let src = r#"
extern "C" __global__ void plain(int a) {}
extern "C" __global__ void __launch_bounds__(W * 32) bounded(int a) {}
extern "C" __global__ __launch_bounds__(B, 5) void after(int a) {}
// extern "C" __global__ void commented(int a) {}
#define TIER(name, NT)                                   \
    extern "C" __global__ void __launch_bounds__(256) name( \
        int a) { body<NT>(a); }
TIER(tier_a, 1)
TIER(tier_b, 2)
"#;
    let got = declared_kernels(src);
    for want in ["plain", "bounded", "after", "tier_a", "tier_b"] {
        assert!(got.contains(want), "{want} in {got:?}");
    }
    assert!(!got.contains("commented") && !got.contains("name"));
}

/// Every kernel the native GLM paths look up (`native_kernel_names`) is
/// declared in the glm-5.3-flash target's sources (its own `nvfp4/` file of
/// that module, else `common/`), under the module name the build gives that
/// file (its stem: none of these modules is renamed in a KERNEL.toml).
#[test]
fn every_native_kernel_lookup_is_declared_in_the_glm_target() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10");
    let tomls = [
        root.join("common/KERNEL.toml"),
        root.join("glm-5.3-flash/nvfp4/KERNEL.toml"),
    ];
    let mut missing = Vec::new();
    for (module, func) in native_kernel_names() {
        for toml in &tomls {
            let text = std::fs::read_to_string(toml).unwrap();
            assert!(
                !text
                    .lines()
                    .any(|l| l.trim_start().starts_with(&format!("{module} ="))
                        || l.contains(&format!("= \"{module}\""))),
                "{module} is renamed in {}: the lookup's module name is wrong",
                toml.display()
            );
        }
        let file = [
            root.join(format!("glm-5.3-flash/nvfp4/{module}.cu")),
            root.join(format!("common/{module}.cu")),
        ]
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("no source file builds module {module}"));
        let src = std::fs::read_to_string(&file).unwrap();
        if !declared_kernels(&src).contains(&func) {
            missing.push(format!("{module}::{func} ({})", file.display()));
        }
    }
    assert!(missing.is_empty(), "not declared: {missing:#?}");
}

#[test]
fn block_fp8_plan_leaves_padding_headroom_and_a_tail_slot() {
    // An arena for 4096 rows of k = 8192 (GLM: max_proj_k = q heads x 256).
    let scratch = Fp8ActScratch {
        act: DevicePtr(0x1000_0000),
        act_bytes: 4096 * 8192,
        scale: DevicePtr(0x2000_0000),
        scale_bytes: 4096 * (8192 / 128) * 4,
    };
    for (k, n) in [(4096, 1536), (8192, 4096), (1536, 8192), (2048, 4096)] {
        let plan = scratch.blk_plan(k, n).unwrap();
        assert!(plan.rows >= 16 && plan.rows.is_multiple_of(16), "{k} {n}");
        let slot = (plan.tail_out.0 - scratch.act.0) as usize;
        assert!(slot.is_multiple_of(256) && slot + 16 * n * 2 <= scratch.act_bytes);
        // A pass plus its 16-row padding stays below the tail slot and in the
        // scale scratch.
        assert!((plan.rows + 16) * k <= slot);
        assert!((plan.rows + 16) * (k / 128) * 4 <= scratch.scale_bytes);
    }
    let tiny = Fp8ActScratch {
        act_bytes: 20 * 4096,
        ..scratch
    };
    assert!(tiny.blk_plan(4096, 4096).is_err());
}
