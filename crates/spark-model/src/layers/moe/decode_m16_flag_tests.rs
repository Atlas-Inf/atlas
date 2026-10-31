// SPDX-License-Identifier: AGPL-3.0-only
//! The decode flags' parse and the NOSYNC expert-offset read, split from
//! `decode_m16_tests.rs` for the file-size cap.
use super::*;

/// ATLAS_GLM_MOE_STREAM_NOSYNC: with the exact M64 tile count at its NVFP4
/// default, a batch the stream twins take launches exactly what it launches
/// without the count (`verify_ffn`'s checks) and reads nothing back; without
/// the switch it still reads the expert offsets (the recording backend fails
/// that read), and batches the twins do not take read them either way.
#[test]
fn stream_batches_read_no_expert_offsets_with_nosync() {
    const SENTINEL: &str = "ATLAS_TEST_STREAM_NOSYNC";
    let Ok(nosync) = std::env::var(SENTINEL) else {
        let name = concat!(
            module_path!(),
            "::stream_batches_read_no_expert_offsets_with_nosync"
        );
        let name = name.split_once("::").unwrap().1;
        for nosync in ["0", "1"] {
            let mut cmd = ffn_child(name, SENTINEL, nosync);
            cmd.env("ATLAS_GLM_MOE_DECODE_M16", "1")
                .env("ATLAS_GLM_MOE_DOWN_ZSKIP", "1")
                .env("ATLAS_GLM_MOE_DECODE_STREAM", "1")
                .env("ATLAS_GLM_MOE_STREAM_NOSYNC", nosync)
                .env_remove("ATLAS_MOE_PREFILL_EXACT_TILES");
            assert_child_passed(nosync, cmd);
        }
        return;
    };
    let reads = |entry, rows| {
        std::panic::catch_unwind(|| {
            let gpu = Gpu::new();
            let trace = verify_ffn(&gpu, 0, false, true, entry, rows);
            trace
                .iter()
                .filter(|e| matches!(e, Event::Read(..)))
                .count()
        })
    };
    // Batches of 9 to 32 rows: the owner-batched verify of two to four streams.
    for (entry, rows) in [
        (Entry::Owner, 12),
        (Entry::Prefill, 12),
        (Entry::Prefill, 24),
    ] {
        match nosync.as_str() {
            "1" => assert_eq!(reads(entry, rows).ok(), Some(0), "{rows} rows"),
            _ => assert!(reads(entry, rows).is_err(), "{rows} rows read nothing"),
        }
    }
    // Wider than the twins: the M64 grid needs the count.
    assert!(reads(Entry::Prefill, 40).is_err());
}

#[test]
fn decode_flags_require_0_or_1() {
    const SENTINEL: &str = "ATLAS_TEST_DECODE_M16_TOGGLE";
    let Some(flag) = std::env::var_os(SENTINEL) else {
        let name = concat!(module_path!(), "::decode_flags_require_0_or_1");
        for flag in [
            "ATLAS_GLM_MOE_DECODE_M16",
            "ATLAS_GLM_MOE_DOWN_ZSKIP",
            "ATLAS_GLM_MOE_DECODE_K128W",
            "ATLAS_GLM_MOE_DECODE_STREAM",
            "ATLAS_GLM_MOE_DECODE_L2PF",
            "ATLAS_GLM_MOE_DECODE_PERSIST",
            "ATLAS_GLM_MOE_STREAM_NOSYNC",
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name.split_once("::").unwrap().1])
                .env(SENTINEL, flag)
                .env(flag, "true")
                .output()
                .unwrap();
            assert!(output.status.success(), "{flag}");
        }
        return;
    };
    let gpu = Gpu::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    let error = crate::layers::moe::decode_m16::DecodeM16::new(&gpu, &config)
        .err()
        .expect("a toggle other than 0 or 1 must fail the load");
    assert!(error.to_string().contains(flag.to_str().unwrap()));
}
