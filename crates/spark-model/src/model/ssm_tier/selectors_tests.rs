// SPDX-License-Identifier: AGPL-3.0-only

//! Env-guarded selector default-path tests (moved from the pre-split
//! ssm_tier.rs).

use super::*;

fn fp() -> ModelFingerprint {
    let cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    ModelFingerprint::derive_with_id(&cfg, 4, "").unwrap()
}

#[test]
fn decode_tier_defaults_to_host_ram_non_dropping() {
    // With ATLAS_SSM_DECODE_TIER unset the decode store is unbounded host-RAM
    // and never drops (the correctness floor). Guard on the var being unset.
    if std::env::var_os("ATLAS_SSM_DECODE_TIER").is_none() {
        let s = build_decode_tier_store(fp(), 4, /*min_slots*/ 8).unwrap();
        for k in 0..2000u64 {
            assert!(s.put(k, &[0; 4]).unwrap(), "non-dropping: nothing refused");
        }
        assert_eq!(s.len(), 2000);
    }
}

#[test]
fn disk_cap_defaults_to_unbounded() {
    // Default-OFF guard for ATLAS_SSM_TIER_DISK_GB: unset ⇒ 0 ⇒ the Marconi
    // arms construct exactly the pre-cap unbounded store. Guarded on the var
    // being unset (same idiom as the other selector tests).
    if std::env::var_os("ATLAS_SSM_TIER_DISK_GB").is_none()
        && std::env::var_os("ATLAS_SSM_RDMA_TIER").is_none()
    {
        assert_eq!(
            ssm_tier_disk_slots(4).unwrap(),
            0,
            "unset budget must resolve to the unbounded sentinel"
        );
        let (s, _) = build_tier_store(fp(), 4).unwrap();
        for k in 0..1000u64 {
            assert!(s.put(k, &[0; 4]).unwrap());
        }
        assert_eq!(s.len(), 1000, "nothing dropped without an explicit budget");
    }
}

#[test]
fn build_tier_store_defaults_to_host_ram_unbounded() {
    // With ATLAS_SSM_RDMA_TIER absent (the byte-identical default), the
    // selector yields the unbounded host-RAM store. Guarded on the var being
    // unset so a concurrent env-setting test can't flake this.
    if std::env::var_os("ATLAS_SSM_RDMA_TIER").is_none() {
        let (s, home) = build_tier_store(fp(), 4).unwrap();
        // A 4-byte blob is never an O_DIRECT record: whichever arm the
        // environment picks, the spills stay in this process's RAM.
        assert_eq!(home, SpillHome::HostRam);
        assert!(s.put(1, &[1, 2, 3, 4]).unwrap());
        let mut o = [0u8; 4];
        assert!(s.get(1, &mut o).unwrap());
        assert_eq!(o, [1, 2, 3, 4]);
        for k in 0..1000u64 {
            assert!(s.put(k, &[0; 4]).unwrap(), "unbounded: nothing dropped");
        }
        assert_eq!(s.len(), 1000);
    }
}

#[test]
fn a_spill_home_reserves_its_staging_blob_and_its_host_arena() {
    // What the KV sizing leaves out of the pool (with the NVMe prefix tier).
    assert_eq!(SpillHome::Peer.lazy_host_bytes(100), 100);
    assert_eq!(SpillHome::Disk { hot_slots: 0 }.lazy_host_bytes(100), 100);
    assert_eq!(SpillHome::Disk { hot_slots: 64 }.lazy_host_bytes(100), 6500);
    // The arena is host RAM only when the swap tier behind it is a file:
    // a host-RAM swap has no bound to reserve.
    assert_eq!(
        SpillHome::unified(SwapBacking::ODirect, 2),
        SpillHome::Disk { hot_slots: 2 }
    );
    assert_eq!(
        SpillHome::unified(SwapBacking::HostRam, 2),
        SpillHome::HostRam
    );
}
