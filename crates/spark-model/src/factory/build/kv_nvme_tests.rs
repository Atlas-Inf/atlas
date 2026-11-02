// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the NVMe spill tier's construction (`kv_nvme.rs`): env parsing,
//! the rank agreement word, attach on both I/O paths and the host-memory reserve.

use super::shard_tests::GLM;
use super::*;

fn switches(fast: Option<&str>, keep: Option<&str>) -> [(&'static str, Option<String>); 2] {
    [
        (FAST_VAR, fast.map(str::to_owned)),
        (KEEP_VAR, keep.map(str::to_owned)),
    ]
}

#[test]
fn off_unless_dir_is_set() {
    assert_eq!(config_from(None, None, switches(None, None)).unwrap(), None);
    assert_eq!(
        config_from(Some(" "), None, switches(None, None)).unwrap(),
        None
    );
    // The switches alone are inert, not an error (arms share a profile) —
    // and with the tier off they are not read at all, whatever they hold.
    for (fast, keep) in [
        (Some("1"), Some("1")),
        (Some("true"), None),
        (None, Some("x")),
    ] {
        assert_eq!(config_from(None, None, switches(fast, keep)).unwrap(), None);
    }
    // … and so is the budget: without the directory nothing of the tier is
    // read, so a rank starts exactly as it does with all four unset.
    for gb in ["40", "lots", "0"] {
        let off = config_from(None, Some(gb), switches(Some("1"), None));
        assert_eq!(off.unwrap(), None, "{gb}");
    }
}

#[test]
fn budget_is_required_and_strict() {
    assert!(config_from(Some("/nvme"), None, switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("lots"), switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("0"), switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("-1"), switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("inf"), switches(None, None)).is_err());
    let c = config_from(Some("/nvme/kv"), Some("1.5"), switches(None, None))
        .unwrap()
        .unwrap();
    assert_eq!(c.dir, PathBuf::from("/nvme/kv"));
    assert_eq!(c.budget_bytes, 3 << 29);
    assert!(!c.fast && !c.keep);
    let on = |fast, keep| config_from(Some("/nvme"), Some("1"), switches(fast, keep));
    let c = on(Some("1"), Some("0")).unwrap().unwrap();
    assert!(c.fast && !c.keep);
    let c = on(None, Some("1")).unwrap().unwrap();
    assert!(!c.fast && c.keep);
    assert!(on(Some("yes"), None).is_err(), "the switches are strict");
    assert!(on(None, Some("true")).is_err());
}

#[test]
fn slots_floor_the_budget() {
    assert_eq!(max_slots(106_496 * 10 + 5, 106_496).unwrap(), 10);
    assert!(max_slots(4095, 4096).is_err());
}

/// `(fast, keep)` for [`rank_fingerprint`].
const OFF: (bool, bool) = (false, false);
pub(super) const FAST: (bool, bool) = (true, false);
const KEEP: (bool, bool) = (false, true);
/// Snapshot-tier homes: on an O_DIRECT file, and on a peer.
pub(super) const DISK: Option<SpillHome> = Some(SpillHome::Disk { hot_slots: 2 });
const PEER: Option<SpillHome> = Some(SpillHome::Peer);

#[test]
fn fingerprint_separates_every_field() {
    // (A pair started with the fast path on one rank only must not start.)
    let mut all = vec![
        rank_fingerprint(10, 4096, None, OFF),
        rank_fingerprint(10, 4096, None, FAST),
        rank_fingerprint(10, 4096, None, KEEP),
        rank_fingerprint(10, 4096, None, (true, true)),
        rank_fingerprint(10, 4096, DISK, OFF),
        rank_fingerprint(10, 4096, PEER, OFF),
        rank_fingerprint(10, 4096, DISK, KEEP),
        rank_fingerprint(10, 4096, PEER, FAST),
        rank_fingerprint(11, 4096, None, OFF),
        rank_fingerprint(10, 8192, None, OFF),
        rank_fingerprint(0, 4096, None, OFF),
        rank_fingerprint(0, 4096, DISK, OFF),
    ];
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 12, "every field moves the word");
    // The hot arena's size is rank-local (it only moves the host reserve).
    assert_eq!(
        rank_fingerprint(10, 4096, DISK, OFF),
        rank_fingerprint(10, 4096, Some(SpillHome::Disk { hot_slots: 64 }), OFF)
    );
}

pub(super) fn tier(gb: &str, fast: bool) -> NvmeKvConfig {
    let on = switches(fast.then_some("1"), None);
    config_from(Some("/nvme"), Some(gb), on).unwrap().unwrap()
}

#[test]
fn the_rank_word_is_zero_only_with_every_spill_tier_off() {
    // Off: the word the ranks already gather stays the bare block count.
    assert_eq!(word_for(None, GLM, None, true).unwrap(), TIER_OFF);
    assert_eq!(word_for(None, GLM, None, false).unwrap(), TIER_OFF);
    // The snapshot tier alone is a word too: its ranks run the restore
    // agreement, so a rank without it must not start beside one with it.
    // Nothing is refused there — that start is integ/next's.
    let ssm_only = word_for(None, GLM, DISK, false).unwrap();
    assert!(ssm_only != TIER_OFF && ssm_only != FAILED_RANK);
    assert_ne!(ssm_only, word_for(None, GLM, PEER, true).unwrap());
    let ram = Some(SpillHome::HostRam);
    assert_ne!(word_for(None, GLM, ram, true).unwrap(), TIER_OFF);
    let on = word_for(Some(&tier("24", false)), GLM, DISK, true).unwrap();
    assert!(on != TIER_OFF && on != FAILED_RANK && on != ssm_only);
    for other in [
        word_for(Some(&tier("24", true)), GLM, DISK, true),
        word_for(Some(&tier("25", false)), GLM, DISK, true),
        word_for(Some(&tier("24", false)), GLM, None, true),
        word_for(Some(&tier("24", false)), GLM, PEER, true),
    ] {
        assert_ne!(on, other.unwrap());
    }
    // A budget below one record is this rank's error, sent as the sentinel.
    assert!(word_for(Some(&tier("0.00001", false)), GLM, DISK, true).is_err());
    // No field combination lands on either reserved word.
    for slots in 0..4096 {
        let home = [None, DISK, PEER, ram][slots as usize % 4];
        let fp = rank_fingerprint(slots, 106_496, home, FAST);
        assert!(fp != TIER_OFF && fp != FAILED_RANK);
    }
}

#[test]
fn the_kv_tier_refuses_a_host_ram_snapshot_tier_and_a_pair_without_lifeline() {
    let cfg = tier("24", true);
    // Spilled snapshots in host RAM are outside the reserve the pool is sized
    // around (legacy store, no usable swap directory, non-4 KiB blob).
    let e = word_for(Some(&cfg), GLM, Some(SpillHome::HostRam), true).unwrap_err();
    let text = format!("{e:#}");
    assert!(text.contains("ATLAS_SSM_TIER_SWAP_DIR"), "{text}");
    assert!(text.contains("host RAM"), "{text}");
    // A rank that fails in its local attach stops its peer through the
    // lifeline only.
    let e = word_for(Some(&cfg), GLM, DISK, false).unwrap_err();
    assert!(format!("{e:#}").contains("ATLAS_EP_PEER_LIFELINE"), "{e:#}");
    assert!(snapshots_off_host(Some(SpillHome::HostRam)).is_err());
    assert!(snapshots_off_host(DISK).is_ok() && snapshots_off_host(None).is_ok());
    // Either refusal reaches every rank as the failure sentinel.
    let peer = rank_fingerprint(10, 4096, DISK, FAST);
    let refused = word_for(Some(&cfg), GLM, Some(SpillHome::HostRam), true);
    assert!(verify_ranks(refused, &[FAILED_RANK, peer]).is_err());
    let e = verify_ranks(Ok(peer), &[FAILED_RANK, peer]).unwrap_err();
    assert!(format!("{e:#}").contains("rank 0"), "{e:#}");
}

#[test]
fn a_failed_rank_fails_every_rank_after_the_exchange() {
    let fp = rank_fingerprint(10, 4096, None, OFF);
    assert!(verify_ranks(Ok(fp), &[fp, fp]).is_ok());
    assert!(verify_ranks(Ok(TIER_OFF), &[TIER_OFF, TIER_OFF]).is_ok());
    let own = verify_ranks(Err(anyhow::anyhow!("bad env")), &[FAILED_RANK, fp]);
    assert!(format!("{:#}", own.unwrap_err()).contains("bad env"));
    let peer = verify_ranks(Ok(fp), &[fp, FAILED_RANK]).unwrap_err();
    assert!(format!("{peer:#}").contains("rank 1"));
    let other = rank_fingerprint(11, 4096, None, OFF);
    assert!(verify_ranks(Ok(fp), &[fp, other]).is_err());
    // The tier on one rank only: both ranks refuse, the one without it too.
    assert!(verify_ranks(Ok(fp), &[fp, TIER_OFF]).is_err());
    assert!(verify_ranks(Ok(TIER_OFF), &[fp, TIER_OFF]).is_err());
}

/// Rank 0 of a two-rank world on the host: records the word this rank sent
/// and lands `[ours, peer]` in the receive buffer.
struct Pair<'a> {
    gpu: &'a spark_runtime::gpu::mock::MockGpuBackend,
    peer: u64,
    sent: std::sync::Mutex<Vec<u64>>,
}

impl spark_comm::CommBackend for Pair<'_> {
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        use spark_runtime::gpu::DevicePtr;
        assert_eq!(bytes, 8);
        let mut ours = [0u8; 8];
        self.gpu.copy_d2h(DevicePtr(send), &mut ours)?;
        self.sent.lock().unwrap().push(u64::from_le_bytes(ours));
        let all = [ours, self.peer.to_le_bytes()].concat();
        self.gpu.copy_h2d(&all, DevicePtr(recv))
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        2
    }
}

/// `agree_kv_blocks` on rank 0 for 1000 local blocks: the agreed count (or
/// the error) and the one word this rank put on the wire.
fn agree(ours: Result<u32>, peer: u64) -> (Result<usize>, u64) {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let pair = Pair {
        gpu: &gpu,
        peer,
        sent: Default::default(),
    };
    let agreed = super::super::glm::agree_kv_blocks(Some(&pair), &gpu, 1000, None, ours);
    let sent = pair.sent.into_inner().unwrap();
    assert_eq!(sent.len(), 1, "exactly one collective");
    (agreed, sent[0])
}

#[test]
fn the_tier_word_rides_the_kv_block_agreement() {
    let on = |blocks: u64, word: u32| blocks | u64::from(word) << 32;
    // Tier off everywhere: the wire carries the bare block count, as it did
    // before the tier existed, and the ranks take the minimum.
    let (agreed, sent) = agree(Ok(TIER_OFF), 900);
    assert_eq!((agreed.unwrap(), sent), (900, 1000));
    let fp = rank_fingerprint(10, 4096, DISK, FAST);
    let (agreed, sent) = agree(Ok(fp), on(1200, fp));
    assert_eq!((agreed.unwrap(), sent), (1000, on(1000, fp)));
    // The tier on one rank only stops BOTH ranks, in the same collective.
    let (agreed, sent) = agree(Ok(TIER_OFF), on(900, fp));
    assert_eq!(sent, 1000);
    assert!(format!("{:#}", agreed.unwrap_err()).contains("differs across ranks"));
    assert!(agree(Ok(fp), 900).0.is_err());
    assert!(
        agree(Ok(fp), on(900, rank_fingerprint(10, 4096, DISK, OFF)))
            .0
            .is_err()
    );
    // A rank whose environment does not parse still takes part, says so on
    // the wire, and reports its own error; its peer names the rank.
    let (agreed, sent) = agree(Err(anyhow::anyhow!("bad env")), on(900, fp));
    assert_eq!(sent, on(1000, FAILED_RANK));
    assert!(format!("{:#}", agreed.unwrap_err()).contains("bad env"));
    let (agreed, _) = agree(Ok(fp), on(900, FAILED_RANK));
    assert!(format!("{:#}", agreed.unwrap_err()).contains("rank 1"));
}

/// [`setup_local`] as `attach` calls it, on rank 0.
fn setup(
    cfg: Option<NvmeKvConfig>,
    kv: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &spark_runtime::gpu::mock::MockGpuBackend,
) -> Result<u32> {
    let geometry = kv.nvme_geometry();
    setup_local(cfg, 0, geometry, kv, prefix_cache, gpu)
}

fn glm_kv(gpu: &spark_runtime::gpu::mock::MockGpuBackend) -> PagedKvCache {
    use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, SparseIndexCacheConfig};
    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 2,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut kv = PagedKvCache::new_with_v_alias(cfg, 4, gpu, true).unwrap();
    kv.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    kv
}

/// A scratch directory under the cargo target directory this test binary was
/// built into — `<target>/<profile>/deps/<binary>`, so `CARGO_TARGET_DIR` and
/// `build.target-dir` are honoured (a container's `/tmp` is an overlay or
/// tmpfs, which the tier refuses by design).
fn scratch_dir(tag: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let d = exe
        .ancestors()
        .nth(3)
        .expect("test binary under <target>/<profile>/deps")
        .join("atlas-kv-nvme-tests")
        .join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// `None` when even the build tree sits on a filesystem the tier refuses —
/// nothing to test there. Never silently: the skip is printed, and a run
/// that must cover the record file sets `ATLAS_TIER_REQUIRE_O_DIRECT` (as
/// `atlas-tier`'s own tests do), which turns the skip into a failure.
pub(super) fn disk_scratch_dir(tag: &str) -> Option<PathBuf> {
    let d = scratch_dir(tag);
    std::fs::create_dir_all(&d).unwrap();
    match atlas_tier::unsuitable_swap_fs(&d) {
        None => Some(d),
        Some(kind) => {
            assert!(
                std::env::var_os("ATLAS_TIER_REQUIRE_O_DIRECT").is_none(),
                "ATLAS_TIER_REQUIRE_O_DIRECT set but {} is on {kind}",
                d.display()
            );
            eprintln!(
                "SKIPPED {tag}: the build tree ({}) is on {kind}",
                d.display()
            );
            None
        }
    }
}

#[test]
fn attach_enables_tier_and_leaves_no_file_behind() {
    use spark_runtime::prefix_cache::NvmePrefixTier;
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = glm_kv(&gpu);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    let Some(dir) = disk_scratch_dir("on") else {
        return;
    };
    // A record file left behind by a process that died before unlinking.
    std::fs::write(dir.join("atlas-kv-prefix.4242.r0.swap"), b"stale").unwrap();
    let cfg = NvmeKvConfig {
        dir: dir.clone(),
        budget_bytes: 1 << 30,
        fast: false,
        keep: false,
    };
    setup(Some(cfg), &mut kv, &tree, &gpu).unwrap();
    assert!(kv.nvme_attached());
    assert!(!kv.nvme_io_stats().fast);
    assert!(tree.is_enabled());
    assert_eq!(
        tree.nvme_stats().max_slots as usize,
        (1usize << 30) / kv.nvme_record_bytes()
    );
    #[cfg(unix)]
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "record file unlinked"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn off_is_a_no_op_and_prefix_caching_is_required() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = glm_kv(&gpu);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    assert_eq!(setup(None, &mut kv, &tree, &gpu).unwrap(), 0);
    assert!(!kv.nvme_attached());
    let none = spark_runtime::prefix_cache::NoPrefixCaching;
    let cfg = NvmeKvConfig {
        dir: scratch_dir("nopc"),
        budget_bytes: 1 << 30,
        fast: false,
        keep: false,
    };
    assert!(setup(Some(cfg), &mut kv, &none, &gpu).is_err());
}

#[test]
fn fast_attach_spills_and_restores_through_the_record_file() {
    use spark_runtime::prefix_cache::{DiskRef, SpillOrder};
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = glm_kv(&gpu);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    let Some(dir) = disk_scratch_dir("fast") else {
        return;
    };
    let cfg = NvmeKvConfig {
        dir: dir.clone(),
        budget_bytes: 64 << 20,
        fast: true,
        keep: true,
    };
    setup(Some(cfg), &mut kv, &tree, &gpu).unwrap();
    assert!(kv.nvme_io_stats().fast);
    #[cfg(unix)]
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "unlinked");
    // Blocks 2, 1 leave for slots 0, 1 (a leaf-first chain) and come back.
    let pattern: Vec<u8> = (0..kv.k_block_stride_bytes_for_layer(0))
        .map(|i| (i % 253) as u8)
        .collect();
    gpu.copy_h2d(&pattern, kv.k_cache_ptr(0, 1)).unwrap();
    let order = |block, slot| SpillOrder {
        block,
        slot,
        tag: 7 + slot as u64,
    };
    assert!(
        kv.nvme_write(&[order(2, 0), order(1, 1)], &gpu, 0)
            .is_empty()
    );
    let disk = [DiskRef { slot: 1, tag: 8 }, DiskRef { slot: 0, tag: 7 }];
    let mut blocks = [3, 0];
    assert_eq!(kv.nvme_read(&disk, &mut blocks, &gpu, 0), (2, false));
    assert_eq!(blocks, [0, 3]);
    let mut back = vec![0u8; pattern.len()];
    gpu.copy_d2h(kv.k_cache_ptr(0, 0), &mut back).unwrap();
    assert_eq!(back, pattern, "block 1's bytes restored into block 0");
    assert!(kv.nvme_take_failed().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unusable_directory_is_refused_by_name() {
    let Some(dir) = disk_scratch_dir("usable") else {
        return;
    };
    usable_disk_dir(DIR_VAR, &dir).unwrap();
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "the probe file is removed"
    );
    // A path that cannot be a directory (its parent is a file).
    let file = dir.join("plain-file");
    std::fs::write(&file, b"x").unwrap();
    let e = usable_disk_dir("ATLAS_SSM_TIER_SWAP_DIR", &file.join("sub")).unwrap_err();
    assert!(
        format!("{e:#}").contains("ATLAS_SSM_TIER_SWAP_DIR"),
        "{e:#}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let ro = dir.join("read-only");
        std::fs::create_dir_all(&ro).unwrap();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o500)).unwrap();
        // (root ignores directory modes — the sandbox runs as root.)
        if std::fs::write(ro.join("w"), b"x").is_err() {
            let e = usable_disk_dir(DIR_VAR, &ro).unwrap_err();
            assert!(format!("{e:#}").contains("not usable"), "{e:#}");
        }
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
    // The snapshot tier's directory only counts when that tier would use it.
    assert_eq!(ssm_swap_dir(None), None);
    assert_eq!(ssm_swap_dir(Some(String::new())), None);
}
