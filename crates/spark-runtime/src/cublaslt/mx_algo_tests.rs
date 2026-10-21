// SPDX-License-Identifier: AGPL-3.0-only
// Ported from Enntity/atlas (branch `sparkqwen` @ 92dd543, AGPL-3.0-only), `crates/spark-runtime/src/cublaslt/mx_algo_tests.rs`.

use super::{Algo, bucket, key, prepare_rows, store};

#[test]
fn rows_map_to_power_of_two_buckets_within_64_to_16384() {
    assert_eq!(bucket(0), 64);
    assert_eq!(bucket(1), 64);
    assert_eq!(bucket(64), 64);
    assert_eq!(bucket(65), 128);
    assert_eq!(bucket(2048), 2048);
    assert_eq!(bucket(2049), 4096);
    assert_eq!(bucket(16384), 16384);
    assert_eq!(bucket(16046), 16384);
    assert_eq!(bucket(40000), 16384);
    assert_eq!(bucket(u32::MAX), 16384);
}

#[test]
fn only_a_swept_shape_on_its_swept_cublaslt_gets_a_key_per_bucket() {
    // Swept on 13.0.0 and 13.1.1: one key per bucket.
    for v in [130000, 130101] {
        assert_eq!(key(v, 8192, 2560, 1), (8192, 2560, 64));
        assert_eq!(key(v, 8192, 2560, 100), (8192, 2560, 128));
        assert_eq!(key(v, 2560, 3072, 16046), (2560, 3072, 16384));
    }
    // Another version, a shape not swept yet, one never swept: a
    // single key for every row count.
    for (v, n, k) in [
        (130200, 8192, 2560),
        (0, 6144, 2560),
        (130000, 256, 2560),
        (130101, 16384, 2560),
        (130000, 320, 2560),
    ] {
        for m in [1, 64, 65, 2048, 16384, 40000] {
            assert_eq!(key(v, n, k, m), (n, k, 0), "{v} {n}x{k} m={m}");
        }
    }
}

#[test]
fn prepare_tunes_every_bucket_or_only_the_largest_pass() {
    assert_eq!(prepare_rows(true, 64), vec![64]);
    assert_eq!(
        prepare_rows(true, 4096),
        vec![64, 128, 256, 512, 1024, 2048, 4096]
    );
    assert_eq!(prepare_rows(true, 16384).len(), 9);
    for top in [64, 4096, 16384] {
        assert_eq!(prepare_rows(false, top), vec![top]);
    }
}

#[test]
fn a_key_keeps_the_kernel_it_holds_while_that_serves() {
    let algo = |b: u8| -> Algo { [b; 64] };
    let (key, other) = ((8192, 2560, 64), (8192, 2560, 128));
    let mut cache = Vec::new();
    // First store wins the key.
    assert_eq!(store(&mut cache, key, algo(1), |_| true), algo(1));
    // A racing tune of the same key gets the cached kernel back, not
    // its own pick.
    assert_eq!(store(&mut cache, key, algo(2), |_| true), algo(1));
    assert_eq!(cache, vec![(key, algo(1))]);
    // A cached kernel that refuses the problem is replaced.
    assert_eq!(store(&mut cache, key, algo(3), |a| *a != algo(1)), algo(3));
    assert_eq!(cache, vec![(key, algo(3))]);
    // Another key is its own entry.
    assert_eq!(store(&mut cache, other, algo(4), |_| true), algo(4));
    assert_eq!(cache, vec![(key, algo(3)), (other, algo(4))]);
}
