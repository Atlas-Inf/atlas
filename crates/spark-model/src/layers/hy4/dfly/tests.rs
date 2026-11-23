// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn config() -> Config {
    Config {
        rank: 0,
        world: 2,
        vocab: 120832,
        mask_token_id: 120009,
        eps: 1e-5,
        theta: 1e7,
    }
}
fn input() -> Input {
    Input {
        target_hidden: DevicePtr(1),
        bonus: 32,
        count: 5,
        caches: [TargetCache {
            latent: DevicePtr(2),
            indices: DevicePtr(3),
            len: 57,
            capacity: 66,
        }; 5],
    }
}
#[test]
fn validates_tp_and_explicit_numerics() {
    assert!(config().validate().is_ok());
    for c in [
        Config {
            rank: 2,
            ..config()
        },
        Config {
            world: 0,
            ..config()
        },
        Config {
            world: 3,
            ..config()
        },
        Config {
            vocab: 120833,
            ..config()
        },
        Config {
            eps: f32::NAN,
            ..config()
        },
        Config {
            theta: 0.,
            ..config()
        },
        Config {
            mask_token_id: 120832,
            ..config()
        },
    ] {
        assert!(c.validate().is_err());
    }
}

#[test]
fn query_rows_use_the_configured_mask_without_replacing_the_bonus() {
    for mask_token_id in [0, 17, 120009] {
        let c = Config {
            mask_token_id,
            ..config()
        };
        assert!(c.validate().is_ok());
        for count in 1..=8 {
            let i = Input { count, ..input() };
            let ids = i.query_ids(c);
            assert_eq!(ids[0], i.bonus);
            assert_eq!(&ids[1..], vec![mask_token_id; count]);
        }
    }
}
#[test]
fn valid_query_fills_capacity_exactly() {
    assert!(input().validate(config()).is_ok());
    for count in 1..=8 {
        let i = Input { count, ..input() };
        assert!(i.validate(config()).is_ok());
    }
}
#[test]
fn rejects_empty_oversized_invalid_and_missing_inputs() {
    for i in [
        Input {
            count: 0,
            ..input()
        },
        Input {
            count: 9,
            ..input()
        },
        Input {
            target_hidden: DevicePtr::NULL,
            ..input()
        },
        Input {
            bonus: 120832,
            ..input()
        },
    ] {
        assert!(i.validate(config()).is_err());
    }
    for cache in [
        TargetCache {
            len: 58,
            ..input().caches[0]
        },
        TargetCache {
            capacity: 62,
            ..input().caches[0]
        },
        TargetCache {
            latent: DevicePtr::NULL,
            ..input().caches[0]
        },
        TargetCache {
            indices: DevicePtr::NULL,
            ..input().caches[0]
        },
    ] {
        let mut i = input();
        i.caches[2] = cache;
        assert!(i.validate(config()).is_err());
    }
}
