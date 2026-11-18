// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn config() -> Config {
    Config {
        rank: 0,
        world: 2,
        vocab: 120832,
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
            capacity: 63,
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
    ] {
        assert!(c.validate().is_err());
    }
}
#[test]
fn valid_query_fills_capacity_exactly() {
    assert!(input().validate(config()).is_ok());
    for count in 1..=5 {
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
            count: 6,
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
