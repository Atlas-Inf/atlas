// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_single_gpu_noop() {
    let comm = SingleGpuBackend;
    assert_eq!(comm.rank(), 0);
    assert_eq!(comm.world_size(), 1);
    comm.all_reduce(0x1000, 1024).unwrap();
    comm.all_gather(0x1000, 0x2000, 512).unwrap();
    comm.reduce_scatter(0x1000, 0x2000, 512).unwrap();
    comm.broadcast(0x1000, 256, 0).unwrap();
    comm.barrier().unwrap();
    comm.send_to(0x1000, 256, 0, 0).unwrap();
    comm.recv_from(0x2000, 256, 0, 0).unwrap();
    comm.group_start().unwrap();
    comm.group_end().unwrap();
    assert!(comm.is_healthy());
    comm.attempt_reconnect().unwrap();
}

#[test]
fn capturable_collectives_default_to_unavailable() {
    // The contract callers (graph capture policies) rely on: a backend
    // without them enqueues nothing and reports no capacity.
    let comm = SingleGpuBackend;
    assert!(!comm.all_reduce_capturable(0x1000, 8192, 0).unwrap());
    assert!(
        !comm
            .peer_exchange_capturable(0x1000, 0x2000, 8192, 0)
            .unwrap()
    );
    assert_eq!(comm.capturable_all_reduce_max_bytes(), 0);
    comm.set_oneshot_kernel(0x1234);
    assert_eq!(comm.capturable_all_reduce_max_bytes(), 0);
}

#[test]
fn command_words_default_to_unavailable() {
    // The contract the model's command path relies on: a backend without
    // a host command channel reports no capacity and moves nothing, so
    // every word stays on the broadcast path.
    let comm = SingleGpuBackend;
    assert_eq!(comm.command_words_max(), 0);
    assert!(comm.send_command_words(&[1]).is_err());
    assert!(comm.recv_command_words(&mut [0]).is_err());
}

#[test]
fn test_single_gpu_symmetric_alloc_unsupported() {
    // SingleGpuBackend doesn't override symmetric_alloc/free — it must
    // fall back to the trait default which returns an error. This is
    // the contract callers depend on for fallback paths.
    let comm = SingleGpuBackend;
    assert!(comm.symmetric_alloc(1024).is_err());
    assert!(comm.symmetric_free(0x1000).is_err());
}
