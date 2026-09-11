// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_runtime.h>

extern "C" __global__ void graph_set_conditional(
    cudaGraphConditionalHandle handle,
    const unsigned int* predicate
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cudaGraphSetConditional(handle, *predicate);
    }
}

extern "C" __global__ void graph_write_value(
    unsigned int* output,
    unsigned int value
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = value;
    }
}

extern "C" __global__ void graph_loop_emit(
    unsigned int* output,
    const unsigned int* iteration,
    unsigned int* continuation,
    unsigned int stop_after
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned int index = *iteration;
        output[index] = index + 1;
        if (stop_after != 0 && index + 1 >= stop_after) {
            *continuation = 0;
        }
    }
}

extern "C" __global__ void graph_while_update(
    cudaGraphConditionalHandle handle,
    unsigned int* continuation,
    unsigned int* iteration,
    unsigned int max_iterations,
    const unsigned int* cancellation,
    unsigned int* exit_reason
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned int next = *iteration + 1;
    unsigned int keep_going = *continuation != 0;
    unsigned int reason = 0;
    if (*cancellation != 0) {
        keep_going = 0;
        reason = 2;
    } else if (!keep_going) {
        reason = 1;
    } else if (next >= max_iterations) {
        keep_going = 0;
        reason = 3;
    }
    *iteration = next;
    *continuation = keep_going;
    *exit_reason = reason;
    cudaGraphSetConditional(handle, keep_going);
}
