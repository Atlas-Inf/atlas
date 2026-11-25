# SPDX-License-Identifier: AGPL-3.0-only
"""Compare Atlas's native projection with a CPU product of the original checkpoint blocks."""
import argparse
import ctypes
import torch
from safetensors import safe_open
from fp8_weights import block_fp8

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library')
    parser.add_argument('first_shard')
    args = parser.parse_args()
    project = ctypes.CDLL(args.library).project
    project.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
    generator = torch.Generator().manual_seed(76)
    with safe_open(args.first_shard, framework='pt') as shard:
        for name in ('model.layers.0.attention.q_proj', 'model.layers.2.mlp.experts.0.gate_proj',
                     'model.layers.5.attention.kv_a_proj_with_mqa'):
            w = shard.get_tensor(name + '.weight')
            s = shard.get_tensor(name + '.weight_scale_inv')
            n, k = w.shape
            for m in (1,9,17,65):
                # Set the maximum of each block to a power-of-two multiple
                # of 448. Both implementations then quantize identical ties.
                x = torch.randn(m,k,generator=generator).bfloat16()
                x[:,::128] = 7
                baseline = block_fp8(x,w,s)
                xg,wg,sg = x.cuda(),w.cuda(),s.float().cuda()
                ag = torch.empty_like(xg,dtype=w.dtype)
                ax = torch.empty((m,k//128),device='cuda',dtype=torch.float32)
                y = torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                torch.cuda.synchronize()
                code = project(*(v.data_ptr() for v in (xg,ag,ax,wg,sg,y)),m,n,k,int(sg.dtype==torch.bfloat16))
                if code:
                    raise RuntimeError(f'CUDA error {code}')
                actual = y.cpu()
                torch.testing.assert_close(actual,baseline,rtol=.008,atol=.002)
                print(name,m,n,k,'max_abs',float((actual.float()-baseline.float()).abs().max()),flush=True)
if __name__=='__main__':
    main()
