"""Stage-by-stage compare, reading the op dumps as FP32.

`dump_bf16` writes `n_elements * 2` RAW bytes from a buffer this model runs
in FP32 (f32 residual stream), so each record is the FIRST HALF (1536 f32) of
the 3072-wide hidden row. Compare against the reference's first half.
"""
import json

import numpy as np
import requests

D = '/home/ms/.claude/jobs/5a7bd33d/tmp/opdump'
ref = np.load('/home/ms/.claude/jobs/5a7bd33d/tmp/stage_ref.npz')
exp = json.load(open('/home/ms/.claude/jobs/5a7bd33d/tmp/slice_expected.json'))

r = requests.post('http://127.0.0.1:8895/v1/completions',
                  json={'model': 'longcat-slice', 'prompt': exp['tokens'],
                        'max_tokens': 1, 'temperature': 0}, timeout=600).json()
print('served text:', repr(r['choices'][0]['text']))

pairs = [('sub0_input_norm_in', 'atlas_op_L0_input_norm_in.bin'),
         ('sub0_input_norm_out', 'atlas_op_L0_input_norm_out.bin'),
         ('sub0_post_attn_norm_out', 'atlas_op_L0_post_attn_norm_out.bin'),
         ('sub0_moe_out', 'atlas_op_L0_moe_out.bin'),
         ('sub1_input_norm_in', 'atlas_op_L1_input_norm_in.bin'),
         ('sub1_input_norm_out', 'atlas_op_L1_input_norm_out.bin'),
         ('sub1_post_attn_norm_out', 'atlas_op_L1_post_attn_norm_out.bin'),
         ('sub1_out', 'atlas_op_L1_moe_out.bin')]

print(f'{"stage":26s} {"|ref|":>9s} {"|atlas|":>9s} {"cos":>8s} {"relerr":>8s}')
for rk, fn in pairs:
    raw = np.fromfile(f'{D}/{fn}', dtype=np.float32)
    n = 1536                      # one record = 6144 bytes = 1536 f32
    a = raw[:n]                   # first record, first half of the row
    rv = ref[rk][:n]
    cos = float(rv @ a / (np.linalg.norm(rv) * np.linalg.norm(a) + 1e-9))
    rel = float(np.linalg.norm(a - rv) / (np.linalg.norm(rv) + 1e-9))
    print(f'{rk:26s} {np.linalg.norm(rv):9.3f} {np.linalg.norm(a):9.3f} '
          f'{cos:8.4f} {rel:8.4f}')
