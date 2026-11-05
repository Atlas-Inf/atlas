// SPDX-License-Identifier: AGPL-3.0-only
//
// Dense SM120 block-scaled NVFP4 GEMM on operands quantized outside the GEMM,
// for checkpoints calibrated for W4A4 (ModelOpt NVFP4 with a static per-tensor
// activation `input_scale`): the format SGLang's `modelopt_fp4` runs.
//
//   activations: quantized at the checkpoint's global scale g_a (`input_scale`),
//                per-16 ue4m3 block scales written straight into the SFA atom
//                (`atlas_cutlass_nvfp4_quant_act_scaled`), or fused with the
//                clamped SwiGLU for the down projection's input
//                (`atlas_cutlass_nvfp4_quant_swiglu_clamp`);
//   weights:     the checkpoint's native [N, K/2] packed E2M1 (already CUTLASS's
//                K-contiguous ColumnMajor B) and an SFB packed once at load
//                (`atlas_cutlass_pack_weight_sfb`, N-major source);
//   epilogue:    alpha = g_a * g_w (weight_scale_2).
//
// Unlike `atlas_cutlass_nvfp4_gemm_bf16_act_weight_t`, nothing is packed per
// call and the activation's block scales carry the global scale, so small
// activation blocks do not underflow ue4m3.
//
// Block-scale encoding, as flashinfer `fp4_quantize` (the reference engine):
//   sf_b = e4m3_rn(amax_b / 6 / g_a);  q = e2m1_rn(x / (sf_b * g_a)).

#include "cutlass_nvfp4_types.cuh"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

__device__ __forceinline__ float rcp_approx_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// E2M1, round-to-nearest-even and saturating, as `cvt.rn.satfinite.e2m1x2`
// (which ptxas rejects at the family target this library builds for). Ties go
// to the even code: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0,
// 2.5 -> 2.0, 3.5 -> 4.0, 5 -> 4.0.
__device__ __forceinline__ unsigned char e2m1_rn(float x) {
  const unsigned char sign = signbit(x) ? 8u : 0u;
  const float a = fabsf(x);
  unsigned char mag;
  if (a <= 0.25f) mag = 0;
  else if (a < 0.75f) mag = 1;
  else if (a <= 1.25f) mag = 2;
  else if (a < 1.75f) mag = 3;
  else if (a <= 2.5f) mag = 4;
  else if (a < 3.5f) mag = 5;
  else if (a <= 5.0f) mag = 6;
  else mag = 7;
  return sign | mag;
}

// Two floats -> packed E2M1 pair (low nibble = `lo`).
__device__ __forceinline__ unsigned char e2m1x2_rn(float lo, float hi) {
  return static_cast<unsigned char>(e2m1_rn(lo) | (e2m1_rn(hi) << 4));
}

// Quantize one row's 16-value group `v` into `packed` / the SFA atom, with
// the reference engine's arithmetic (flashinfer `cvt_warp_fp16_to_fp4`, the
// default fast-math branch, SFScaleVal = 1 / input_scale): RNE E4M3 block
// scale, approximate reciprocals, RNE E2M1.
template <class Layout>
__device__ __forceinline__ void quant_group16(
    const float (&v)[16], unsigned char* __restrict__ packed, unsigned char* __restrict__ sf,
    int row, int k, int base, float inv_g, Layout layout_sfa) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const __nv_fp8_e4m3 s(inv_g * (amax * rcp_approx_ftz(6.0f)));
  sf[layout_sfa(row, base, 0)] = s.__x;  // positive: E4M3 bits == UE4M3
  const float q = amax != 0.0f ? rcp_approx_ftz(static_cast<float>(s) * rcp_approx_ftz(inv_g)) : 0.0f;
  uint2 out;
  unsigned char* o = reinterpret_cast<unsigned char*>(&out);
#pragma unroll
  for (int i = 0; i < 16; i += 2) o[i / 2] = e2m1x2_rn(v[i] * q, v[i + 1] * q);
  *reinterpret_cast<uint2*>(packed + (size_t)row * (k / 2) + base / 2) = out;
}

__device__ __forceinline__ void load16(const __nv_bfloat16* __restrict__ p, float (&v)[16]) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
  const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f0 = __bfloat1622float2(h0[i]);
    const float2 f1 = __bfloat1622float2(h1[i]);
    v[2 * i] = f0.x;
    v[2 * i + 1] = f0.y;
    v[8 + 2 * i] = f1.x;
    v[8 + 2 * i + 1] = f1.y;
  }
}

// One thread per 16-value group. Grid (m, ceil(k/16 / 256)), block 256.
template <class Layout>
__global__ void atlas_prequant_quant_act_scaled(
    const __nv_bfloat16* __restrict__ act, unsigned char* __restrict__ packed,
    unsigned char* __restrict__ sf, int m, int k, float inv_g, Layout layout_sfa) {
  const int row = blockIdx.x;
  const int group = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= m || group >= k / 16) return;
  float v[16];
  load16(act + (size_t)row * k + group * 16, v);
  quant_group16(v, packed, sf, row, k, group * 16, inv_g, layout_sfa);
}

// silu(min(gate, L)) * clamp(up, -L, L), rounded to BF16 as the unfused
// activation kernel stores it, then quantized as `atlas_prequant_quant_act_scaled`.
template <class Layout>
__global__ void atlas_prequant_quant_swiglu_clamp(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed, unsigned char* __restrict__ sf, int m, int k,
    float inv_g, float limit, Layout layout_sfa) {
  const int row = blockIdx.x;
  const int group = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= m || group >= k / 16) return;
  float g[16], u[16], v[16];
  load16(gate + (size_t)row * k + group * 16, g);
  load16(up + (size_t)row * k + group * 16, u);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float gc = fminf(g[i], limit);
    const float uc = fminf(fmaxf(u[i], -limit), limit);
    v[i] = __bfloat162float(__float2bfloat16(gc / (1.0f + __expf(-gc)) * uc));
  }
  quant_group16(v, packed, sf, row, k, group * 16, inv_g, layout_sfa);
}

static auto sfa_layout(int m, int k) {
  return CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, 128, k, 1));
}

static int launch_status() {
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
}

#endif

extern "C" int atlas_cutlass_nvfp4_quant_act_scaled(
    const void* act, void* packed, void* sf, int m, int k, float inv_global_scale,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (m <= 0 || k <= 0 || (k % 16) != 0) return -1;
  const dim3 block(256), grid(m, (k / 16 + 255) / 256);
  atlas_prequant_quant_act_scaled<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(act), static_cast<unsigned char*>(packed),
      static_cast<unsigned char*>(sf), m, k, inv_global_scale, sfa_layout(m, k));
  return launch_status();
#else
  (void)act; (void)packed; (void)sf; (void)m; (void)k; (void)inv_global_scale; (void)stream;
  return -120;
#endif
}

extern "C" int atlas_cutlass_nvfp4_quant_swiglu_clamp(
    const void* gate, const void* up, void* packed, void* sf, int m, int k,
    float inv_global_scale, float limit, cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (m <= 0 || k <= 0 || (k % 16) != 0) return -1;
  const dim3 block(256), grid(m, (k / 16 + 255) / 256);
  atlas_prequant_quant_swiglu_clamp<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up),
      static_cast<unsigned char*>(packed), static_cast<unsigned char*>(sf), m, k,
      inv_global_scale, limit, sfa_layout(m, k));
  return launch_status();
#else
  (void)gate; (void)up; (void)packed; (void)sf; (void)m; (void)k; (void)inv_global_scale;
  (void)limit; (void)stream;
  return -120;
#endif
}

// C[m, n] (BF16, row-major) = alpha * A[m, k] x W[n, k]^T on pre-quantized
// operands (see the header comment). `workspace` is CUTLASS's own scratch only.
// `swizzle` > 0 sets the persistent tile scheduler's max swizzle (tiles of a
// wave grouped so they share operand panels in L2; long-K shapes are
// otherwise DRAM-bound), 0 keeps CUTLASS's default.
extern "C" int atlas_cutlass_nvfp4_gemm_prequant(
    const void* a_packed, const void* a_sf, const void* w_packed, const void* w_sf,
    float alpha, void* out, int m, int n, int k, int swizzle, void* workspace,
    size_t workspace_size, cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (m <= 0 || n <= 0 || k <= 0 || (k % 16) != 0) return -1;
  const StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  const StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  const StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  const StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  const auto layout_sfa = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  const auto layout_sfb = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
          static_cast<ElementA::DataType const*>(a_packed),
          stride_a,
          static_cast<ElementB::DataType const*>(w_packed),
          stride_b,
          static_cast<ElementA::ScaleFactorType const*>(a_sf),
          layout_sfa,
          static_cast<ElementB::ScaleFactorType const*>(w_sf),
          layout_sfb,
      },
      {
          {alpha, 0.0f},
          static_cast<ElementC const*>(out),
          stride_c,
          static_cast<ElementD*>(out),
          stride_d,
      }};
  if (swizzle > 0) args.scheduler.max_swizzle_size = swizzle;
  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_size) return -2;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  return atlas_launch_status(static_cast<int>(gemm(stream)));
#else
  (void)a_packed; (void)a_sf; (void)w_packed; (void)w_sf; (void)alpha; (void)out; (void)m;
  (void)n; (void)k; (void)swizzle; (void)workspace; (void)workspace_size; (void)stream;
  return -120;
#endif
}
