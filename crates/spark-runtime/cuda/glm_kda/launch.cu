// SPDX-License-Identifier: AGPL-3.0-only AND MIT
// Atlas-owned native launch and layout contract. Device math is derived from
// MoonshotAI/FlashKDA @1ce47ea3 (MIT), see vendor/LICENSE and provenance.json.
// TMA construction follows the licensed kernel layouts, without the upstream
// launch_fwd API, framework dispatch, variable-sequence or bridge interface.
#include "params.h"
#include <type_traits>
#include <utility>
#include "vendor/fwd_kernel1.cuh"
#include "vendor/fwd_kernel2.cuh"

namespace atlas_kda_detail {
constexpr int H = 32, D = 128, CHUNK = 16;
constexpr int K1_THREADS = 256, K2_THREADS = 192;
constexpr int INPUT_STAGES = 3, OUTPUT_STAGES = 2;
using BF16 = cutlass::bfloat16_t;
using L1 = K1Layouts<D,CHUNK>;
using L2 = K2Layouts<D,CHUNK>;

__global__ void stage_beta(const BF16* input, BF16* output, int tokens) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < unsigned(H * tokens)) {
    const unsigned int head = i / tokens, row = i % tokens;
    output[i] = input[row * H + head];
  }
}

template<class T> T* ptr(std::uint64_t address) {
  return reinterpret_cast<T*>(address);
}
// Unevaluated descriptor types let startup resolve the exact entry points
// without constructing tensor maps from fake device pointers. Launch-side
// static assertions below keep the preload and execution specializations equal.
using QkvLayout = decltype(make_layout(make_shape(H,int{},D),make_stride(D,3*H*D,1)));
using PlaneLayout = decltype(make_layout(make_shape(H,int{},D),make_stride(D,H*D,1)));
using BetaLayout = decltype(make_layout(make_shape(int{})));
using StateLayout = decltype(make_layout(make_shape(H,D,D),LayoutRight{}));
using BiasLayout = decltype(make_layout(make_shape(H,D),LayoutRight{}));
using WsPlaneLayout = decltype(make_layout(make_shape(int{},CHUNK,D),LayoutRight{}));
using WsGateLayout = decltype(make_layout(make_shape(int{},D),LayoutRight{}));
using WsMatrixLayout = decltype(make_layout(make_shape(int{},CHUNK,CHUNK),LayoutRight{}));
template<class Op, class T, class Layout, class Smem>
using Tma = decltype(make_tma_copy(Op{},
    make_tensor(make_gmem_ptr(static_cast<T*>(nullptr)), std::declval<Layout>()), Smem{}));
using load_q_type = Tma<SM90_TMA_LOAD, const BF16, QkvLayout, L1::TMAQKLayout>;
using load_k_type = Tma<SM90_TMA_LOAD, const BF16, QkvLayout, L1::TMAQKLayout>;
using load_g_type = Tma<SM90_TMA_LOAD, const BF16, PlaneLayout, L1::TMAQKLayout>;
using load_b_type = Tma<SM90_TMA_LOAD, BF16, BetaLayout, L1::TMABetaSmemLayout>;
using load_d_type = Tma<SM90_TMA_LOAD, const float, BiasLayout, L1::TMAGTotalSmemLayout>;
using store_kd_type = Tma<SM90_TMA_STORE, BF16, WsPlaneLayout, L1::TMAVOLayout>;
using store_qd_type = Tma<SM90_TMA_STORE, BF16, WsPlaneLayout, L1::TMAVOLayout>;
using store_kr_type = Tma<SM90_TMA_STORE, BF16, WsPlaneLayout, L1::TMAVOLayout>;
using store_gt_type = Tma<SM90_TMA_STORE, float, WsGateLayout, L1::TMAGTotalSmemLayout>;
using store_inv_type = Tma<SM90_TMA_STORE, BF16, WsMatrixLayout, L1::TMALMLayout>;
using store_mqk_type = Tma<SM90_TMA_STORE, BF16, WsMatrixLayout, L1::TMALMLayout>;
using load_v_type = Tma<SM90_TMA_LOAD, const BF16, QkvLayout, L1::TMAVOLayout>;
using load_kd_type = Tma<SM90_TMA_LOAD, BF16, WsPlaneLayout, L1::TMAVOLayout>;
using load_qd_type = Tma<SM90_TMA_LOAD, BF16, WsPlaneLayout, L1::TMAVOLayout>;
using load_kr_type = Tma<SM90_TMA_LOAD, BF16, WsPlaneLayout, L1::TMAVOLayout>;
using load_gt_type = Tma<SM90_TMA_LOAD, float, WsGateLayout, L1::TMAGTotalSmemLayout>;
using load_inv_type = Tma<SM90_TMA_LOAD, BF16, WsMatrixLayout, L1::TMALMLayout>;
using load_mqk_type = Tma<SM90_TMA_LOAD, BF16, WsMatrixLayout, L1::TMALMLayout>;
using load_s_type = Tma<SM90_TMA_LOAD, float, StateLayout, L2::TMAFP32StateSmemLayout>;
using store_s_type = Tma<SM90_TMA_STORE, float, StateLayout, L2::TMAFP32StateSmemLayout>;
using store_o_type = Tma<SM90_TMA_STORE, BF16, PlaneLayout, L1::TMAVOLayout>;
inline constexpr auto prepare = _flash_kda_fwd_prepare<load_q_type,load_k_type,
      load_b_type,load_g_type,load_d_type,store_kd_type,
      store_qd_type,store_kr_type,store_gt_type,store_inv_type,
      store_mqk_type,CHUNK,D,K1_THREADS,false>;
inline constexpr auto recur = _flash_kda_fwd_recurrence<load_v_type,load_b_type,
      load_kd_type,load_qd_type,load_kr_type,load_gt_type,
      load_inv_type,load_mqk_type,load_s_type,store_s_type,
      store_o_type,CHUNK,D,INPUT_STAGES,OUTPUT_STAGES,K2_THREADS,true,true,true,false>;
} // namespace atlas_kda_detail

// Resolve all code residency and shared-memory requirements before the engine
// samples free memory for KV sizing. No operand pointers, launches or scratch.
extern "C" int atlas_glm_kda_preload() {
  using namespace atlas_kda_detail;
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, stage_beta);
  if (status != cudaSuccess) return int(status);
  status = cudaFuncGetAttributes(&attributes, prepare);
  if (status != cudaSuccess) return int(status);
  status = cudaFuncGetAttributes(&attributes, recur);
  if (status != cudaSuccess) return int(status);
  status = cudaFuncSetAttribute(prepare, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SharedStorageK1<L1>));
  if (status != cudaSuccess) return int(status);
  return int(cudaFuncSetAttribute(recur, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SharedStorageK2<L2,INPUT_STAGES,OUTPUT_STAGES>)));
}

// The Rust constructor validates byte extents, alignment, disjointness and
// scratch offsets before entry. One sequence's canonical FP32 state is updated
// in place. This function only enqueues work on the supplied stream.
extern "C" int atlas_glm_kda_prefill(const AtlasKdaParams* p) {
  using namespace atlas_kda_detail;
  if (!p || p->tokens < 2048 || p->tokens > 4100 || p->reserved != 0)
    return int(cudaErrorInvalidValue);
  // Do not silently erase a previous asynchronous failure and then mutate state.
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  const int t = int(p->tokens), tiles = (t + CHUNK - 1) / CHUNK;
  auto stream = reinterpret_cast<cudaStream_t>(p->stream);
  const BF16* qkv = ptr<const BF16>(p->qkv);
  const BF16* q = qkv;
  const BF16* k = qkv + H * D;
  const BF16* v = qkv + 2 * H * D;
  char* work = ptr<char>(p->workspace);
  BF16* beta = reinterpret_cast<BF16*>(work + p->layout.beta);

  // Read the engine's interleaved [T,3,H,D] convolution output directly.
  // Q/K are raw: the prepare kernel applies its own norm with epsilon1e-6.
  auto qkv_layout = make_layout(make_shape(H,t,D),make_stride(D,3*H*D,1));
  auto plane_layout = make_layout(make_shape(H,t,D),make_stride(D,H*D,1));
  auto beta_layout = make_layout(make_shape(H*t));
  auto state_layout = make_layout(make_shape(H,D,D),LayoutRight{});
  auto bias_layout = make_layout(make_shape(H,D),LayoutRight{});
  auto ws_plane = make_layout(make_shape(H*tiles,CHUNK,D),LayoutRight{});
  auto ws_gate = make_layout(make_shape(H*tiles,D),LayoutRight{});
  auto ws_matrix = make_layout(make_shape(H*tiles,CHUNK,CHUNK),LayoutRight{});

  auto mq = make_tensor(make_gmem_ptr(q),qkv_layout);
  auto mk = make_tensor(make_gmem_ptr(k),qkv_layout);
  auto mv = make_tensor(make_gmem_ptr(v),qkv_layout);
  auto mg = make_tensor(make_gmem_ptr(ptr<const BF16>(p->gate)),plane_layout);
  auto mb = make_tensor(make_gmem_ptr(beta),beta_layout);
  auto mo = make_tensor(make_gmem_ptr(ptr<BF16>(p->output)),plane_layout);
  auto md = make_tensor(make_gmem_ptr(ptr<const float>(p->dt_bias)),bias_layout);
  auto ms = make_tensor(make_gmem_ptr(ptr<float>(p->state)),state_layout);
  auto kd = make_tensor(make_gmem_ptr(reinterpret_cast<BF16*>(work+p->layout.kd)),ws_plane);
  auto qd = make_tensor(make_gmem_ptr(reinterpret_cast<BF16*>(work+p->layout.qd)),ws_plane);
  auto kr = make_tensor(make_gmem_ptr(reinterpret_cast<BF16*>(work+p->layout.kr)),ws_plane);
  auto gt = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(work+p->layout.gt)),ws_gate);
  auto inv = make_tensor(make_gmem_ptr(reinterpret_cast<BF16*>(work+p->layout.inv)),ws_matrix);
  auto mqk = make_tensor(make_gmem_ptr(reinterpret_cast<BF16*>(work+p->layout.mqk)),ws_matrix);

  auto load_q = make_tma_copy(SM90_TMA_LOAD{},mq,typename L1::TMAQKLayout{});
  auto load_k = make_tma_copy(SM90_TMA_LOAD{},mk,typename L1::TMAQKLayout{});
  auto load_g = make_tma_copy(SM90_TMA_LOAD{},mg,typename L1::TMAQKLayout{});
  auto load_b = make_tma_copy(SM90_TMA_LOAD{},mb,typename L1::TMABetaSmemLayout{});
  auto load_d = make_tma_copy(SM90_TMA_LOAD{},md,typename L1::TMAGTotalSmemLayout{});
  auto store_kd = make_tma_copy(SM90_TMA_STORE{},kd,typename L1::TMAVOLayout{});
  auto store_qd = make_tma_copy(SM90_TMA_STORE{},qd,typename L1::TMAVOLayout{});
  auto store_kr = make_tma_copy(SM90_TMA_STORE{},kr,typename L1::TMAVOLayout{});
  auto store_gt = make_tma_copy(SM90_TMA_STORE{},gt,typename L1::TMAGTotalSmemLayout{});
  auto store_inv = make_tma_copy(SM90_TMA_STORE{},inv,typename L1::TMALMLayout{});
  auto store_mqk = make_tma_copy(SM90_TMA_STORE{},mqk,typename L1::TMALMLayout{});
  auto load_v = make_tma_copy(SM90_TMA_LOAD{},mv,typename L1::TMAVOLayout{});
  auto load_kd = make_tma_copy(SM90_TMA_LOAD{},kd,typename L1::TMAVOLayout{});
  auto load_qd = make_tma_copy(SM90_TMA_LOAD{},qd,typename L1::TMAVOLayout{});
  auto load_kr = make_tma_copy(SM90_TMA_LOAD{},kr,typename L1::TMAVOLayout{});
  auto load_gt = make_tma_copy(SM90_TMA_LOAD{},gt,typename L1::TMAGTotalSmemLayout{});
  auto load_inv = make_tma_copy(SM90_TMA_LOAD{},inv,typename L1::TMALMLayout{});
  auto load_mqk = make_tma_copy(SM90_TMA_LOAD{},mqk,typename L1::TMALMLayout{});
  auto load_s = make_tma_copy(SM90_TMA_LOAD{},ms,typename L2::TMAFP32StateSmemLayout{});
  auto store_s = make_tma_copy(SM90_TMA_STORE{},ms,typename L2::TMAFP32StateSmemLayout{});
  auto store_o = make_tma_copy(SM90_TMA_STORE{},mo,typename L1::TMAVOLayout{});

  static_assert(std::is_same_v<decltype(load_q), load_q_type>);
  static_assert(std::is_same_v<decltype(load_k), load_k_type>);
  static_assert(std::is_same_v<decltype(load_g), load_g_type>);
  static_assert(std::is_same_v<decltype(load_b), load_b_type>);
  static_assert(std::is_same_v<decltype(load_d), load_d_type>);
  static_assert(std::is_same_v<decltype(store_kd), store_kd_type>);
  static_assert(std::is_same_v<decltype(store_qd), store_qd_type>);
  static_assert(std::is_same_v<decltype(store_kr), store_kr_type>);
  static_assert(std::is_same_v<decltype(store_gt), store_gt_type>);
  static_assert(std::is_same_v<decltype(store_inv), store_inv_type>);
  static_assert(std::is_same_v<decltype(store_mqk), store_mqk_type>);
  static_assert(std::is_same_v<decltype(load_v), load_v_type>);
  static_assert(std::is_same_v<decltype(load_kd), load_kd_type>);
  static_assert(std::is_same_v<decltype(load_qd), load_qd_type>);
  static_assert(std::is_same_v<decltype(load_kr), load_kr_type>);
  static_assert(std::is_same_v<decltype(load_gt), load_gt_type>);
  static_assert(std::is_same_v<decltype(load_inv), load_inv_type>);
  static_assert(std::is_same_v<decltype(load_mqk), load_mqk_type>);
  static_assert(std::is_same_v<decltype(load_s), load_s_type>);
  static_assert(std::is_same_v<decltype(store_s), store_s_type>);
  static_assert(std::is_same_v<decltype(store_o), store_o_type>);
  const int smem1 = sizeof(SharedStorageK1<L1>);
  const int smem2 = sizeof(SharedStorageK2<L2,INPUT_STAGES,OUTPUT_STAGES>);
  status = cudaFuncSetAttribute(prepare,cudaFuncAttributeMaxDynamicSharedMemorySize,smem1);
  if (status != cudaSuccess) return int(status);
  status = cudaFuncSetAttribute(recur,cudaFuncAttributeMaxDynamicSharedMemorySize,smem2);
  if (status != cudaSuccess) return int(status);

  stage_beta<<<(H*t+255)/256,256,0,stream>>>(ptr<const BF16>(p->beta),beta,t);
  status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  prepare<<<dim3(tiles,H),K1_THREADS,smem1,stream>>>(
      load_q,load_k,load_b,load_g,load_d,store_kd,store_qd,store_kr,store_gt,
      store_inv,store_mqk,1.0f/sqrtf(float(D)),t,H,1,nullptr,tiles,
      ptr<const float>(p->a_log),-5.0f*1.4426950408889634f,nullptr);
  status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  recur<<<dim3(1,H),K2_THREADS,smem2,stream>>>(
      load_v,load_b,load_kd,load_qd,load_kr,load_gt,load_inv,load_mqk,
      load_s,store_s,store_o,ptr<BF16>(p->output),t,H,1,nullptr,tiles);
  return int(cudaGetLastError());
}
