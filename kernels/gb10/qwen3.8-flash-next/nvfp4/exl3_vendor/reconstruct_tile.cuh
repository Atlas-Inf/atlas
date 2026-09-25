template <int K, int cb, bool HALF = false>
__device__ __forceinline__
void reconstruct_tile
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb, HALF>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
//    __shared__ half tile[16 * 8 * 16];
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}
