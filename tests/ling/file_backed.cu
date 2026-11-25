// SPDX-License-Identifier: AGPL-3.0-only
// Probe a read-only file mapping with a CUDA kernel. Arguments are a shard,
// an absolute byte offset and a byte count. No host registration or device copy.
#include <cuda_runtime.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

__global__ void page_sums(const uint8_t* src, uint64_t count, uint64_t* out) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sum = 0;
    for (uint64_t j = i; j < count; j += gridDim.x * blockDim.x)
        sum += src[j];
    out[i] = sum;
}

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 1; } } while (0)

static bool parse_u64(const char* s, uint64_t& value) {
    if (!s[0] || s[0] == '-') return false;
    char* end = nullptr;
    errno = 0;
    value = std::strtoull(s, &end, 10);
    return !errno && *end == 0;
}
int main(int argc, char** argv) {
    uint64_t offset, count;
    if (argc != 4 || !parse_u64(argv[2], offset) || !parse_u64(argv[3], count) || !count)
        return 2;
    int access, tables;
    CHECK(cudaDeviceGetAttribute(&access, cudaDevAttrPageableMemoryAccess, 0));
    CHECK(cudaDeviceGetAttribute(&tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    if (!access || !tables) { std::fprintf(stderr,"device lacks pageable host-page-table access\n"); return 1; }
    const int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd,&st) || st.st_size < 0 ||
        offset > static_cast<uint64_t>(st.st_size) || count > st.st_size-offset) return 2;
    const size_t page = sysconf(_SC_PAGESIZE);
    const uint64_t base = offset - offset % page;
    const size_t len = count + (offset-base);
    void* mapping = mmap(nullptr,len,PROT_READ,MAP_PRIVATE,fd,base);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap"); return 1; }
    const auto* src = static_cast<const uint8_t*>(mapping)+(offset-base);
    constexpr int threads = 8 * 128;
    uint64_t *device, partial[threads];
    CHECK(cudaMalloc(&device,sizeof(partial)));
    page_sums<<<8,128>>>(src,count,device);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(partial,device,sizeof(partial),cudaMemcpyDeviceToHost));
    uint64_t actual = 0, expected = 0;
    for (auto v : partial) actual += v;
    for (uint64_t i=0; i<count; ++i) expected += src[i];
    std::printf("file-mapped bytes=%llu GPU=%llu CPU=%llu\n",
                (unsigned long long)count,(unsigned long long)actual,(unsigned long long)expected);
    CHECK(cudaFree(device));
    munmap(mapping,len);
    return actual != expected;
}
