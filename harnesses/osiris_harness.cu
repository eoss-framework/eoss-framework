/**
 * OSIRIS parametrized measurement harness.
 *
 * A single template that dynamically assembles a measurement kernel from
 * feature blocks toggled at compile time, so one source covers every GPU
 * generation and every resource "flavor" (a campaign harness with texture +
 * surface + shared, a TMA harness, a cluster harness, ...). The build passes
 * ``-DFEAT_*=1`` for the features a given generation supports (see
 * eoss.isa.registry capability profiles); the offset extractor and register
 * map are then read back from the compiled binary.
 *
 * Features (GLOBAL buffer + the OSIRIS slot grammar are always present):
 *   FEAT_SHARED   __shared__ region            (LDS/STS/LDL/STL)          all gens
 *   FEAT_TEXTURE  bound texture object         (TEX/TLD/TXQ)              all gens
 *   FEAT_SURFACE  bound surface object         (SULD/SUST)                all gens
 *   FEAT_TMA      CUtensorMap + mbarrier       (UTMALDG/STG/PF/REDG)      sm_90+
 *   FEAT_CLUSTER  thread-block cluster         (UCGABAR arrive/wait)      sm_90+
 *
 * Kernel name is fixed ("osiris_kernel") so a single kernel_hint works.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if FEAT_TMA
#include <cuda.h>
#endif
#if FEAT_CLUSTER
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

#ifndef NUM_SLOTS
#define NUM_SLOTS 4096
#endif

#define OSIRIS_SLOT(N) \
    do { \
        asm volatile("bar.sync 0;"); \
        float tmp_a = sink_a[N]; \
        float tmp_b = sink_b[N]; \
        float tmp_c = sink_c[N]; \
        asm volatile("add.f32 %0, %1, %2;" : "=f"(tmp_a) : "f"(tmp_a), "f"(tmp_b)); \
        asm volatile("mul.f32 %0, %1, %2;" : "=f"(tmp_b) : "f"(tmp_b), "f"(tmp_a)); \
        asm volatile("mov.u32 %0, %%clock;" : "=r"(start_reg)); \
        asm volatile("add.f32 %0, %1, %2;" : "=f"(tmp_c) : "f"(tmp_c), "f"(tmp_a)); \
        asm volatile("mov.u32 %0, %%clock;" : "=r"(end_reg)); \
        d_start[N] = start_reg; \
        d_end[N] = end_reg; \
        sink_a[N] = tmp_a; \
        sink_b[N] = tmp_b; \
        sink_c[N] = tmp_c; \
        asm volatile("bar.sync 0;"); \
    } while(0)

#define SLOTS_1(N)    OSIRIS_SLOT(N);
#define SLOTS_2(N)    SLOTS_1(N) SLOTS_1(N+1)
#define SLOTS_4(N)    SLOTS_2(N) SLOTS_2(N+2)
#define SLOTS_8(N)    SLOTS_4(N) SLOTS_4(N+4)
#define SLOTS_16(N)   SLOTS_8(N) SLOTS_8(N+8)
#define SLOTS_32(N)   SLOTS_16(N) SLOTS_16(N+16)
#define SLOTS_64(N)   SLOTS_32(N) SLOTS_32(N+32)
#define SLOTS_128(N)  SLOTS_64(N) SLOTS_64(N+64)
#define SLOTS_256(N)  SLOTS_128(N) SLOTS_128(N+128)
#define SLOTS_512(N)  SLOTS_256(N) SLOTS_256(N+256)
#define SLOTS_1024(N) SLOTS_512(N) SLOTS_512(N+512)
#define SLOTS_2048(N) SLOTS_1024(N) SLOTS_1024(N+1024)
#define SLOTS_4096(N) SLOTS_2048(N) SLOTS_2048(N+2048)

#if NUM_SLOTS == 1024
#define SLOTS_ALL SLOTS_1024(0)
#elif NUM_SLOTS == 2048
#define SLOTS_ALL SLOTS_2048(0)
#elif NUM_SLOTS == 4096
#define SLOTS_ALL SLOTS_4096(0)
#else
#error "NUM_SLOTS must be 1024, 2048, or 4096"
#endif

#if FEAT_TMA
__device__ __forceinline__ void tma_load(unsigned smem, unsigned bar,
                                          const CUtensorMap* tma, int coord) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global"
                 ".mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                 :: "r"(smem), "l"(tma), "r"(coord), "r"(bar) : "memory");
}
#endif

__global__ void
#if FEAT_CLUSTER
__cluster_dims__(2, 1, 1)
#endif
osiris_kernel(
    unsigned int* d_start,
    unsigned int* d_end,
    float* sink_a,
    float* sink_b,
    float* sink_c,
    void* mem_buffer,
    unsigned long long* atom_scratch
#if FEAT_TEXTURE
    , cudaTextureObject_t tex
#endif
#if FEAT_SURFACE
    , cudaSurfaceObject_t surf
#endif
#if FEAT_TMA
    , const __grid_constant__ CUtensorMap tma
#endif
) {
#if FEAT_SHARED
    __shared__ float smem[2048];
#endif
#if FEAT_TMA
    __shared__ alignas(128) int tma_smem[256];
    __shared__ alignas(8) unsigned long long tma_bar;
#endif
    int tid = threadIdx.x;
#if FEAT_SHARED
    for (int i = tid; i < 2048; i += blockDim.x) smem[i] = (float)i;
#endif
#if FEAT_CLUSTER
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
#elif FEAT_SHARED
    __syncthreads();
#endif
    if (tid != 0) return;

    unsigned int start_reg, end_reg;
    unsigned long long addr = (unsigned long long)mem_buffer;
    asm volatile("mov.u64 %0, %1;" : "=l"(addr) : "l"(addr));
    /* Aligned atomic scratch: emit a real global reduction so the compiled SASS
       carries an aligned (cudaMalloc'd, >=256B) atomic address in a live
       register the register-map extractor can target for ATOMG/RED/CAS. Without
       this, patched atomics inherit an offset address that isn't 8/16-byte
       aligned and fault with "misaligned address". */
    unsigned long long atom_addr = (unsigned long long)atom_scratch;
    unsigned long long atom_ret;
    /* A real global ATOMG (with return) + a RED, both on the aligned scratch.
       These give the byte-templating harvest a known-good, correctly-aligned
       ATOMG/RED exemplar to lift on every generation (the generator can't
       reliably produce a runnable global atomic — the address register isn't
       stable at the measure slot). */
    asm volatile("atom.global.add.u64 %0, [%1], %2;"
                 : "=l"(atom_ret) : "l"(atom_addr), "l"((unsigned long long)tid) : "memory");
    asm volatile("red.global.add.u64 [%0], %1;"
                 :: "l"(atom_addr), "l"(atom_ret) : "memory");
    asm volatile("" :: "l"(atom_addr), "l"(atom_ret));
#if FEAT_SHARED
    unsigned int shared_off = (unsigned)__cvta_generic_to_shared(&smem[0]);
    asm volatile("" :: "r"(shared_off));
#endif
#if FEAT_TMA
    unsigned int smem_ptr = (unsigned)__cvta_generic_to_shared(&tma_smem[0]);
    unsigned int bar_ptr = (unsigned)__cvta_generic_to_shared(&tma_bar);
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_ptr));
    asm volatile("fence.proxy.async.shared::cta;");
    tma_load(smem_ptr, bar_ptr, &tma, 0);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;" :: "r"(bar_ptr));
    asm volatile("{ .reg .pred p; LW: mbarrier.try_wait.parity.shared.b64 p, [%0], 0;"
                 " @!p bra LW; }" :: "r"(bar_ptr));
#endif
#if FEAT_TEXTURE
    float tval = tex2D<float>(tex, 0.5f, 0.5f);
#endif
#if FEAT_SURFACE
    float sval = 0.0f;
    surf2Dread(&sval, surf, 0, 0, cudaBoundaryModeZero);
#endif
    asm volatile("" :: "l"(addr));

    SLOTS_ALL

    /* Keep every enabled resource live across the whole slot region. */
    if (addr == 0xDEADDEADDEADDEADULL || atom_addr == 0xDEADDEADDEADDEADULL
        || end_reg == 0xFFFFFFFFu) {
        float keep = (float)(atom_addr & 0xFu);
#if FEAT_TEXTURE
        keep += tex2D<float>(tex, 1.5f, 1.5f) + tval;
#endif
#if FEAT_SURFACE
        float s2 = 0.0f; surf2Dread(&s2, surf, 4, 0, cudaBoundaryModeZero);
        keep += s2 + sval;
#endif
#if FEAT_TMA
        tma_load(smem_ptr, bar_ptr, &tma, 64);
        keep += (float)(tma_smem[0] + smem_ptr + bar_ptr);
#endif
#if FEAT_SHARED
        keep += smem[2] + (float)shared_off;
#endif
        sink_a[63] = keep;
    }
}

/* ====================================================================== */

static void check(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
    check(cudaSetDevice(device), "setDevice");

    unsigned int *dd_start, *dd_end;
    float *d_sink_a, *d_sink_b, *d_sink_c;
    void *d_mem_buffer;
    check(cudaMalloc(&dd_start, NUM_SLOTS * sizeof(unsigned int)), "start");
    check(cudaMalloc(&dd_end, NUM_SLOTS * sizeof(unsigned int)), "end");
    check(cudaMalloc(&d_sink_a, NUM_SLOTS * sizeof(float)), "sa");
    check(cudaMalloc(&d_sink_b, NUM_SLOTS * sizeof(float)), "sb");
    check(cudaMalloc(&d_sink_c, NUM_SLOTS * sizeof(float)), "sc");
    check(cudaMalloc(&d_mem_buffer, 1024 * 1024), "buf");
    unsigned long long *d_atom_scratch;
    check(cudaMalloc(&d_atom_scratch, 4096 * sizeof(unsigned long long)), "atom");
    cudaMemset(d_atom_scratch, 0, 4096 * sizeof(unsigned long long));
    float *init = (float*)malloc(NUM_SLOTS * sizeof(float));
    for (int i = 0; i < NUM_SLOTS; i++) init[i] = 1.0f + i * 0.01f;
    cudaMemcpy(d_sink_a, init, NUM_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sink_b, init, NUM_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sink_c, init, NUM_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
    free(init);
    cudaMemset(d_mem_buffer, 0, 1024 * 1024);

#if FEAT_TEXTURE || FEAT_SURFACE
    cudaChannelFormatDesc chan = cudaCreateChannelDesc<float>();
    cudaArray_t arr;
    check(cudaMallocArray(&arr, &chan, 256, 256, cudaArraySurfaceLoadStore), "mallocArray");
    float *harr = (float*)calloc(256 * 256, sizeof(float));
    check(cudaMemcpy2DToArray(arr, 0, 0, harr, 256 * sizeof(float),
                              256 * sizeof(float), 256, cudaMemcpyHostToDevice), "memcpyArr");
    free(harr);
    cudaResourceDesc resd; memset(&resd, 0, sizeof(resd));
    resd.resType = cudaResourceTypeArray; resd.res.array.array = arr;
#endif
#if FEAT_TEXTURE
    cudaTextureDesc texd; memset(&texd, 0, sizeof(texd));
    texd.addressMode[0] = cudaAddressModeClamp; texd.addressMode[1] = cudaAddressModeClamp;
    texd.filterMode = cudaFilterModePoint; texd.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex = 0;
    check(cudaCreateTextureObject(&tex, &resd, &texd, NULL), "createTex");
#endif
#if FEAT_SURFACE
    cudaSurfaceObject_t surf = 0;
    check(cudaCreateSurfaceObject(&surf, &resd), "createSurf");
#endif
#if FEAT_TMA
    int *d_tensor;
    check(cudaMalloc(&d_tensor, 1024 * sizeof(int)), "tensor");
    cudaMemset(d_tensor, 0, 1024 * sizeof(int));
    CUtensorMap tma{};
    uint64_t gsize[1] = {1024}; uint64_t gstride[1] = {sizeof(int)};
    uint32_t bsize[1] = {256}; uint32_t estride[1] = {1};
    CUresult r = cuTensorMapEncodeTiled(
        &tma, CU_TENSOR_MAP_DATA_TYPE_INT32, 1, d_tensor, gsize, gstride,
        bsize, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "TMA encode failed: %d\n", r); return 1; }
#endif

#if FEAT_CLUSTER
    dim3 grid(2, 1, 1), block(32, 1, 1);
#else
    dim3 grid(1, 1, 1), block(32, 1, 1);
#endif
    for (int i = 0; i < 4; i++) {
        osiris_kernel<<<grid, block>>>(dd_start, dd_end, d_sink_a, d_sink_b,
                                       d_sink_c, d_mem_buffer, d_atom_scratch
#if FEAT_TEXTURE
            , tex
#endif
#if FEAT_SURFACE
            , surf
#endif
#if FEAT_TMA
            , tma
#endif
        );
        if (i == 2) check(cudaDeviceSynchronize(), "warmup");
    }
    check(cudaDeviceSynchronize(), "measure");

    unsigned int *h_start = (unsigned int*)malloc(NUM_SLOTS * sizeof(unsigned int));
    unsigned int *h_end = (unsigned int*)malloc(NUM_SLOTS * sizeof(unsigned int));
    cudaMemcpy(h_start, dd_start, NUM_SLOTS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end, dd_end, NUM_SLOTS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("PERSLOT_TIMING_START\n");
    for (int i = 0; i < NUM_SLOTS; i++) printf("%d,%u\n", i, h_end[i] - h_start[i]);
    printf("PERSLOT_TIMING_END\n");
    printf("SUCCESS: %d slots timed\n", NUM_SLOTS);
    return 0;
}
