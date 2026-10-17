/**
 * OSIRIS Portable Measurement Harness
 *
 * A clean, turn-key GPU ISA timing characterization kernel.
 * Supports SM70 (Volta) through SM120+ (Blackwell).
 *
 * Design:
 *   Each measurement "slot" is a 7-instruction sequence:
 *     Slot 0: RESET    - patchable (FADD placeholder)
 *     Slot 1: CS2R     - FIXED clock read (R_START)
 *     Slot 2: TRIGGER  - patchable (FMUL placeholder, NOP in cold)
 *     Slot 3: MEASURE  - patchable (FADD placeholder)
 *     Slot 4: CS2R     - FIXED clock read (R_END)
 *     Slot 5: IADD3    - FIXED delta computation
 *     Slot 6: STG      - FIXED store delta
 *
 * Marker strategy:
 *   - RESET  = FADD (add.f32) — identified by FADD opcode
 *   - TRIGGER = FMUL (mul.f32) — identified by FMUL opcode (distinct from FADD)
 *   - MEASURE = FADD (add.f32) — the FADD after the FMUL in each slot group
 *   - BAR.SYNC 0 between slots for isolation
 *
 * The offset extractor identifies patchable locations by:
 *   1. Finding all FMUL instructions (TRIGGER markers)
 *   2. For each FMUL: closest preceding FADD = RESET, closest following FADD = MEASURE
 *   3. CS2R instructions before and after each FADD group = clock reads
 *
 * Binary patching:
 *   - Bytes 0-12: opcode and operands (replaced from instruction encoding)
 *   - Byte 13: delay bits merged (instruction's delay, harness scheduling preserved)
 *   - Bytes 14-15: scoreboard bits (preserved for Mode 1, patched for Mode 2)
 *
 * Modes:
 *   Mode 1 (Issue Latency): No scoreboard dependencies — measures pipeline contention
 *   Mode 2 (Completion Latency): Full dependency chain via scoreboard barriers
 *
 * Usage:
 *   Compile: nvcc -gencode arch=compute_XX,code=sm_XX -O2 --no-compress -o harness harness_template.cu
 *   Run:     ./harness [--verbose] [--device N]
 *   Output:  Per-slot timing deltas to stdout in parseable format
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Number of measurement slots. Each slot = one (RESET, TRIGGER, MEASURE) triple.
 * 4096 is a good balance: large enough for statistical power, small enough for
 * fast compilation and reasonable binary size. */
#ifndef NUM_SLOTS
#define NUM_SLOTS 4096
#endif

/**
 * Single OSIRIS measurement slot.
 *
 * Each slot contains:
 *   BAR.SYNC 0       — warp-level barrier for isolation
 *   FADD  (RESET)    — patchable: bring pipeline to known state
 *   FMUL  (TRIGGER)  — patchable: instruction under test (NOP-patched in cold)
 *   CS2R  (START)    — read clock (fixed, never patched)
 *   FADD  (MEASURE)  — patchable: instruction to measure
 *   CS2R  (END)      — read clock (fixed, never patched)
 *   IADD3+STG        — compute delta and store (done on host for simplicity)
 *
 * Register usage:
 *   tmp_a, tmp_b, tmp_c: float operands for RESET/TRIGGER/MEASURE
 *   start_reg, end_reg: clock cycle counters
 *   Sink arrays: prevent cross-slot optimization
 */
#define OSIRIS_SLOT(N) \
    do { \
        asm volatile("bar.sync 0;"); \
        /* Load fresh temporaries from sinks — prevents cross-slot optimization */ \
        float tmp_a = sink_a[N]; \
        float tmp_b = sink_b[N]; \
        float tmp_c = sink_c[N]; \
        /* RESET — patchable FADD (bring microarch to known state) */ \
        asm volatile("add.f32 %0, %1, %2;" : "=f"(tmp_a) : "f"(tmp_a), "f"(tmp_b)); \
        /* TRIGGER — patchable FMUL (NOP-patched in cold binary) */ \
        asm volatile("mul.f32 %0, %1, %2;" : "=f"(tmp_b) : "f"(tmp_b), "f"(tmp_a)); \
        /* START CLOCK — fixed, never patched */ \
        asm volatile("mov.u32 %0, %%clock;" : "=r"(start_reg)); \
        /* MEASURE — patchable FADD (instruction timing target) */ \
        asm volatile("add.f32 %0, %1, %2;" : "=f"(tmp_c) : "f"(tmp_c), "f"(tmp_a)); \
        /* END CLOCK — fixed, never patched */ \
        asm volatile("mov.u32 %0, %%clock;" : "=r"(end_reg)); \
        /* Store clock samples to global memory */ \
        d_start[N] = start_reg; \
        d_end[N] = end_reg; \
        /* Write back to sinks — prevents dead-code elimination */ \
        sink_a[N] = tmp_a; \
        sink_b[N] = tmp_b; \
        sink_c[N] = tmp_c; \
        asm volatile("bar.sync 0;"); \
    } while(0)

/* ======================================================================
 * Unroll macros — binary power-of-2 expansion for N slots
 * ====================================================================== */
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

/* Top-level expansion depends on NUM_SLOTS */
#if NUM_SLOTS == 1024
#define SLOTS_ALL SLOTS_1024(0)
#elif NUM_SLOTS == 2048
#define SLOTS_ALL SLOTS_2048(0)
#elif NUM_SLOTS == 4096
#define SLOTS_ALL SLOTS_4096(0)
#else
#error "NUM_SLOTS must be 1024, 2048, or 4096"
#endif

/**
 * OSIRIS Portable Timing Kernel
 *
 * Args:
 *   d_start     — output: per-slot start clock values
 *   d_end       — output: per-slot end clock values
 *   sink_a/b/c  — float sink arrays (prevent optimization)
 *   mem_buffer  — general-purpose memory buffer (for patched LDG/STG)
 *
 * Launch config:
 *   <<<1, 1>>>     — single-warp measurement (default)
 *   <<<N, 128>>>   — multi-warp (lane 0 per warp stores)
 */
__global__ void osiris_portable_kernel(
    unsigned int* d_start,
    unsigned int* d_end,
    float* sink_a,
    float* sink_b,
    float* sink_c,
    void* mem_buffer
) {
    int tid = threadIdx.x;
    if (tid != 0) return;

    unsigned int start_reg, end_reg;

    /* Set up a 64-bit memory address in case patched instructions are LDG/STG/ATOM.
     * The address register pair (typically R2/R3 after compilation) will hold
     * a valid device pointer. */
    unsigned long long addr = (unsigned long long)mem_buffer;
    asm volatile("mov.u64 %0, %1;" : "=l"(addr) : "l"(addr));

    /* Keep addr live throughout the kernel to prevent register reallocation */
    asm volatile("" :: "l"(addr));

    /* Execute all measurement slots */
    SLOTS_ALL

    /* Final sink to prevent addr from being optimized out */
    if (addr == 0xDEADDEADDEADDEADULL) {
        sink_a[63] = 0.0f;
    }
}

/* ======================================================================
 * Host-side driver
 * ====================================================================== */

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int device = 0;
    int verbose = 0;

    /* Parse command line */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    /* Query device info */
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    if (verbose) {
        fprintf(stderr, "OSIRIS Portable Harness\n");
        fprintf(stderr, "=======================\n");
        fprintf(stderr, "Device: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);
        fprintf(stderr, "Slots: %d\n", NUM_SLOTS);
    }

    /* Allocate device memory */
    unsigned int *dd_start, *dd_end;
    float *d_sink_a, *d_sink_b, *d_sink_c;
    void *d_mem_buffer;

    check_cuda(cudaMalloc(&dd_start, NUM_SLOTS * sizeof(unsigned int)), "malloc start");
    check_cuda(cudaMalloc(&dd_end, NUM_SLOTS * sizeof(unsigned int)), "malloc end");
    check_cuda(cudaMalloc(&d_sink_a, NUM_SLOTS * sizeof(float)), "malloc sink_a");
    check_cuda(cudaMalloc(&d_sink_b, NUM_SLOTS * sizeof(float)), "malloc sink_b");
    check_cuda(cudaMalloc(&d_sink_c, NUM_SLOTS * sizeof(float)), "malloc sink_c");
    check_cuda(cudaMalloc(&d_mem_buffer, 1024 * 1024), "malloc mem_buffer");

    /* Initialize sinks with distinct float values */
    float *init = (float*)malloc(NUM_SLOTS * sizeof(float));
    for (int i = 0; i < NUM_SLOTS; i++) init[i] = 1.0f + i * 0.01f;
    cudaMemcpy(d_sink_a, init, NUM_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sink_b, init, NUM_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sink_c, init, NUM_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
    free(init);
    cudaMemset(d_mem_buffer, 0, 1024 * 1024);

    /* Warmup: 3 kernel launches to prime caches and TLBs */
    if (verbose) fprintf(stderr, "Warmup (3 launches)...\n");
    for (int i = 0; i < 3; i++) {
        osiris_portable_kernel<<<1, 1>>>(dd_start, dd_end,
            d_sink_a, d_sink_b, d_sink_c, d_mem_buffer);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");

    /* Measurement launch */
    if (verbose) fprintf(stderr, "Measurement launch...\n");
    osiris_portable_kernel<<<1, 1>>>(dd_start, dd_end,
        d_sink_a, d_sink_b, d_sink_c, d_mem_buffer);
    check_cuda(cudaDeviceSynchronize(), "measure");

    /* Copy results to host */
    unsigned int *h_start = (unsigned int*)malloc(NUM_SLOTS * sizeof(unsigned int));
    unsigned int *h_end = (unsigned int*)malloc(NUM_SLOTS * sizeof(unsigned int));
    check_cuda(cudaMemcpy(h_start, dd_start, NUM_SLOTS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost), "copy start");
    check_cuda(cudaMemcpy(h_end, dd_end, NUM_SLOTS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost), "copy end");

    /* Output per-slot timing in parseable format */
    printf("PERSLOT_TIMING_START\n");
    for (int i = 0; i < NUM_SLOTS; i++) {
        unsigned int delta = h_end[i] - h_start[i];
        printf("%d,%u\n", i, delta);
    }
    printf("PERSLOT_TIMING_END\n");
    printf("SUCCESS: %d slots timed\n", NUM_SLOTS);

    /* Cleanup */
    free(h_start);
    free(h_end);
    cudaFree(dd_start);
    cudaFree(dd_end);
    cudaFree(d_sink_a);
    cudaFree(d_sink_b);
    cudaFree(d_sink_c);
    cudaFree(d_mem_buffer);

    return 0;
}
