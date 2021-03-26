#include "utils.hpp"

#include <cstdint>
#include <cuda_runtime.h>

__global__ void next_gen(uint32_t *out, uint32_t *bitsets, uint64_t top,
                         uint64_t center, uint64_t bottom);

__device__ void get_bitsets(uint32_t *bitsets, uint64_t top, uint64_t center,
                            uint64_t bottom);

__device__ uint32_t bitwise_sum63(uint32_t *bs);

uint32_t wrapper(uint64_t t, uint64_t m, uint64_t b);