#include <cstdint>
#include <cuda_runtime.h>

/* mask = 1 << pos;
 * masked = num & mask;
 * bit = masked >> pos
 */
__device__ inline uint32_t getBit(uint32_t num, int pos) {
    return (num & (1 << pos)) >> pos;
}

/* mask = 1 << pos
 * set = num | mask
 */

__device__ inline uint32_t setBit(uint32_t num, int pos) {
    return num | (1 << pos);
}

/* mask = ~(1 << pos)
 * unset = num & mask
 */

__device__ inline uint32_t unsetBit(uint32_t num, int pos) {
    return num & (~(1 << pos));
}