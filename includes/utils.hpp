#ifndef _UTIL_H
#define _UTIL_H

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

/* mask = 1 << pos;
 * masked = num & mask;
 * bit = masked >> pos
 */

template <typename T> __device__ inline T getBit(T num, short pos) {
    return (num & (1 << pos)) >> pos;
}

/* mask = 1 << pos
 * set = num | mask
 */

template <typename T> __device__ inline T setBit(T num, short pos) {
    return num | (1 << pos);
}

/* mask = ~(1 << pos)
 * unset = num & mask
 */

template <typename T> __device__ inline T unsetBit(T num, short pos) {
    return num & (~(1 << pos));
}

/* dim3 << overload */
std::ostream &operator<<(std::ostream &os, const dim3 d);

#endif // end _UTIL_H