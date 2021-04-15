#ifndef _UTIL_H
#define _UTIL_H

#include <cstdint>
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif
#include <iostream>

/* mask = 1 << pos;
 * masked = num & mask;
 * bit = masked >> pos
 */

template <typename T>
#ifdef __NVCC__
__device__
#endif
    inline T
    getBit(T num, short pos) {
    return (num & (1 << pos)) >> pos;
}

/* mask = 1 << pos
 * set = num | mask
 */

template <typename T>
#ifdef __NVCC__
__device__
#endif
    inline T
    setBit(T num, short pos) {
    return num | (1 << pos);
}

/* mask = ~(1 << pos)
 * unset = num & mask
 */

template <typename T>
#ifdef __NVCC__
__device__
#endif
    inline T
    unsetBit(T num, short pos) {
    return num & (~(1 << pos));
}

/* dim3 << overload */
#ifdef __NVCC__
std::ostream &operator<<(std::ostream &os, const dim3 d);
#endif

#endif // end _UTIL_H