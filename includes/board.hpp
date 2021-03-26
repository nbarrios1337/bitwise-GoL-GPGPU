#include "compute.hpp"

/* Looking down a row, we'd have 32 ints * 32 bits/int for a total
 * of 1024
 */

// 2^10 = 1024
#define Y_DIM 1 << 10

// 2^5 = 32 == sizeof(int)
// 1024 >> 5 == 1024 / 32
#define X_DIM Y_DIM >> 5

