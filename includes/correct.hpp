#include "utils.hpp"
#include <cstdint>
#include <cuda_runtime.h>

// Copy/Pasted code from BOL and GOL
// Ideally, we'd redefined BOL and GOL to be callable functions

// BOL Stuff
#define SIZE 1 << 5

// (SIZE / 4) by SIZE elements
// plus border of zeroes
#define Y_DIM (SIZE)
#define X_DIM (SIZE >> 5)

#define TOTAL_INTS ((X_DIM) + 2) * ((Y_DIM) + 2)
#define TOTAL_ELEMENTS (TOTAL_INTS << 5)

#define NUM_INTS (X_DIM) * (Y_DIM)
#define NUM_ELEMENTS (NUM_INTS << 5)

#define NUM_THREADS 32
// number of blocks >= num elements to process
#define NUM_BLOCKS (NUM_INTS + NUM_THREADS - 1) / NUM_THREADS

// GOL Stuff
#define BLOCK_SIZE 16

// GOL
__global__ void ghostRows(int dim, uint32_t *grid) {
  // We want id ∈ [1,dim]
  int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

  if (id <= dim) {
    // Copy first real row to bottom ghost row
    grid[(dim + 2) * (dim + 1) + id] = grid[(dim + 2) + id];
    // Copy last real row to top ghost row
    grid[id] = grid[(dim + 2) * dim + id];
  }
}

__global__ void ghostCols(int dim, uint32_t *grid) {
  // We want id ∈ [0,dim+1]
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id <= dim + 1) {
    // Copy first real column to right most ghost column
    grid[id * (dim + 2) + dim + 1] = grid[id * (dim + 2) + 1];
    // Copy last real column to left most ghost column
    grid[id * (dim + 2)] = grid[id * (dim + 2) + dim];
  }
}

__global__ void GOL(int dim, uint32_t *grid, uint32_t *newGrid) {
  // We want id ∈ [1,dim]
  int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int id = iy * (dim + 2) + ix;

  int numNeighbors;

  if (iy <= dim && ix <= dim) {

    // Get the number of neighbors for a given grid point
    numNeighbors = grid[id + (dim + 2)] + grid[id - (dim + 2)]   // upper lower
                   + grid[id + 1] + grid[id - 1]                 // right left
                   + grid[id + (dim + 3)] + grid[id - (dim + 3)] // diagonals
                   + grid[id - (dim + 1)] + grid[id + (dim + 1)];

    int cell = grid[id];
    // Here we have explicitly all of the game rules
    if (cell == 1 && numNeighbors < 2)
      newGrid[id] = 0;
    else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
      newGrid[id] = 1;
    else if (cell == 1 && numNeighbors > 3)
      newGrid[id] = 0;
    else if (cell == 0 && numNeighbors == 3)
      newGrid[id] = 1;
    else
      newGrid[id] = cell;
  }
}

// BOL
// See compute.cu for more information
// This function combines the functionalities of
// get_bitsets and bitwise_sum63 in order to
// remove the need for the bitsets array.
__device__ uint32_t compute63(uint64_t top, uint64_t center, uint64_t bottom) {
    // 111...100
    // uint32_t notTwoLSB = (~(uint32_t)0) << 2;
    // 001...111
    uint32_t notTwoMSB = unsetBit(unsetBit(~0, 31), 30);

    uint32_t upper_left = top >> 2;
    uint32_t upper = unsetBit(top >> 1, 31);
    uint32_t upper_right = top & notTwoMSB;

    uint32_t middle_left = center >> 2;
    uint32_t middle = unsetBit(center >> 1, 31);
    uint32_t middle_right = center & notTwoMSB;

    uint32_t lower_left = bottom >> 2;
    uint32_t lower = unsetBit(bottom >> 1, 31);
    uint32_t lower_right = bottom & notTwoMSB;

    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    // upper_left + upper addition (4 bitwise ops)
    s2 = upper_left & upper;
    s1 = upper_left ^ upper;
    s0 = ~(upper_left | upper);

    // upper_right addition (9 bitwise ops)
    uint32_t nc = ~upper_right;
    s3 = s2 & upper_right;
    s2 = (s2 & nc) | (s1 & upper_right);
    s1 = (s1 & nc) | (s0 & upper_right);
    s0 &= nc;

    // middle_left addition (11 b-ops)
    uint32_t nd = ~middle_left;
    s3 = (s3 & nd) | (s2 & middle_left);
    s2 = (s2 & nd) | (s1 & middle_left);
    s1 = (s1 & nd) | (s0 & middle_left);
    s0 &= nd;

    // middle_right add (11 b-ops)
    uint32_t ne = ~middle_right;
    s3 = (s3 & ne) | (s2 & middle_right);
    s2 = (s2 & ne) | (s1 & middle_right);
    s1 = (s1 & ne) | (s0 & middle_right);
    s0 &= ne;

    // lower_left add (11 b-ops)
    uint32_t nf = ~lower_left;
    s3 = (s3 & nf) | (s2 & lower_left);
    s2 = (s2 & nf) | (s1 & lower_left);
    s1 = (s1 & nf) | (s0 & lower_left);
    s0 &= nf;

    // lower add (10 b-ops)
    uint32_t ng = ~lower;
    s3 = (s3 & ng) | (s2 & lower);
    s2 = (s2 & ng) | (s1 & lower);
    s1 = (s1 & ng) | (s0 & lower);

    // lower_right add (7 b-ops)
    uint32_t nh = ~lower_right;
    s3 = (s3 & nh) | (s2 & lower_right);
    s2 = (s2 & nh) | (s1 & lower_right);

    return s3 | (middle & s2);
}

/* bitwise implementation requires a lot
 * fewer threads for the row, but fits
 * nicely when copying over columns
 */

// Call with X_DIM + 2 threads
__global__ void ghost_rows(uint32_t *g) {
    // Need X_DIM + 2 threads for one thread to take care
    // of a given column's first and last element,
    // INCLUDING the column's ghost row elements

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int bottom = (Y_DIM + 1) * (X_DIM + 2) + index;
    int top = 0 * (X_DIM + 2) + index;

#ifdef DEBUG
    printf("[rows] Blk: (%d,%d) Thread: (%d,%d) -> Col = (%d)\n", blockIdx.x,
           blockIdx.y, threadIdx.x, threadIdx.y, index);
    g[bottom] = 3;
    g[top] = 4;
#else
    // Mirror first real row to bottom ghost row
    g[bottom] = g[top + (X_DIM + 2)];
    // Mirror last real row to top ghost row
    g[top] = g[bottom - (X_DIM + 2)];
#endif
}

// Call with a block size as defined in main
__global__ void ghost_columns(uint32_t *g) {
    // Need a block_size (see main) amt of threads
    // one for each row of the two needed columns,
    // EXCLUDING the rows's ghost column elements.

    int index = blockDim.x * blockIdx.x + threadIdx.x + 1;

    int left = index * (X_DIM + 2) + 0;
    int right = index * (X_DIM + 2) + X_DIM + 1;

#ifdef DEBUG
    printf("[cols] Blk: (%d,%d) Thread: (%d,%d) -> Row = (%d)\n", blockIdx.x,
           blockIdx.y, threadIdx.x, threadIdx.y, index);
    printf("[cols] Row = (%d) -> Left: %d, Right: %d\n", index, left, right);
    g[right] = 1;
    g[left] = 2;
#else
    // Mirror first real column to right ghost column
    g[right] = g[left + 1];
    // Mirror last real column to left ghost column
    g[left] = g[right - 1];
#endif
}

/* Note: ghost_columns does NOT account for the diagonals
 * but ghost_rows DOES. Be sure to call ghost_columns FIRST
 * in order for ghost_rows to fill the diagonals.
 */

__global__ void simulate(uint32_t *g) {
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int index = iy * (X_DIM + 2) + ix;
#ifdef DEBUG
    // printf("index: %d (%d * %d, %d)\n", index, iy, (X_DIM + 2), ix);
    printf("[simulate] Blk: (%d,%d) Thread: (%d,%d) -> Row/Col = "
           "(%d,%d)\tindex: %d (%d * "
           "%d, %d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, ix, iy, index, iy,
           (X_DIM + 2), ix);

    // g[index] = 255;
#else
    // Get surrounding ints and then compute
    int top_index = (iy - 1) * (X_DIM + 2) + ix;
    int bottom_index = (iy + 1) * (X_DIM + 2) + ix;

    // make space for the LSB from the next number over's MSB
    uint64_t top_num = g[top_index] << 1;
    uint64_t center_num = g[index] << 1;
    uint64_t bottom_num = g[bottom_index] << 1;

    // since we shift in a 0, xor will set the LSB to
    // the bit from the next's MSB. See XOR truth table
    top_num ^= getBit(g[top_index + 1], 31);
    center_num ^= getBit(g[index + 1], 31);
    bottom_num ^= getBit(g[bottom_index + 1], 31);

    // explicitly specify type in order to avoid shifting out all bits
    top_num ^= getBit<uint64_t>(g[top_index - 1], 0) << 32;
    center_num ^= getBit<uint64_t>(g[index - 1], 0) << 32;
    bottom_num ^= getBit<uint64_t>(g[bottom_index - 1], 0) << 32;

    g[index] = compute63(top_num, center_num, bottom_num);
#endif
}

