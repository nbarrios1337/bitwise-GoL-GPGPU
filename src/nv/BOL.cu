#include "compute.hpp"
#include <bitset>
#include <iostream>
#include <random>

#define SIZE 1 << 6

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

// With the above, all kernel calls should use <<<NUM_BLOCKS, NUM_THREADS>>>

// cudaMalloc the mem then pass here
void init_grid(uint32_t *g) {
    for (int i = 1; i < Y_DIM + 1; i++) {
        for (int j = 1; j < X_DIM + 1; j++) {
            g[i * (X_DIM + 2) + j] = rand();
        }
    }
}

__global__ void simulate(uint32_t *g) {
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int index = iy * (X_DIM + 2) + ix;
#ifdef DEBUG
    // printf("index: %d (%d * %d, %d)\n", index, iy, (X_DIM + 2), ix);
    printf("Blk: (%d,%d) Thread: (%d,%d) -> Row/Col = (%d,%d)\tindex: %d (%d * "
           "%d, %d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, ix, iy, index, iy,
           (X_DIM + 2), ix);

    g[index] = threadIdx.y;
#else
    // TODO: get surrounding ints and then compute
    g[index] = g[index];
#endif
}

/* Dimensions
 * A single thread concerns itself with [0-31] bit cells in a row
 * ...][010...110][...
 *
 * A block of (1, 32) threads concerns itself with a 32x32 submatrix
 * -----------
 * [010...110]
 * ... x 30
 * [100...010]
 * -----------
 *
 * The grid needs to cover the SIZE x SIZE matrix with these submatrices
 * X_DIM happens to solve this in both directions
 * - rows divided up to fulfill 32*n == SIZE, where n == # threads
 * - cols divided up to fulfill 32*n == SIZE, where n == # blocks
 */
int main() {
#ifdef DEBUG
    std::cout << "size: " << (SIZE) << std::endl;
    std::cout << "x dim: " << (X_DIM + 2) << " (" << (X_DIM) << "+2)"
              << std::endl;
    std::cout << "y dim: " << (Y_DIM + 2) << " (" << (Y_DIM) << "+2)"
              << std::endl;
    std::cout << "Num integers (x by y): " << NUM_INTS << std::endl;
    std::cout << "Num blocks necessary: " << NUM_BLOCKS << std::endl;
#endif

    uint32_t *grid = NULL;
    uint32_t *tmpGrid = NULL;

    uint32_t *bitsets = NULL;
    uint32_t *out = NULL;

    cudaMallocManaged(&grid, TOTAL_INTS * sizeof(uint32_t));
    cudaMallocManaged(&tmpGrid, TOTAL_INTS * sizeof(uint32_t));

    cudaMallocManaged(&bitsets, 9 * sizeof(uint32_t));
    cudaMallocManaged(&out, sizeof(uint32_t));

    // init on host
    init_grid(grid);

    // Adapted from ORNL
    dim3 block_size(1, NUM_THREADS);
    dim3 grid_size(X_DIM, X_DIM);

#ifdef DEBUG
    std::cout << "block_size: " << block_size << std::endl;
    std::cout << "grid_size: " << grid_size << std::endl;
    std::cout << "Calling simulate<<<" << block_size << ", " << grid_size
              << ">>>" << std::endl;
#endif

    simulate<<<grid_size, block_size>>>(grid);

    // CALL THIS BEFORE ANY DEVICE -> HOST MEM ACCESS
    cudaDeviceSynchronize();

    int count = 0;
    for (int i = 0; i < Y_DIM + 2; i++) {
        for (int j = 0; j < X_DIM + 2; j++) {
#ifdef DEBUG
            std::cout << std::bitset<32>(grid[i * (X_DIM + 2) + j]).to_ullong()
                      << ' ';
#else
            std::cout << std::bitset<32>(grid[i * (X_DIM + 2) + j]) << ' ';
#endif
            count += 32;
        }
        std::cout << std::endl;
    }

#ifdef DEBUG
    std::cout << "Sizes " << (count == TOTAL_ELEMENTS ? "" : "not ")
              << "match: ";
    std::cout << count << " counted, " << TOTAL_ELEMENTS << " total"
              << std::endl;
#endif

    cudaFree(grid);
    cudaFree(tmpGrid);
    cudaFree(bitsets);
    cudaFree(out);

    return 0;
}
