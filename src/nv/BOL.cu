#include "compute.hpp"
#include <bitset>
#include <iostream>
#include <random>

#define SIZE 1 << 6

// (SIZE / 4) by SIZE elements
// plus border of zeroes
#define Y_DIM (SIZE) + 2
#define X_DIM (SIZE >> 5) + 2

#define NUM_INTS (X_DIM) * (Y_DIM)
#define NUM_ELEMENTS (NUM_INTS << 5)

#define NUM_THREADS 32
// number of blocks >= num elements to process
#define NUM_BLOCKS (NUM_INTS + NUM_THREADS - 1) / NUM_THREADS

// With the above, all kernel calls should use <<<NUM_BLOCKS, NUM_THREADS>>>

// cudaMalloc the mem then pass here
void init_grid(uint32_t *g) {
    for (int i = 1; i < Y_DIM - 1; i++) {
        for (int j = 1; j < X_DIM - 1; j++) {
            g[i * (X_DIM) + j] = rand();
        }
    }
}

int main() {
    std::cout << "size: " << (SIZE) << std::endl;
    std::cout << "x dim: " << (X_DIM) << " (" << (X_DIM)-2 << "+2)" << std::endl;
    std::cout << "y dim: " << (Y_DIM) << " (" << (Y_DIM)-2 << "+2)" << std::endl;
    std::cout << "num integers (x by y): " << NUM_INTS << std::endl;

    uint32_t *grid = NULL;
    uint32_t *tmpGrid = NULL;

    uint32_t *bitsets = NULL;
    uint32_t *out = NULL;

    cudaMallocManaged(&grid, NUM_INTS * sizeof(uint32_t));
    cudaMallocManaged(&tmpGrid, NUM_INTS * sizeof(uint32_t));

    cudaMallocManaged(&bitsets, 9 * sizeof(uint32_t));
    cudaMallocManaged(&out, sizeof(uint32_t));

    // init on host or device?
    init_grid(grid);

    // TODO call next_gen

    // CALL THIS BEFORE ANY DEVICE -> HOST MEM ACCESS
    cudaDeviceSynchronize();

    int count = 0;
    for (int i = 0; i < Y_DIM; i++) {
        for (int j = 0; j < X_DIM; j++) {
            std::cout << std::bitset<32>(grid[i * (X_DIM) + j]) << ' ';
            count += 32;
        }
        std::cout << std::endl;
    }

    std::cout << "Sizes " << (count == NUM_ELEMENTS ? "" : "not ") << "match: ";
    std::cout << count << " counted, " << NUM_ELEMENTS << " total" << std::endl;
    cudaFree(grid);
    cudaFree(tmpGrid);
    cudaFree(bitsets);
    cudaFree(out);

    return 0;
}
