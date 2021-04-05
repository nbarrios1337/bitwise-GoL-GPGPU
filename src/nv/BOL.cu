#include "compute.hpp"
#include <iostream>

#define SIZE 1 << 6

// (SIZE / 4) by SIZE elements
#define Y_DIM SIZE
#define X_DIM (SIZE) >> 2
#define NUM_ELEMENTS (X_DIM * Y_DIM)

#define NUM_THREADS 32
// number of blocks >= num elements to process
#define NUM_BLOCKS (NUM_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS

// With the above, all kernel calls should use <<<NUM_BLOCKS, NUM_THREADS>>>

// cudaMalloc the mem then pass here
__global__ void init_grid(uint32_t *g) {
    // grid-stride loop

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < NUM_ELEMENTS; index += blockDim.x * gridDim.x) {
        // arbitrary init value, we'll see how random it is
        g[index] = 1; //(gridDim.x + blockIdx.x + blockDim.x + threadIdx.x) & 0x1;
    }
}

int main() {
    uint32_t *grid = NULL;
    uint32_t *tmpGrid = NULL;

    cudaMallocManaged(&grid, NUM_ELEMENTS * sizeof(uint32_t));
    cudaMallocManaged(&tmpGrid, NUM_ELEMENTS * sizeof(uint32_t));

    // init on host or device?
    init_grid<<<NUM_BLOCKS, NUM_THREADS>>>(grid);
    
    // CALL THIS BEFORE ANY DEVICE -> HOST MEM ACCESS
    cudaDeviceSynchronize();

    int count = 0;
    for(int i = 0; i < Y_DIM; i++) {
        for(int j = 0; j < X_DIM; j++){
            std::cout << grid[i * X_DIM + j];
            count++;
        }
        std::cout << std::endl;
    }

    std::cout << count << ", " << NUM_ELEMENTS << std::endl;
    cudaFree(grid);
    cudaFree(tmpGrid);

    return 0;
}
