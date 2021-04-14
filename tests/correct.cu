#include "compute.hpp"
#include "correct.hpp"
#include <bitset>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>


// cudaMalloc the mem then pass here
void init_grid(uint32_t *bg, uint32_t *gg) {
    for (int i = 1; i < Y_DIM + 1; i++) {
        for (int j = 1; j < X_DIM + 1; j++) {
            bg[i * (X_DIM + 2) + j] = rand();
            auto val = std::bitset<32>(bg[i * (X_DIM + 2) + j]);
            for (int k = 0; k < 32; k++) {
                gg[i * (Y_DIM + 2) + k] = val[k];
            }
        }
    }
}

int main() {
    srand(1985);

    uint32_t *GOL_grid = NULL;
    uint32_t *GOL_tmpGrid = NULL;
    uint32_t *BOL_grid = NULL;
    uint32_t *BOL_tmpGrid = NULL;

    uint32_t dim = SIZE;

    cudaMallocManaged(&GOL_grid, sizeof(uint32_t) * (dim + 2) * (dim + 2));
    cudaMallocManaged(&GOL_tmpGrid, sizeof(uint32_t) * (dim + 2) * (dim + 2));
    cudaMallocManaged(&BOL_grid, TOTAL_INTS * sizeof(uint32_t));
    cudaMallocManaged(&BOL_tmpGrid, TOTAL_INTS * sizeof(uint32_t));

    // BOL
    init_grid(BOL_grid, GOL_grid);

    // GOL, same data
    memccpy(GOL_grid, BOL_grid, TOTAL_INTS, sizeof(uint32_t));

    // GOL params

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    int linGrid = (int)ceil(dim / (float)BLOCK_SIZE);
    dim3 gridSize(linGrid, linGrid, 1);

    dim3 cpyBlockSize(BLOCK_SIZE);
    dim3 cpyGridRowsGridSize((int)ceil(dim / (float)cpyBlockSize.x));
    dim3 cpyGridColsGridSize((int)ceil((dim + 2) / (float)cpyBlockSize.x));

    // BOL params
    dim3 block_size(1, NUM_THREADS);
    dim3 grid_size(X_DIM, X_DIM);

    dim3 grid_size_gCols(X_DIM);
    dim3 block_size_gCols(NUM_THREADS);

    dim3 grid_size_gRows(1);
    dim3 block_size_gRows(X_DIM + 2);

    // GOL single iter
    ghostRows<<<cpyGridRowsGridSize, cpyBlockSize>>>(dim, GOL_grid);
    ghostCols<<<cpyGridColsGridSize, cpyBlockSize>>>(dim, GOL_grid);
    GOL<<<gridSize, blockSize>>>(dim, GOL_grid, GOL_tmpGrid);

    // BOL single iter
    ghost_columns<<<grid_size_gCols, block_size_gCols>>>(BOL_grid);
    ghost_rows<<<grid_size_gRows, block_size_gRows>>>(BOL_grid);
    simulate<<<grid_size, block_size>>>(BOL_grid);

    // checks
    std::cout << "BOL" << std::endl;
    int BOL_sum = 0;
    for (int i = 1; i < Y_DIM + 1; i++) {
        for (int j = 1; j < X_DIM + 1; j++) {

            auto val = std::bitset<32>(BOL_grid[i * (X_DIM + 2) + j]);
            std::cout << val << ' ';
            BOL_sum += val.count();
        }
        std::cout << std::endl;
    }

    std::cout << BOL_sum << std::endl;

    std::cout << "GOL" << std::endl;
    int total = 0;
    for (uint i = 1; i <= dim; i++) {
        for (uint j = 1; j <= dim; j++) {
            printf("%d", GOL_grid[i * (dim + 2) + j]);
            total += GOL_grid[i * (dim + 2) + j];
        }
        printf("\n");
    }

    printf("Total Alive: %d\n", total);
}