#include "compute.hpp"
#include <bitset>
#include <iostream>
#include <random>

#define SIZE 1 << 10
#define ITERATIONS 100

// (SIZE / 32) by SIZE elements
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
#ifdef DEBUG
            g[i * (X_DIM + 2) + j] = -i;
#else
            g[i * (X_DIM + 2) + j] = rand();
#endif
        }
    }
}

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

    uint64_t bottom = uint64_t(Y_DIM + 1) * uint64_t(X_DIM + 2) + index;
    uint64_t top = 0 * uint64_t(X_DIM + 2) + index;

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

    int gi_top = (iy - 1) * (X_DIM + 2) + ix;
    int gi_center = iy * (X_DIM + 2) + ix;
    int gi_bottom = (iy + 1) * (X_DIM + 2) + ix;

#ifdef DEBUG
    // printf("index: %d (%d * %d, %d)\n", index, iy, (X_DIM + 2), ix);
    printf("[simulate] Blk: (%d,%d) Thread: (%d,%d) -> Row/Col = "
           "(%d,%d)\tindex: %d (%d * "
           "%d, %d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, ix, iy, index, iy,
           (X_DIM + 2), ix);

    // g[index] = 255;
#else
    // Each block is called with 32 vertical threads
    // where each thread can store its respective datum
    // and its adjacent integers.
    // Buffer necessary for edges
    __shared__ uint32_t shared_data[NUM_THREADS + 2][3];
    uint32_t s_index = threadIdx.y + 1;

    // Get surrounding ints and then compute
    if (threadIdx.y == 0) {
        shared_data[0][0] = g[gi_top - 1];
        shared_data[0][1] = g[gi_top];
        shared_data[0][2] = g[gi_top + 1];
    }

    if (threadIdx.y == NUM_THREADS - 1) {
        shared_data[NUM_THREADS + 1][0] = g[gi_bottom - 1];
        shared_data[NUM_THREADS + 1][1] = g[gi_bottom];
        shared_data[NUM_THREADS + 1][2] = g[gi_bottom + 1];
    }

    shared_data[s_index][0] = g[gi_center - 1]; // left
    shared_data[s_index][1] = g[gi_center];     // center
    shared_data[s_index][2] = g[gi_center + 1]; // right

    __syncthreads();

/*     printf("[simulate] Blk: (%d,%d) Thrd: (%d,%d) -> shared i = %d (after "
           "shared init)\n\t"
           "Shared [%d]\n\tGlobal [%d]\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.y + 1,
           shared_data[threadIdx.y][1], g[gi_center]); */

    // make space for the LSB from the next number over's MSB
    uint64_t top_num = uint64_t(shared_data[s_index - 1][1]) << 1;
    uint64_t center_num = uint64_t(shared_data[s_index][1]) << 1;
    uint64_t bottom_num = uint64_t(shared_data[s_index + 1][1]) << 1;

    // since we shift in a 0, xor will set the LSB to
    // the bit from the next's MSB. See XOR truth table
    top_num ^= getBit(shared_data[s_index - 1][2], 31);
    center_num ^= getBit(shared_data[s_index][2], 31);
    bottom_num ^= getBit(shared_data[s_index + 1][2], 31);

    // explicitly specify type in order to avoid shifting out all bits
    top_num ^= getBit<uint64_t>(shared_data[s_index - 1][0], 0) << 32;
    center_num ^= getBit<uint64_t>(shared_data[s_index][0], 0) << 32;
    bottom_num ^= getBit<uint64_t>(shared_data[s_index + 1][0], 0) << 32;

    shared_data[s_index][1] = compute63(top_num, center_num, bottom_num);
    // could write directly to global...
    __syncthreads();

/*     printf("[simulate] Blk: (%d,%d) Thrd: (%d,%d) -> (after compute)\n\t"
           "Shared [%d]\n\tGlobal [%d]\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           shared_data[s_index][1], g[gi_center]); */

    g[gi_center] = shared_data[s_index][1];
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
    // Remove when benchmarking
    srand(1985);
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

    // See
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint64_t total_bytes =
        uint64_t(X_DIM + 2) * uint64_t(Y_DIM + 2) * sizeof(uint32_t);
    cudaMallocManaged(&grid, total_bytes);
    cudaMallocManaged(&tmpGrid, total_bytes);

    // init on host
    init_grid(grid);

    std::cout << "Before" << std::endl;
    for (int i = 1; i < Y_DIM + 1; i++) {
        for (int j = 1; j < X_DIM + 1; j++) {
            auto val = std::bitset<32>(grid[i * (X_DIM + 2) + j]);
            std::cout << val << ' ';
        }
        std::cout << std::endl;
    }

    // See https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(grid, total_bytes, device);

    // Adapted from ORNL
    dim3 block_size(1, NUM_THREADS);
    dim3 grid_size(X_DIM, X_DIM);

    dim3 grid_size_gCols(X_DIM);
    dim3 block_size_gCols(NUM_THREADS);

    dim3 grid_size_gRows(1);
    dim3 block_size_gRows(X_DIM + 2);

    cudaEventRecord(start);

#ifdef DEBUG
    std::cout << "block_size: " << block_size << std::endl;
    std::cout << "grid_size: " << grid_size << std::endl;
#endif

    for (int i = 0; i < ITERATIONS; i++) {
#ifdef DEBUG
        std::cout << "Calling ghost_columns<<<" << grid_size_gCols << ", "
                  << block_size_gCols << ">>>" << std::endl;
#endif
        ghost_columns<<<grid_size_gCols, block_size_gCols>>>(grid);

#ifdef DEBUG
        std::cout << "Calling ghost_rows<<<" << grid_size_gRows << ", "
                  << block_size_gRows << ">>>" << std::endl;
#endif
        ghost_rows<<<grid_size_gRows, block_size_gRows>>>(grid);

#ifdef DEBUG
        std::cout << "Calling simulate<<<" << grid_size << ", " << block_size
                  << ">>>" << std::endl;
#endif

        simulate<<<grid_size, block_size>>>(grid);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // CALL THIS BEFORE ANY DEVICE -> HOST MEM ACCESS
    cudaDeviceSynchronize();

#ifdef DEBUG
    // Prints the border cells
    int count = 0;
    for (int i = 0; i < Y_DIM + 2; i++) {
        for (int j = 0; j < X_DIM + 2; j++) {
            auto val = std::bitset<32>(grid[i * (X_DIM + 2) + j]);
            std::cout << val.to_ullong() << '\t';
            count += 32;
        }
        std::cout << std::endl;
    }
#else
    int sum = 0;
    std::cout << "After" << std::endl;
    for (int i = 1; i < Y_DIM + 1; i++) {
        for (int j = 1; j < X_DIM + 1; j++) {
            auto val = std::bitset<32>(grid[i * (X_DIM + 2) + j]);
            std::cout << val << ' ';
            sum += val.count();
        }
        std::cout << std::endl;
    }
#endif

    // std::cout << "Columns" << std::endl;
    // // column checking
    // for (int k = 0; k < Y_DIM + 2; k++) {
    //     auto a = std::bitset<32>(grid[k * (X_DIM + 2)]);
    //     auto b = std::bitset<32>(grid[k * (X_DIM + 2) + X_DIM]);
    //     std::cout << a << " vs " << b << (a == b ? ": match" : ": no match")
    //     << std::endl;
    // }

    // std::cout << "Rows" << std::endl;
    // // row checking
    //     for (int k = 0; k < X_DIM + 2; k++) {
    //     auto a = std::bitset<32>(grid[k + (X_DIM + 2)]);
    //     auto b = std::bitset<32>(grid[k + (X_DIM + 2) * (Y_DIM+1)]);
    //     std::cout << a << " vs " << b << (a == b ? ": match" : ": no match")
    //     << std::endl;
    // }

#ifdef DEBUG
    std::cout << "Sizes " << (count == TOTAL_ELEMENTS ? "" : "not ")
              << "match: ";
    std::cout << count << " counted, " << TOTAL_ELEMENTS << " total"
              << std::endl;
#else
    std::cout << sum << std::endl;
#endif

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "ElapsedTime: " << ms << " ms" << std::endl;

    cudaFree(grid);
    cudaFree(tmpGrid);

    return 0;
}
