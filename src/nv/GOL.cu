#include <stdio.h>
#include <stdlib.h>

#define SRAND_VALUE 1985
#define BLOCK_SIZE 16

__global__ void ghostRows(int dim, int *grid)
{
    // We want id ∈ [1,dim]
    int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (id <= dim)
    {
        //Copy first real row to bottom ghost row
        grid[(dim+2)*(dim+1)+id] = grid[(dim+2)+id];
        //Copy last real row to top ghost row
        grid[id] = grid[(dim+2)*dim + id];
    }
}

__global__ void ghostCols(int dim, int *grid)
{
    // We want id ∈ [0,dim+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id <= dim+1)
    {
        //Copy first real column to right most ghost column
        grid[id*(dim+2)+dim+1] = grid[id*(dim+2)+1];
        //Copy last real column to left most ghost column
        grid[id*(dim+2)] = grid[id*(dim+2) + dim];
    }
}

__global__ void GOL(int dim, int *grid, int *newGrid)
{
    // We want id ∈ [1,dim]
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (dim+2) + ix;

    int numNeighbors;

    if (iy <= dim && ix <= dim) {

        // Get the number of neighbors for a given grid point
        numNeighbors = grid[id+(dim+2)] + grid[id-(dim+2)] //upper lower
                     + grid[id+1] + grid[id-1]             //right left
                     + grid[id+(dim+3)] + grid[id-(dim+3)] //diagonals
                     + grid[id-(dim+1)] + grid[id+(dim+1)];

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

__global__ void bitLifeKernelNoLookup(const ubyte* lifeData, uint worldDataWidth,
    uint worldHeight, uint bytesPerThread, ubyte* resultLifeData) {

  uint worldSize = (worldDataWidth * worldHeight);

  for (uint cellId = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * bytesPerThread;
      cellId < worldSize;
      cellId += blockDim.x * gridDim.x * bytesPerThread) {

    uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
    uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
    uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
    uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

    // Initialize data with previous byte and current byte.
    uint data0 = (uint)lifeData[x + yAbsUp] << 16;
    uint data1 = (uint)lifeData[x + yAbs] << 16;
    uint data2 = (uint)lifeData[x + yAbsDown] << 16;

    x = (x + 1) % worldDataWidth;
    data0 |= (uint)lifeData[x + yAbsUp] << 8;
    data1 |= (uint)lifeData[x + yAbs] << 8;
    data2 |= (uint)lifeData[x + yAbsDown] << 8;

    for (uint i = 0; i < bytesPerThread; ++i) {
      uint oldX = x;  // old x is referring to current center cell
      x = (x + 1) % worldDataWidth;
      data0 |= (uint)lifeData[x + yAbsUp];
      data1 |= (uint)lifeData[x + yAbs];
      data2 |= (uint)lifeData[x + yAbsDown];

      uint result = 0;
      for (uint j = 0; j < 8; ++j) {
        uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
        aliveCells >>= 14;
        aliveCells = (aliveCells & 0x3) + (aliveCells >> 2)
          + ((data0 >> 15) & 0x1u) + ((data2 >> 15) & 0x1u);

        result = result << 1
          | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

        data0 <<= 1;
        data1 <<= 1;
        data2 <<= 1;
      }

      resultLifeData[oldX + yAbs] = result;
    }
  }
}

int main(int argc, char* argv[])
{
    int i,j,iter;
    int* h_grid; //Grid on host
    ubyte* d_grid; //Grid on device
    ubyte* d_newGrid; //Second grid used on device only
    ubyte* d_tmpGrid; //tmp grid pointer used to switch between grid and newGrid

    uint dim = 1024; //Linear dimension of our grid - not counting ghost cells
    int maxIter = 1<<10; //Number of game steps

    size_t bytes = sizeof(int)*(dim+2)*(dim+2);//2 added for periodic boundary condition ghost cells
    // Allocate host Grid used for initial setup and read back from device
    h_grid = (int*)malloc(bytes);

    // Allocate device grids
    cudaMalloc(&d_grid, bytes);
    cudaMalloc(&d_newGrid, bytes);

    // Assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 1; i<=dim; i++) {
        for(j = 1; j<=dim; j++) {
            h_grid[i*(dim+2)+j] = rand() % 2;
        }
    }

    // Copy over initial game grid (Dim-1 threads)
    cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGrid = (int)ceil(dim/(float)BLOCK_SIZE);
    dim3 gridSize(linGrid,linGrid,1);

    dim3 cpyBlockSize(BLOCK_SIZE,1,1);
    dim3 cpyGridRowsGridSize((int)ceil(dim/(float)cpyBlockSize.x),1,1);
    dim3 cpyGridColsGridSize((int)ceil((dim+2)/(float)cpyBlockSize.x),1,1);

    // Main game loop
    for (iter = 0; iter<maxIter; iter++) {

        ghostRows<<<cpyGridRowsGridSize, cpyBlockSize>>>(dim, d_grid);
        ghostCols<<<cpyGridColsGridSize, cpyBlockSize>>>(dim, d_grid);
        //GOL<<<gridSize, blockSize>>>(dim, d_grid, d_newGrid);
        bitLifeKernelNoLookup<<<gridSize, blockSize>>>(d_grid, dim,
           dim, 4, d_newgrid);;
        // Swap our grids and iterate again
        d_tmpGrid = d_grid;
        d_grid = d_newGrid;
        d_newGrid = d_tmpGrid;
    }//iter loop

    // Copy back results and sum
    cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost);

    // Sum up alive cells and print results
    int total = 0;
    for (i = 1; i<=dim; i++) {
        for (j = 1; j<=dim; j++) {
            total += h_grid[i*(dim+2)+j];
        }
    }
    printf("Total Alive: %d\n", total);

    // Release memory
    cudaFree(d_grid);
    cudaFree(d_newGrid);
    free(h_grid);

    return 0;
}
