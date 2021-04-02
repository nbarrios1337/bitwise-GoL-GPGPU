#include "../nv/compute.cu"

int i, j, iter;
// Linear game grid dimension
int dim = 1024;
// Number of game iterations
int maxIter = 1024;

// Allocate rectangular grid of 1024 + 2 rows by 32 + 2 columns
int **grid = (int **)malloc(sizeof(int *) * (dim + 2));
for (i = 0; i < dim + 2; i++)
  grid[i] = (int *)malloc(sizeof(int *) * ((dim / 32) + 2));

// Allocate newGrid
int **newGrid = (int **)malloc(sizeof(int *) * (dim + 2));
for (i = 0; i < dim + 2; i++)
  newGrid[i] = (int *)malloc(sizeof(int *) * ((dim / 32) + 2));

// Main game loop
for (iter = 0; iter < maxIter; iter++) {
  // Left-Right columns
  for (i = 1; i <= dim; i++) {
    grid[i][0] = grid[i][dim]; // Copy last real column to left ghost column
    grid[i][dim + 1] =
        grid[i][1]; // Copy first real column to right ghost column
  }
  // Top-Bottom rows
  for (j = 0; j <= dim + 1;
       j++) { // I’m pretty sure j=1; j <= dim would be fine too?
    grid[0][j] = grid[dim][j];     // Copy last real row to top ghost row
    grid[dim + 1][j] = grid[1][j]; // Copy first real row to bottom ghost row
  }

  // Now we loop over all cells and determine their fate
  for (i = 1; i <= dim; i++) {
    for (j = 1; j <= (dim / 32); j++) {
      // Get the number of neighbors for a given grid point
      uint64_t top = grid[i - 1][j] uint64_t center = grid[i][j];
      uint64_t bottom = grid[i + 1][j];
      uint32_t *bitsets = NULL; // these should be alloc-ed before func calls
      uint32_t *out = NULL;
      get_bitsets(bitsets, top, center, bottom);
      *out = bitwise_sum63(bitsets);
      uint32_t numNeighbors = *out;
      newGrid[i][j] = *out;
    }
  }

  // Done with one step so we swap our grids and iterate again
  int **tmpGrid = grid;
  grid = newGrid;
  newGrid = tmpGrid;
} // End main game loop
