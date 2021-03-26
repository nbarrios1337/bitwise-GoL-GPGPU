#include <bitset>
#include <iostream>
#include <random>

#define GRID_DIM 1024

// ORNL Code uses a 1D grid
int dim = 32;

int *generate() {
  int *grid = (int *)calloc((dim + 2) * (dim + 2), sizeof(int));
  for (int i = 1; i <= dim; i++) {
    for (int j = 1; j <= dim; j++) {
      grid[i * (dim + 2) + j] = rand() % 2;
    }
  }
  return grid;
}

int main() {
  int *grid = generate();
  for (int i = 1; i <= dim; i++) {
    for (int j = 1; j <= dim; j++) {
      std::cout << grid[i * (dim + 2) + j];
    }
    std::cout << std::endl;
  }

  free(grid);
  return 0;
}
