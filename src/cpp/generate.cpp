#include <bitset>
#include <cstdio>
#include <iostream>
#include <random>

#define GRID_DIM 1024

// ORNL Code uses a 1D grid
int x_dim = 4;
int y_dim = x_dim * sizeof(int);

int *generate() {
  int *grid = (int *)calloc((y_dim + 2) * (x_dim + 2), sizeof(int));
  // int *grid = new int[(y_dim + 2) * (x_dim + 2)];
  // printf("%p\n", grid);
  //  i \in [1, dim+1)
  for (int i = 1; i <= y_dim; i++) {
    for (int j = 1; j <= x_dim; j++) {
      grid[i * (x_dim + 2) + j] = 1;
    }
  }
  return grid;
}

int main() {
  int *grid = generate();
  for (int i = 1; i <= y_dim; i++) {
    for (int j = 1; j <= x_dim; j++) {
      std::cout << grid[i * (x_dim + 2) + j];
    }
    std::cout << std::endl;
  }

  free(grid);
  // delete [] (grid);
  return 0;
}
