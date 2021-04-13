#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <bitset>
#define SRAND_VALUE 1985

// Linear game grid dimension
int x_dim = 1;
int y_dim = x_dim * sizeof(int);
// Number of game iterations
int maxIter = 1;

/* mask = 1 << pos;
 * masked = num & mask;
 * bit = masked >> pos
 */
inline uint32_t getBit(uint32_t num, int pos) {
  return (num & (1 << pos)) >> pos;
}

/* mask = 1 << pos
 * set = num ^ mask
 */

inline uint32_t setBit(uint32_t num, int pos) { return num | (1 << pos); }

/* mask = ~(1 << pos)
 * unset = num ^ mask
 */

inline uint32_t unsetBit(uint32_t num, int pos) { return num & (~(1 << pos)); }

void get_bitsets(uint32_t *bitsets, uint64_t top, uint64_t center,
                 uint64_t bottom) {
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

  // Now I could do the assignments directly,
  // but I want to be able to read my code
  bitsets[0] = upper_left;
  bitsets[1] = upper;
  bitsets[2] = upper_right;

  bitsets[3] = middle_left;
  bitsets[4] = middle;
  bitsets[5] = middle_right;

  bitsets[6] = lower_left;
  bitsets[7] = lower;
  bitsets[8] = lower_right;
}

/*
 * Index|   Bitset
 * ---------------------
 * 0 a  |   upper_left
 * 1 b  |   upper
 * 2 c  |   upper_right
 * 3 d  |   middle_left
 * 4    |   middle
 * 5 e  |   middle_right
 * 6 f  |   lower_left
 * 7 g  |   lower
 * 8 h  |   lower_right
 *
 * Visually:
 * 0 a      1 b     2 c
 * 3 d      4       5 e
 * 6 f      7 g     8 h
 */

// See Tsuda (http://vivi.dyndns.org/tech/games/LifeGame.html)
uint32_t bitwise_sum63(uint32_t *bs) {
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

  // upper_left + upper addition (4 bitwise ops)
  s2 = bs[0] & bs[1];
  s1 = bs[0] ^ bs[1];
  s0 = ~(bs[0] | bs[1]);

  // upper_right addition (9 bitwise ops)
  uint32_t nc = ~bs[2];
  s3 = s2 & bs[2];
  s2 = (s2 & nc) | (s1 & bs[2]);
  s1 = (s1 & nc) | (s0 & bs[2]);
  s0 &= nc;

  // middle_left addition (11 b-ops)
  uint32_t nd = ~bs[3];
  s3 = (s3 & nd) | (s2 & bs[3]);
  s2 = (s2 & nd) | (s1 & bs[3]);
  s1 = (s1 & nd) | (s0 & bs[3]);
  s0 &= nd;

  // middle_right add (11 b-ops)
  uint32_t ne = ~bs[5];
  s3 = (s3 & ne) | (s2 & bs[5]);
  s2 = (s2 & ne) | (s1 & bs[5]);
  s1 = (s1 & ne) | (s0 & bs[5]);
  s0 &= ne;

  // lower_left add (11 b-ops)
  uint32_t nf = ~bs[6];
  s3 = (s3 & nf) | (s2 & bs[6]);
  s2 = (s2 & nf) | (s1 & bs[6]);
  s1 = (s1 & nf) | (s0 & bs[6]);
  s0 &= nf;

  // lower add (10 b-ops)
  uint32_t ng = ~bs[7];
  s3 = (s3 & ng) | (s2 & bs[7]);
  s2 = (s2 & ng) | (s1 & bs[7]);
  s1 = (s1 & ng) | (s0 & bs[7]);

  // lower_right add (7 b-ops)
  uint32_t nh = ~bs[8];
  s3 = (s3 & nh) | (s2 & bs[8]);
  s2 = (s2 & nh) | (s1 & bs[8]);

  return s3 | (bs[4] & s2);
}

/*
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
x x x x
*/
int main() {
  // Allocate rectangular grid of 1024 + 2 rows by 32 + 2 columns
  srand(SRAND_VALUE);
  int *grid = (int *)calloc((y_dim + 2) * (x_dim + 2), sizeof(int));
  int *newGrid = (int *)calloc((y_dim + 2) * (x_dim + 2), sizeof(int));
  for (int i = 1; i <= y_dim; i++) {
    for (int j = 1; j <= x_dim; j++) {
      int randomBits =
          rand(); // Advanced random bit generation (I'm sorry Dr. Zola LOL.)
      grid[i * (x_dim + 2) + j] = randomBits;
      newGrid[i * (x_dim + 2) + j] = randomBits;
    }
  }

  // Main game loop
  for (int iter = 0; iter < maxIter; iter++) {
    // Left-Right columns
    for (int i = 1; i <= y_dim; i++) {
      grid[i * (x_dim + 2)] =
          grid[i * (x_dim + 2) +
               x_dim]; // Copy last real column to left ghost column
      grid[i * (x_dim + 2) + x_dim] =
          grid[i * (x_dim + 2) +
               1]; // Copy first real column to right ghost column
    }
    // Top-Bottom rows
    for (int j = 0; j <= x_dim + 1; j++) {
      grid[(0) * (x_dim + 2) + j] =
          grid[(y_dim - 1) * (x_dim + 2) +
               j]; // Copy last real row to top ghost row
      grid[(y_dim) * (x_dim + 2) + j] =
          grid[(1) * (x_dim + 2) +
               j]; // Copy first real row to bottom ghost row
    }

    // Now we loop over all cells and determine their fate
    for (int i = 1; i <= y_dim; i++) {
      for (int j = 1; j <= x_dim; j++) {
        // Get the number of neighbors for a given grid point
        uint64_t top = grid[(i - 1) * (x_dim + 2) + j];
        uint64_t center = grid[i * (x_dim + 2) + j];
        uint64_t bottom = grid[(i + 1) * (x_dim + 2) + j];
        uint32_t *bitsets = (uint32_t *)malloc(sizeof(uint32_t *)); //put this on the stack not heap
        uint32_t *out = (uint32_t *)malloc(sizeof(uint32_t *)); //also should be on the stack
        get_bitsets(bitsets, top, center, bottom);
        *out = bitwise_sum63(bitsets);
        newGrid[i * (x_dim + 2) + j] = *out;
        free(bitsets);
        //free(out); //Having trouble freeing this memory
      }
    }

    // Done with one step so we swap our grids and iterate again
    int *tmpGrid = grid;
    grid = newGrid;
    newGrid = tmpGrid;
  } // End main game loop
  int count = 0;
  for (int i = 1; i <= y_dim; i++) {
    for (int j = 1; j <= x_dim; j++) {
      std::bitset<32> cells (newGrid[i * (x_dim + 2) + j]);
      count += cells.count();
      std::cout << cells << std::endl;
    }
  }
  std::cout << count << std::endl;
  free(grid);
  free(newGrid);
  return 0;
}
