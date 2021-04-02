#include <cstdlib>
#include <cstdint>

int i, j, iter;
// Linear game grid dimension
int dim = 1024;
// Number of game iterations
int maxIter = 1024;

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

inline uint32_t setBit(uint32_t num, int pos) {
    return num | (1 << pos);
}

/* mask = ~(1 << pos)
 * unset = num ^ mask
 */

inline uint32_t unsetBit(uint32_t num, int pos) {
    return num & (~(1 << pos));
}

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

int main() {
    // Allocate rectangular grid of 1024 + 2 rows by 32 + 2 columns
    int **grid = (int **)malloc(sizeof(int *) * (dim + 2));
    for (i = 0; i < dim + 2; i++) {
        grid[i] = (int *)malloc(sizeof(int *) * ((dim / 32) + 2));
    }

    // Allocate newGrid
    int **newGrid = (int **)malloc(sizeof(int *) * (dim + 2));
    for (i = 0; i < dim + 2; i++) {
        newGrid[i] = (int *)malloc(sizeof(int *) * ((dim / 32) + 2));
    }

    // Main game loop
    for (iter = 0; iter < maxIter; iter++) {
        // Left-Right columns
        for (i = 1; i <= dim; i++) {
            grid[i][0] =
                grid[i][dim / 32]; // Copy last real column to left ghost column
            grid[i][(dim / 32) + 1] =
                grid[i][1]; // Copy first real column to right ghost column
        }
        // Top-Bottom rows
        for (j = 0; j <= (dim / 32) + 1;
             j++) { // I’m pretty sure j=1; j <= dim would be fine too?
            grid[0][j] = grid[dim][j]; // Copy last real row to top ghost row
            grid[dim + 1][j] =
                grid[1][j]; // Copy first real row to bottom ghost row
        }

        // Now we loop over all cells and determine their fate
        for (i = 1; i <= dim; i++) {
            for (j = 1; j <= (dim / 32); j++) {
                // Get the number of neighbors for a given grid point
                uint64_t top = grid[i - 1][j];
                uint64_t center = grid[i][j];
                uint64_t bottom = grid[i + 1][j];
                uint32_t *bitsets =
                    NULL; // these should be alloc-ed before func calls
                uint32_t *out = NULL;
                get_bitsets(bitsets, top, center, bottom);
                *out = bitwise_sum63(bitsets);
                newGrid[i][j] = *out;
            }
        }

        // Done with one step so we swap our grids and iterate again
        int **tmpGrid = grid;
        grid = newGrid;
        newGrid = tmpGrid;
    } // End main game loop
    return 0;
}
