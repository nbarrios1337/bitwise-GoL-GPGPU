#include "compute.hpp"

__device__ void get_bitsets(uint32_t *bitsets, uint64_t top, uint64_t center,
                            uint64_t bottom) {
    // 111...100
    uint32_t notTwoLSB = (~(uint32_t)0) << 2;
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
__device__ uint32_t bitwise_sum63(uint32_t *bs) {
    uint32_t s0, s1, s2, s3;

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

__global__ void next_gen(uint32_t *out, uint32_t *bitsets, uint64_t top,
                         uint64_t center, uint64_t bottom) {
    get_bitsets(bitsets, top, center, bottom);
    *out = bitwise_sum63(bitsets);
}

uint32_t wrapper(uint64_t t, uint64_t m, uint64_t b) {
    uint32_t *bitsets, *out;

    cudaMallocManaged(&bitsets, 9 * sizeof(uint32_t));
    //cudaMemset(bitsets, 0, 9);
    cudaMallocManaged(&out, sizeof(uint32_t));

    next_gen<<<1, 1>>>(out, bitsets, t, m, b);

    cudaDeviceSynchronize();

    uint32_t retVal = *out;

    cudaFree(bitsets);
    cudaFree(out);

    return retVal;
}

// TODO Ideally we want the getting of bitsets and the computation of
// the bitwise sums to be in a single function to remove the need for
// array indexing