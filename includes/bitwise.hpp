#include <bitset>
using std::bitset;

#include <iostream>
#include <vector>

#define SIZE 8
#define LIVE 1
#define DEAD 0

bitset<SIZE> computeNextState(bitset<SIZE + 2> top, bitset<SIZE + 2> middle,
                              bitset<SIZE + 2> bottom);

template <typename T> inline T getBit(T num, short pos) {
    return (num & (1 << pos)) >> pos;
}

/* mask = 1 << pos
 * set = num | mask
 */

template <typename T> inline T setBit(T num, short pos) {
    return num | (1 << pos);
}

/* mask = ~(1 << pos)
 * unset = num & mask
 */

template <typename T> inline T unsetBit(T num, short pos) {
    return num & (~(1 << pos));
}