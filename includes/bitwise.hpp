#include <bitset>
using std::bitset;

#include <iostream>
#include <vector>
#include "utils.hpp"

#define SIZE 8
#define LIVE 1
#define DEAD 0

bitset<SIZE> computeNextState(bitset<SIZE + 2> top, bitset<SIZE + 2> middle,
                              bitset<SIZE + 2> bottom);
