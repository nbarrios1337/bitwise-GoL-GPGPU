#include "bitwise.hpp"
#include "compute.hpp"

int main() {
    /* Copied form bitwise.cpp */
    // initialization
    bitset<SIZE + 2> top = bitset<SIZE + 2>(rand());
    bitset<SIZE + 2> middle = bitset<SIZE + 2>(rand());
    bitset<SIZE + 2> bottom = bitset<SIZE + 2>(rand());

    // test using naive impl

    // get old middle
    bitset<SIZE> test;
    for (int i = 1; i < SIZE + 1; i++) {
        test[i - 1] = middle[i];
    }

#ifdef DEBUG
    std::cout << "Start:" << std::endl;
    std::cout << "\t       " << top << std::endl;
    std::cout << "\t       " << middle << std::endl;
    std::cout << "\t       " << bottom << std::endl;

    std::cout << "Center\t\t" << test << std::endl;
#endif

    std::vector<int> neighbors;
    for (int i = SIZE; i > 0; i--) {
        int n = top[i + 1] + top[i] + top[i - 1] + middle[i + 1] +
                middle[i - 1] + bottom[i + 1] + bottom[i] + bottom[i - 1];
        neighbors.push_back(n);
        // std::cout << i << " Looking at" << std::endl;
        // std::cout << top[i+1] << top[i] << top[i-1] << std::endl;
        // std::cout << middle[i+1] << middle[i] << middle[i-1] << std::endl;
        // std::cout << bottom[i+1] << bottom[i] << bottom[i-1] << std::endl;
        // std::cout << n << std::endl;
        if (test[i - 1] == LIVE) {
            if (n < 2 || n > 3) {
                test[i - 1] = DEAD;
            }
        } else {
            if (n == 3) {
                test[i - 1] = LIVE;
            }
        }
    }

#ifdef DEBUG
    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;

    std::cout << "Center now\t" << test << std::endl;
    std::cout << std::endl;
#endif

    // get algo output
    uint32_t out = wrapper(top.to_ullong(), middle.to_ullong(), bottom.to_ullong());

#ifdef DEBUG
    std::cout << std::endl;

    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;
#endif

    std::cout << "computeState\t" << bitset<SIZE>(out) << std::endl;
    std::cout << "NaiveCompare\t" << test << std::endl;

    return bitset<SIZE>(out) != test;
}
