#include <bitset>
using std::bitset;

#include <iostream>
#include <vector>

#define SIZE 16
#define LIVE 1
#define DEAD 0

bitset<SIZE> computeNextState(bitset<SIZE + 2> top, bitset<SIZE + 2> middle,
                              bitset<SIZE + 2> bottom) {
    // masks init
    // 111...100
    bitset<SIZE + 2> notTwoLSB = bitset<SIZE + 2>().set();
    notTwoLSB.reset(1);
    notTwoLSB.reset(0);

    #ifdef DEBUG
    std::cout << "LSB Mask\t" << notTwoLSB << std::endl;
    #endif

    // 001...111
    bitset<SIZE + 2> notTwoMSB = bitset<SIZE + 2>().set();
    notTwoMSB.reset(SIZE + 1);
    notTwoMSB.reset(SIZE);

    #ifdef DEBUG
    std::cout << "MSB Mask\t" << notTwoMSB << std::endl;
    std::cout << std::endl;
    #endif

    // get 8 bitsets
    bitset<SIZE> upperleft = bitset<SIZE>((top >> 2).to_ullong());
    #ifdef DEBUG
    std::cout << "old\t       " << top << std::endl;
    std::cout << "Upper Left     " << upperleft << std::endl;
    #endif

    bitset<SIZE> upper = bitset<SIZE>((top.reset(SIZE + 1) >> 1).to_ullong());
    #ifdef DEBUG
    std::cout << "Upper Mid\t" << upper << std::endl;
    #endif

    bitset<SIZE> upperRight = bitset<SIZE>((top & notTwoMSB).to_ullong());
    #ifdef DEBUG
    std::cout << "Upper Right\t " << upperRight << std::endl;
    std::cout << std::endl;
    #endif

    bitset<SIZE> midLeft = bitset<SIZE>((middle >> 2).to_ullong());
    #ifdef DEBUG
    std::cout << "old\t       " << middle << std::endl;
    std::cout << "Middle Left    " << midLeft << std::endl;
    #endif

    bitset<SIZE> mid = bitset<SIZE>((middle.reset(SIZE + 1) >> 1).to_ullong());
    #ifdef DEBUG
    std::cout << "Middle Middle\t" << mid << std::endl;
    #endif

    bitset<SIZE> midRight = bitset<SIZE>((middle & notTwoMSB).to_ullong());
    #ifdef DEBUG
    std::cout << "Middle Right\t " << midRight << std::endl;
    std::cout << std::endl;
    #endif

    bitset<SIZE> lowerleft = bitset<SIZE>((bottom >> 2).to_ullong());
    #ifdef DEBUG
    std::cout << "old\t       " << bottom << std::endl;
    std::cout << "Lower Left     " << lowerleft << std::endl;
    #endif

    bitset<SIZE> lower = bitset<SIZE>((bottom.reset(SIZE + 1) >> 1).to_ullong());
    #ifdef DEBUG
    std::cout << "Lower Mid\t" << lower << std::endl;
    #endif

    bitset<SIZE> lowerRight = bitset<SIZE>((bottom & notTwoMSB).to_ullong());
    #ifdef DebUG
    std::cout << "Lower Right\t " << lowerRight << std::endl;
    std::cout << std::endl;

    // neighbor sum
    std::cout << "\t" << upperleft;
    std::cout << " " << upper;
    std::cout << " " << upperRight << std::endl;
    std::cout << "\t" << midLeft;
    std::cout << " " << mid;
    std::cout << " " << midRight << std::endl;
    std::cout << "\t" << lowerleft;
    std::cout << " " << lower;
    std::cout << " " << lowerRight << std::endl;
    #endif

    std::vector<int> neighbors;
    for(int i = SIZE-1; i >= 0; i--){
        int s = upperleft[i] + upper[i] + upperRight[i] +
                midLeft[i] + midRight[i] + 
                lowerleft[i] + lower[i] + lowerRight[i];
        neighbors.push_back(s);
    }

    #ifdef DEBUG
    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;
    #endif

    // s0 to s8, all set to 000...000
    bitset<SIZE> sumBits[] = {0, 0, 0, 0, 0, 0, 0, 0};

    // set the i-th MSB of the bitset for the value read
    // i.e. with 3_5_553345, sets the 2nd bit of s5
    // since indexing is 0-7, we need to offset indexing sumBits
    for(int i = 0; i < SIZE; i++){
        #ifdef DEBUG
        std::cout << "Looking at the " << i << "th sum (" << neighbors[i] << ")" << std::endl;
        #endif
        // no way to have more than 8 neighbors
        if (neighbors[i] < 8){
            sumBits[neighbors[i]-1].set(SIZE-1-i);
        }
    }

    #ifdef DEBUG
    std::cout << "N:  ";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;

    for (int i = 0; i < 8; i++){
        std::cout << "s" << i+1 << ": " << sumBits[i] << std::endl;
    }
    #endif

    // (center & s2) | s3
    return (mid & sumBits[1]) | sumBits[2];
}

int main() {
    // initialization
    // srand(time(NULL));
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
        int n = top[i + 1] + top[i] + top[i - 1] + 
                middle[i + 1] + middle[i - 1] + 
                bottom[i + 1] + bottom[i] + bottom[i - 1];
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
    bitset<SIZE> next = computeNextState(top, middle, bottom);

    #ifdef DEBUG
    std::cout << std::endl;

    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;
    #endif
    
    std::cout << "computeState\t" << next << std::endl;
    std::cout << "NaiveCompare\t" << test << std::endl;

    return next != test;
}
