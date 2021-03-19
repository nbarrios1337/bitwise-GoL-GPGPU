#include <bitset>
using std::bitset;

#include <iostream>
#include <vector>

#define SIZE 8
#define LIVE 1
#define DEAD 0

bitset<SIZE> computeNextState(bitset<SIZE + 2> top, bitset<SIZE + 2> middle,
                              bitset<SIZE + 2> bottom) {
    std::cout << "Computing..." << std::endl;
    // masks init
    // 111...100
    bitset<SIZE + 2> notTwoLSB = bitset<SIZE + 2>().set();
    notTwoLSB.reset(1);
    notTwoLSB.reset(0);

    std::cout << "LSB Mask\t" << notTwoLSB << std::endl;

    // 001...111
    bitset<SIZE + 2> notTwoMSB = bitset<SIZE + 2>().set();
    notTwoMSB.reset(SIZE + 1);
    notTwoMSB.reset(SIZE);

    std::cout << "MSB Mask\t" << notTwoMSB << std::endl;
    std::cout << std::endl;

    // get 8 bitsets
    std::cout << "old\t       " << top << std::endl;
    bitset<SIZE> upperleft = bitset<SIZE>((top >> 2).to_ullong());
    std::cout << "Upper Left     " << upperleft << std::endl;

    bitset<SIZE> upper = bitset<SIZE>((top.reset(SIZE + 1) >> 1).to_ullong());
    std::cout << "Upper Mid\t" << upper << std::endl;

    bitset<SIZE> upperRight = bitset<SIZE>((top & notTwoMSB).to_ullong());
    std::cout << "Upper Right\t " << upperRight << std::endl;
    std::cout << std::endl;

    std::cout << "old\t       " << middle << std::endl;
    bitset<SIZE> midLeft = bitset<SIZE>((middle >> 2).to_ullong());
    std::cout << "Middle Left    " << midLeft << std::endl;

    bitset<SIZE> mid = bitset<SIZE>((middle.reset(SIZE + 1) >> 1).to_ullong());
    std::cout << "Middle Middle\t" << mid << std::endl;

    bitset<SIZE> midRight = bitset<SIZE>((middle & notTwoMSB).to_ullong());
    std::cout << "Middle Right\t " << midRight << std::endl;
    std::cout << std::endl;

    std::cout << "old\t       " << bottom << std::endl;
    bitset<SIZE> lowerleft = bitset<SIZE>((bottom >> 2).to_ullong());
    std::cout << "Lower Left     " << lowerleft << std::endl;

    bitset<SIZE> lower = bitset<SIZE>((bottom.reset(SIZE + 1) >> 1).to_ullong());
    std::cout << "Lower Mid\t" << lower << std::endl;

    bitset<SIZE> lowerRight = bitset<SIZE>((bottom & notTwoMSB).to_ullong());
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

    std::vector<int> neighbors;
    for(int i = SIZE-1; i >= 0; i--){
        int s = upperleft[i] + upper[i] + upperRight[i] +
                midLeft[i] + midRight[i] + 
                lowerleft[i] + lower[i] + lowerRight[i];
        neighbors.push_back(s);
    }

    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;



    return bitset<SIZE>();
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

    std::cout << "Start:" << std::endl;
    std::cout << "\t       " << top << std::endl;
    std::cout << "\t       " << middle << std::endl;
    std::cout << "\t       " << bottom << std::endl;

    std::cout << "Center\t\t" << test << std::endl;

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

    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;

    std::cout << "Center now\t" << test << std::endl;
    std::cout << std::endl;

    // get algo output
    bitset<SIZE> next = computeNextState(top, middle, bottom);
    std::cout << std::endl;

    std::cout << "Neighbors\t";
    for (auto i : neighbors) {
        std::cout << i;
    }
    std::cout << std::endl;
    
    std::cout << "computeState\t" << next << std::endl;

    return 0;
}
