#include "utils.hpp"

std::ostream& operator<<(std::ostream&os, const dim3 d) {
    os << '{' << d.x << ' ' << d.y << ' ' << d.z << '}';
    return os;
} 