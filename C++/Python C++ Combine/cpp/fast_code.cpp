#include <iostream>
#include <pybind11/pybind11.h>

void slowFunction() {

    long long total = 0;

    for (long long i = 0; i < 100000000; i++) {
        total += i;
    }

    std::cout << total << '\n';
}

PYBIND11_MODULE(fast_code, m) {

    m.def(
        "slow_function",
        &slowFunction
    );
}