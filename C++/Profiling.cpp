#include <iostream>
#include <chrono>

void slowFunction() {

    long long total = 0;

    for (long long i = 0; i < 100000000; i++) {
        total += i;
    }

    std::cout << total << '\n';
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    slowFunction();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        );

    std::cout << "Time: "
              << duration.count()
              << " ms\n";

    return 0;
}

