#include <iostream>

void stackExample() {
    int stackNumber = 10;

    std::cout << "Stack value: " << stackNumber << '\n';
    std::cout << "Stack address: " << &stackNumber << '\n';

    // stackNumber is automatically destroyed
    // when this function finishes
}

void heapExample() {
    int* heapNumber = new int(20);

    std::cout << "Heap value: " << *heapNumber << '\n';
    std::cout << "Heap address: " << heapNumber << '\n';

    delete heapNumber;
    heapNumber = nullptr;
}

int main() {

    stackExample();

    std::cout << '\n';

    heapExample();

    return 0;
}