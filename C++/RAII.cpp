#include <iostream>

class Resource {
public:
    Resource() {
        std::cout << "Resource acquired\n";
    }

    ~Resource() {
        std::cout << "Resource released\n";
    }
};

void example() {
    Resource resource;

    std::cout << "Using resource\n";

} // destructor runs automatically here

int main() {

    example();

    std::cout << "Function finished\n";

    return 0;
}
