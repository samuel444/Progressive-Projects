#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <tuple>
#include <optional>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>

// =========================
// MAIN
// =========================

int main() {
    return 0;
}


// =========================
// VARIABLES
// =========================

int age = 22;
double price = 100.5;
float x = 1.5f;
bool active = true;
char letter = 'A';
std::string name = "Sam";


// =========================
// CONSTANTS
// =========================

const double rate = 0.05;

constexpr int days = 252;


// =========================
// AUTO
// =========================

auto number = 10;
auto decimal = 10.5;
auto text = std::string("hello");


// =========================
// OUTPUT
// =========================

std::cout << "Hello\n";

std::cout << price << '\n';

std::cerr << "Error\n";


// =========================
// INPUT
// =========================

int value;

std::cin >> value;


// =========================
// ARITHMETIC
// =========================

int a = 10;
int b = 3;

int add = a + b;
int subtract = a - b;
int multiply = a * b;
int divide = a / b;
int remainder = a % b;

a += 1;
a -= 1;
a *= 2;
a /= 2;

a++;
++a;

a--;
--a;


// =========================
// CASTING
// =========================

double result =
    static_cast<double>(a) / b;


// =========================
// COMPARISONS
// =========================

a == b;
a != b;
a > b;
a < b;
a >= b;
a <= b;


// =========================
// LOGICAL OPERATORS
// =========================

true && false;

true || false;

!true;


// =========================
// IF
// =========================

if (a > b) {
    std::cout << "A\n";
}
else if (a == b) {
    std::cout << "Equal\n";
}
else {
    std::cout << "B\n";
}


// =========================
// TERNARY
// =========================

int max_value =
    a > b ? a : b;


// =========================
// SWITCH
// =========================

int choice = 2;

switch (choice) {

    case 1:
        std::cout << "One\n";
        break;

    case 2:
        std::cout << "Two\n";
        break;

    default:
        std::cout << "Other\n";
}


// =========================
// FOR LOOP
// =========================

for (int i = 0; i < 10; ++i) {
    std::cout << i << '\n';
}


// =========================
// WHILE
// =========================

int i = 0;

while (i < 10) {
    ++i;
}


// =========================
// DO WHILE
// =========================

do {
    ++i;
}
while (i < 20);


// =========================
// BREAK / CONTINUE
// =========================

for (int i = 0; i < 10; ++i) {

    if (i == 3) {
        continue;
    }

    if (i == 8) {
        break;
    }
}


// =========================
// FUNCTIONS
// =========================

double add_numbers(
    double x,
    double y
) {
    return x + y;
}


// =========================
// VOID FUNCTION
// =========================

void print_value(double x) {
    std::cout << x << '\n';
}


// =========================
// DEFAULT ARGUMENT
// =========================

double multiply_value(
    double x,
    double multiplier = 2.0
) {
    return x * multiplier;
}


// =========================
// REFERENCES
// =========================

void change_value(int& x) {
    x = 100;
}


// =========================
// CONST REFERENCE
// =========================

void print_name(
    const std::string& name
) {
    std::cout << name << '\n';
}


// =========================
// POINTERS
// =========================

int number = 10;

int* pointer = &number;

std::cout << pointer << '\n';

std::cout << *pointer << '\n';

*pointer = 20;


// =========================
// NULL POINTER
// =========================

int* ptr = nullptr;

if (ptr == nullptr) {
}


// =========================
// ARRAY
// =========================

int raw_array[3] = {
    1,
    2,
    3
};


// =========================
// STD::ARRAY
// =========================

std::array<int, 3> arr = {
    1,
    2,
    3
};

arr[0];

arr.size();


// =========================
// VECTOR
// =========================

std::vector<double> values = {
    1.0,
    2.0,
    3.0
};

values.push_back(4.0);

values.pop_back();

values[0];

values.at(0);

values.size();

values.empty();

values.clear();

values.reserve(100);


// =========================
// RANGE LOOP
// =========================

for (double value : values) {
}


// =========================
// RANGE LOOP WITH REFERENCE
// =========================

for (double& value : values) {
}


// =========================
// RANGE LOOP WITH CONST REF
// =========================

for (const auto& value : values) {
}


// =========================
// STRING
// =========================

std::string ticker = "AAPL";

ticker.size();

ticker += "_US";

ticker[0];

ticker.substr(0, 2);


// =========================
// PAIR
// =========================

std::pair<int, double> pair = {
    1,
    2.5
};

pair.first;

pair.second;


// =========================
// STRUCTURED BINDING
// =========================

auto [first, second] = pair;


// =========================
// TUPLE
// =========================

std::tuple<int, double, std::string> data = {
    1,
    2.5,
    "AAPL"
};

std::get<0>(data);

std::get<1>(data);

std::get<2>(data);


// =========================
// MAP
// =========================

std::map<std::string, double> prices;

prices["AAPL"] = 200.0;

prices["MSFT"] = 400.0;

prices["AAPL"];


// =========================
// MAP LOOP
// =========================

for (const auto& [key, value] : prices) {
}


// =========================
// UNORDERED MAP
// =========================

std::unordered_map<std::string, double> lookup;

lookup["AAPL"] = 200.0;

lookup.contains("AAPL");


// =========================
// OPTIONAL
// =========================

std::optional<double> maybe_value;

maybe_value = 5.0;

if (maybe_value.has_value()) {
    std::cout << maybe_value.value();
}

maybe_value = std::nullopt;


// =========================
// STRUCT
// =========================

struct Person {

    std::string name;

    int age;

};


Person person {
    "Sam",
    22
};

person.name;

person.age;


// =========================
// CLASS
// =========================

class Account {

private:

    double balance_;


public:

    Account(double balance)
        : balance_(balance)
    {
    }


    double balance() const {

        return balance_;

    }


    void set_balance(double balance) {

        balance_ = balance;

    }

};


// =========================
// OBJECT
// =========================

Account account(100.0);

account.balance();

account.set_balance(200.0);


// =========================
// ENUM
// =========================

enum class Side {
    Buy,
    Sell
};


Side side = Side::Buy;


// =========================
// NAMESPACE
// =========================

namespace maths {

double square(double x) {

    return x * x;

}

}


maths::square(5.0);


// =========================
// LAMBDA
// =========================

auto square_lambda =
    [](double x) {

        return x * x;

    };


square_lambda(5.0);


// =========================
// LAMBDA CAPTURE
// =========================

double multiplier = 2.0;

auto multiply_lambda =
    [multiplier](double x) {

        return x * multiplier;

    };


// =========================
// ALGORITHM
// =========================

std::sort(
    values.begin(),
    values.end()
);


std::reverse(
    values.begin(),
    values.end()
);


auto found =
    std::find(
        values.begin(),
        values.end(),
        2.0
    );


auto minimum =
    std::min_element(
        values.begin(),
        values.end()
    );


auto maximum =
    std::max_element(
        values.begin(),
        values.end()
    );


// =========================
// ACCUMULATE
// =========================

double sum =
    std::accumulate(
        values.begin(),
        values.end(),
        0.0
    );


// =========================
// EXCEPTIONS
// =========================

try {

    throw std::runtime_error(
        "Something went wrong"
    );

}
catch (const std::exception& error) {

    std::cerr
        << error.what()
        << '\n';

}


// =========================
// TEMPLATE FUNCTION
// =========================

template <typename T>
T get_max(
    T a,
    T b
) {

    return a > b ? a : b;

}


// =========================
// USING ALIAS
// =========================

using Price = double;

Price p = 100.0;


// =========================
// SMART POINTER
// =========================

auto unique_value =
    std::make_unique<int>(10);

std::cout
    << *unique_value;


// =========================
// SHARED POINTER
// =========================

auto shared_value =
    std::make_shared<int>(10);


// =========================
// MOVE
// =========================

std::vector<int> first_vector = {
    1,
    2,
    3
};

std::vector<int> second_vector =
    std::move(first_vector);


// =========================
// CONST MEMBER FUNCTION
// =========================

class Example {

public:

    int get_value() const {

        return 10;

    }

};


// =========================
// STATIC
// =========================

class Config {

public:

    static constexpr int days = 252;

};


Config::days;


// =========================
// INHERITANCE
// =========================

class Base {

public:

    virtual void run() {
    }

    virtual ~Base() = default;

};


class Child : public Base {

public:

    void run() override {
    }

};


// =========================
// FILE OUTPUT
// =========================

#include <fstream>

std::ofstream output("file.txt");

output << "Hello\n";


// =========================
// FILE INPUT
// =========================

std::ifstream input("file.txt");

std::string line;

while (std::getline(input, line)) {
}


// =========================
// PREPROCESSOR
// =========================

#define VALUE 10

#ifdef VALUE
#endif


// =========================
// HEADER INCLUDE
// =========================

#include "my_file.hpp"


// =========================
// COMMON .HPP STYLE
// =========================

// my_file.hpp

#pragma once

double calculate(double x);


// =========================
// COMMON .CPP STYLE
// =========================

// my_file.cpp

#include "my_file.hpp"

double calculate(double x) {

    return x * 2.0;

}


// =========================
// COMMON MODERN C++ FORMS
// =========================

const std::vector<double>& values_ref = values;

auto& mutable_reference = values;

const auto& read_only_reference = values;


// =========================
// USEFUL MATH
// =========================

std::sqrt(4.0);

std::pow(2.0, 3.0);

std::exp(1.0);

std::log(10.0);

std::abs(-5.0);

std::max(1.0, 2.0);

std::min(1.0, 2.0);


// =========================
// SIZE TYPE
// =========================

std::size_t size =
    values.size();


// =========================
// MAIN EXAMPLE
// =========================

int main() {

    std::vector<double> numbers = {
        1.0,
        2.0,
        3.0
    };


    for (const auto& number : numbers) {

        std::cout
            << number
            << '\n';

    }


    return 0;
}