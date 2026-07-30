#pragma once

#include <string>

class Player {
public:
    Player(std::string name, int health);

    void takeDamage(int damage);
    void printInfo() const;

private:
    std::string name;
    int health;
};