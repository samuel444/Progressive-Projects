#include "../include/Player.h"

#include <iostream>

Player::Player(std::string name, int health)
    : name(name), health(health) {}

void Player::takeDamage(int damage) {
    health -= damage;
}

void Player::printInfo() const {
    std::cout << name
              << " has "
              << health
              << " health\n";
}