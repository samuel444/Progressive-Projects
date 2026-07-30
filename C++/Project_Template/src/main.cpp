#include "../include/Player.h"

int main() {

    Player player("Sam", 100);

    player.printInfo();

    player.takeDamage(20);

    player.printInfo();

    return 0;
}