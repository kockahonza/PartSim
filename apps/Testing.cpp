#include <iostream>
#include <random>

#include <eigen3/Eigen/Dense>

#include "PartSim/PartSim.h"
#include "PartSim/util.h"

double kak(Particle p) { return p.get_position().norm(); }

int main() {
    using std::cout, std::endl, std::vector, Eigen::Vector3d;

    LabeledParticle lp1{42, 2, {3, 0, 0}};
    /* Particle* p1{&lp1}; */

    cout << kak(lp1);

    return 0;
}
