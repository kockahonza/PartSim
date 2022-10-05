#include <functional>
#include <iostream>

#include <eigen3/Eigen/Dense>
/* #include <highfive/H5File.hpp> */

#include "PartSim/PartSim.h"


constexpr double mass{0.001};
constexpr int N{50};

constexpr double epsilon{0.1};
constexpr double sigma{50};
const double sigma12{pow(sigma, 12)};
const double sigma6{pow(sigma, 6)};

constexpr double initial_spacing{50};
constexpr double dt{0.001};
constexpr double T{100000000000};
const int max_iter{static_cast<int>(T / dt)};

Eigen::Vector3d lennard_jones(const Particle& p1, const Particle& p2) {
    const Eigen::Vector3d r12{p1.get_position() - p2.get_position()};
    const double r{r12.norm()};
    return 24 * epsilon * (2 * (sigma12 / pow(r, 13)) - (sigma6 / pow(r, 7))) * r12;
}

int main() {
    using std::vector, std::cout, std::endl;

    vector<Particle> particles(N*N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            particles[i * N + j] = Particle{mass, {i * initial_spacing, j * initial_spacing, 0}};
        }
    }
    PartSim ps{particles, lennard_jones};

    ps.run(dt, T, max_iter, [](const PartSim& ps, int i) {
            cout << i << endl;

            return true;
            });

    return 0;
}
