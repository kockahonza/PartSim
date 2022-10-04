#include <functional>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <SFML/Graphics.hpp>

#include "PartSim/PartSim.h"
#include "PartSim/util.h"


constexpr double mass{0.001};
constexpr int N{20};

constexpr double epsilon{0.1};
constexpr double sigma{50};
const double sigma12{pow(sigma, 12)};
const double sigma6{pow(sigma, 6)};

constexpr double initial_spacing{50};
constexpr double T{100000000000};
constexpr double dt{0.001};

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

    sf::RenderWindow window{sf::VideoMode{1000, 1000}, "KAK"};

    ps.run(dt, T, [&window](const PartSim& ps, int i) {
            cout << ps.get_time() << endl;

            sf::Event event;
            while (window.pollEvent(event)) {}

            const std::vector<Particle>& particles{ps.get_particles()};

            window.clear({0, 0, 0});

            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                const Eigen::Vector3d& pos{particles[i].get_position()};
                const Eigen::Vector3d& vel{particles[i].get_velocity()};
                const double vel_angle{angle_to_e1_2d(vel)};
                const Eigen::Vector3d& foc{particles[i].get_force()};
                const double foc_angle{angle_to_e1_2d(foc)};

                /* cout << "p" << i << " - " << pos[0] << ", " << pos[1] << ", " << pos[2] << */
                /*                   " and " << vel[0] << ", " << vel[1] << ", " << vel_angle << */
                /*                   " and " << foc[0] << ", " << foc[1] << ", " << foc_angle << endl; */

                sf::CircleShape pos_circle{10};
                pos_circle.setOrigin(10, 10);
                pos_circle.setFillColor({static_cast<sf::Uint8>(100*i), 100, 0});
                pos_circle.setPosition(pos[0], pos[1]);

                sf::RectangleShape vel_line{{20, 2}};
                vel_line.setOrigin(0, 1);
                vel_line.setFillColor({static_cast<sf::Uint8>(100*i), 100, 0});
                vel_line.setPosition(pos[0], pos[1]);
                vel_line.setRotation(vel_angle);

                sf::RectangleShape foc_line{{20, 2}};
                foc_line.setOrigin(0, 1);
                foc_line.setFillColor({static_cast<sf::Uint8>(100*i), 100, 0});
                foc_line.setPosition(pos[0], pos[1]);
                foc_line.setRotation(foc_angle);

                #pragma omp critical
                {
                    window.draw(pos_circle);
                    window.draw(vel_line);
                    window.draw(foc_line);
                }
            }

            window.display();

            return true;
            });

    return 0;
}
