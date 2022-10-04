#include <functional>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <SFML/Graphics.hpp>

#include "PartSim/PartSim.h"


// Angle to get from {1, 0, 0} to the given 2d vector (the third component should be 0)
double angle_to(const Eigen::Vector3d& v_2d) {
    const double v_2d_norm = v_2d.norm();
    return 180 * (atan2(v_2d[1] / v_2d_norm, v_2d[0] / v_2d_norm) / M_PI);
}

int main() {
    using std::cout, std::endl;

    constexpr double G{6.6743e-11}; //m^3 kg^-1 s^-2
    constexpr double m1{1000};
    constexpr double m2{1000};
    constexpr double M{m1 + m2};
    constexpr double R{200};
    const double v{5e-4};
    cout << G << ", " << m1 << ", " << m2 << ", " << M << ", " << R << ", " << v << endl;

    const Eigen::Vector3d e1{1, 0, 0};
    const Eigen::Vector3d e2{0, 1, 0};
    const Eigen::Vector3d CM{500, 500, 0};

    PartSim ps{{
        Particle{m1, CM - (R * m2 / M) * e1, (- v * m2 / M) * e2},
        Particle{m2, CM + (R * m1 / M) * e1, (v * m1 / M) * e2}
    }};


    sf::RenderWindow window{sf::VideoMode{1000, 1000}, "GAG"};

    ps.run(500, 1000000000, [&window](const PartSim& ps, int i) {
            sf::Event event;
            while (window.pollEvent(event)) {}

            const std::vector<Particle>& particles{ps.get_particles()};

            window.clear({0, 0, 0});

            for (size_t i = 0; i < particles.size(); i++) {
                const Eigen::Vector3d& pos{particles[i].get_position()};
                const Eigen::Vector3d& vel{particles[i].get_velocity()};
                const double vel_angle{angle_to(vel)};
                const Eigen::Vector3d& foc{particles[i].get_force()};
                const double foc_angle{angle_to(foc)};

                cout << "p" << i << " - " << pos[0] << ", " << pos[1] << ", " << pos[2] <<
                                  " and " << vel[0] << ", " << vel[1] << ", " << vel_angle <<
                                  " and " << foc[0] << ", " << foc[1] << ", " << foc_angle << endl;

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

                window.draw(pos_circle);
                window.draw(vel_line);
                window.draw(foc_line);
            }

            window.display();
            return true;
            });

    cout << "Sim is done, just waiting for exit now" << endl;
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
    }

    /* earth_orbit_sim("lal"); */

    return 0;
}
