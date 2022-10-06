#include <functional>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <libconfig.hh>

#include "PartSim/PartSim.h"
#include "PartSim/util.h"


const std::string config_filename{"LennardJones.cfg"};

struct config {
    int N;
    double mass;
    double epsilon;
    double sigma;

    double initial_spacing;

    double dt;
    double T;
    int max_iter;

    std::string output_filename;
};


int main() {
    using std::vector, std::cout, std::endl;

    // Read the configuration file (the location of which is hard coded)
    libconfig::Config config_file{};
    config_file.readFile(config_filename);
    const config cfg{
        config_file.lookup("LennardJonesFast.N"),
        config_file.lookup("LennardJonesFast.mass"),
        config_file.lookup("LennardJonesFast.epsilon"),
        config_file.lookup("LennardJonesFast.sigma"),
        config_file.lookup("LennardJonesFast.initial_spacing"),
        config_file.lookup("LennardJonesFast.dt"),
        config_file.lookup("LennardJonesFast.T"),
        config_file.lookup("LennardJonesFast.max_iter"),
        config_file.lookup("LennardJonesFast.output_filename")
    };

    const int particle_count{cfg.N * cfg.N};

    // Setup the particles and environment
    vector<Particle> particles(particle_count);
    for (int i = 0; i < cfg.N; i++) {
        for (int j = 0; j < cfg.N; j++) {
            particles[i * cfg.N + j] = Particle{cfg.mass, {i * cfg.initial_spacing, j * cfg.initial_spacing, 0}};
        }
    }
    PartSim ps{particles, LennardJonesForce(cfg.epsilon, cfg.sigma)};

    sf::RenderWindow window{sf::VideoMode{1000, 1000}, "LennardJones with SFML"};

    ps.run(cfg.dt, cfg.T, [&window](const PartSim& ps, int i) {
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
