#include <iostream>
#include <random>

#include <eigen3/Eigen/Dense>
#include <highfive/H5File.hpp>
#include <libconfig.hh>

#include "PartSim/PartSim.h"
#include "PartSim/util.h"


const std::string config_filename{"LJF3D.cfg"};

struct config {
    int N;
    double mass;
    double epsilon;
    double sigma;

    double initial_spacing;
    double random_offset_max;

    double dt;
    double T;
    int max_iter;

    int save_every;
    std::string output_filename;
};


int main() {
    using std::cout, std::endl, std::vector, Eigen::Vector3d;

    // Read the configuration file (the location of which is hard coded)
    libconfig::Config config_file{};
    config_file.readFile(config_filename);
    const config cfg{
        config_file.lookup("LennardJonesFast.N"),
        config_file.lookup("LennardJonesFast.mass"),
        config_file.lookup("LennardJonesFast.epsilon"),
        config_file.lookup("LennardJonesFast.sigma"),
        config_file.lookup("LennardJonesFast.initial_spacing"),
        config_file.lookup("LennardJonesFast.random_offset_max"),
        config_file.lookup("LennardJonesFast.dt"),
        config_file.lookup("LennardJonesFast.T"),
        config_file.lookup("LennardJonesFast.max_iter"),
        config_file.lookup("LennardJonesFast.save_every"),
        config_file.lookup("LennardJonesFast.output_filename")
    };

    const int particle_count{cfg.N * cfg.N * cfg.N};

    // Setup the particles and environment
    std::default_random_engine re{};
    std::uniform_real_distribution<double> ud{0, cfg.random_offset_max};
    vector<Particle> particles(particle_count);
    for (int i = 0; i < cfg.N; i++) {
        for (int j = 0; j < cfg.N; j++) {
            for (int k = 0; k < cfg.N; k++) {
                particles[i * cfg.N * cfg.N + j * cfg.N + k] = Particle{cfg.mass, {i * cfg.initial_spacing + ud(re),
                    j * cfg.initial_spacing + ud(re), k * cfg.initial_spacing + ud(re)}};
            }
        }
    }
    PartSim ps{particles, LennardJonesForce(cfg.epsilon, cfg.sigma)};

    // Open the output file and write metadata
    H5Saver h5saver{cfg.output_filename, cfg.max_iter / cfg.save_every, particle_count};
    HighFive::File h5_file{h5saver.get_file()};

    h5_file.createAttribute("N", cfg.N);
    h5_file.createAttribute("mass", cfg.mass);
    h5_file.createAttribute("epsilon", cfg.epsilon);
    h5_file.createAttribute("sigma", cfg.sigma);
    h5_file.createAttribute("initial_spacing", cfg.initial_spacing);
    h5_file.createAttribute("random_offset_max", cfg.random_offset_max);
    h5_file.createAttribute("dt", cfg.dt);
    h5_file.createAttribute("T", cfg.T);
    h5_file.createAttribute("save_every", cfg.save_every);
    h5_file.createAttribute("max_iter", cfg.max_iter);

    // Run the simulation
    ps.run(cfg.dt, cfg.T, cfg.max_iter, [&h5saver, &cfg](const PartSim& ps, int i) {
            if (i % cfg.save_every == 0) {
                cout << i << endl;
                return h5saver.safe_append_row(ps);
            }

            return true;
            });

    // Also collect the final state
    h5saver.save_extra("final", ps);

    return 0;
}
