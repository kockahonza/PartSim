#include <functional>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <highfive/H5File.hpp>
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
    using std::vector, std::cout, std::endl, HighFive::File;

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

    // Open the output file now, so that any errors are caught before the simulation
    File h5_file{cfg.output_filename, File::Overwrite};

    // Setup storage for the data to be collected and saved
    vector<vector<Eigen::Vector3d>> positions(cfg.max_iter, vector<Eigen::Vector3d>(particle_count));
    vector<vector<Eigen::Vector3d>> velocities(cfg.max_iter, vector<Eigen::Vector3d>(particle_count));
    vector<vector<Eigen::Vector3d>> forces(cfg.max_iter, vector<Eigen::Vector3d>(particle_count));

    // Run the simulation
    ps.run(cfg.dt, cfg.T, cfg.max_iter, [&positions, &velocities, &forces](const PartSim& ps, int i) {
            cout << i << endl;
            const vector<Particle>& particles{ps.get_particles()};
            #pragma omp for default(none) shared(positions, velocities, forces)
            for (size_t j = 0; j < particles.size(); j++) {
                positions[i][j] = particles[j].get_position();
                velocities[i][j] = particles[j].get_velocity();
                forces[i][j] = particles[j].get_force();
            }

            return true;
            });

    // Finally store the data and metadata
    h5_file.createDataSet("positions", positions);
    h5_file.createDataSet("velocities", velocities);
    h5_file.createDataSet("forces", forces);

    h5_file.createAttribute("N", cfg.N);
    h5_file.createAttribute("mass", cfg.mass);
    h5_file.createAttribute("epsilon", cfg.epsilon);
    h5_file.createAttribute("sigma", cfg.sigma);
    h5_file.createAttribute("initial_spacing", cfg.initial_spacing);
    h5_file.createAttribute("dt", cfg.dt);
    h5_file.createAttribute("T", cfg.T);
    h5_file.createAttribute("max_iter", cfg.max_iter);

    return 0;
}
