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
    HighFive::File h5_file{cfg.output_filename, HighFive::File::Overwrite};

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

    // Setup storage for the data to be collected and saved, the extra 1 last dimension is due to the way HighFive treats Eigen matrices
    HighFive::DataSpace main_storage_dataspace(cfg.max_iter / cfg.save_every, particle_count, 3, 1);
    HighFive::DataSet positions_ds{h5_file.createDataSet<double>("positions", main_storage_dataspace)};
    HighFive::DataSet velocities_ds{h5_file.createDataSet<double>("velocities", main_storage_dataspace)};
    HighFive::DataSet forces_ds{h5_file.createDataSet<double>("forces", main_storage_dataspace)};
    HighFive::DataSet time_ds{h5_file.createDataSet<double>("times", HighFive::DataSpace(cfg.max_iter / cfg.save_every))};

    // Run the simulation
    ps.run(cfg.dt, cfg.T, cfg.max_iter, [&cfg, &particle_count, &positions_ds, &velocities_ds, &forces_ds, &time_ds](const PartSim& ps, int i) {
            if (i % cfg.save_every == 0) {
                cout << i << endl;

                const vector<Particle>& particles{ps.get_particles()};

                // Collect the newly calculated data
                vector<vector<Vector3d>> new_positions{1, vector<Vector3d>(particle_count)};
                vector<vector<Vector3d>> new_velocities{1, vector<Vector3d>(particle_count)};
                vector<vector<Vector3d>> new_forces{1, vector<Vector3d>(particle_count)};
                #pragma omp for
                for (size_t j = 0; j < particle_count; j++) {
                    new_positions[0][j] = particles[j].get_position();
                    new_velocities[0][j] = particles[j].get_velocity();
                    new_forces[0][j] = particles[j].get_force();
                }

                // And write it to the right place in the hdf5 file
                size_t k{static_cast<size_t>(i / cfg.save_every)};
                HighFive::Selection positions_selection{positions_ds.select({k, 0, 0, 0}, {1, 125, 3, 1})};
                HighFive::Selection velocities_selection{velocities_ds.select({k, 0, 0, 0}, {1, 125, 3, 1})};
                HighFive::Selection forces_selection{forces_ds.select({k, 0, 0, 0}, {1, 125, 3, 1})};
                positions_selection.write(new_positions);
                velocities_selection.write(new_velocities);
                forces_selection.write(new_forces);

                // Don't forget time
                time_ds.select({k}, {1}).write(vector<double>{ps.get_time()});
            }

            return true;
            });

    // Also collect the final state
    vector<Vector3d> final_positions(particle_count);
    vector<Vector3d> final_velocities(particle_count);
    vector<Vector3d> final_forces(particle_count);

    const vector<Particle>& final_particles{ps.get_particles()};
    #pragma omp for default(none) shared(positions, velocities, forces)
    for (size_t j = 0; j < particle_count; j++) {
        final_positions[j] = final_particles[j].get_position();
        final_velocities[j] = final_particles[j].get_velocity();
        final_forces[j] = final_particles[j].get_force();
    }
    h5_file.createDataSet("final_positions", final_positions);
    h5_file.createDataSet("final_velocities", final_velocities);
    h5_file.createDataSet("final_forces", final_forces);
    h5_file.createAttribute("final_time", ps.get_time());

    return 0;
}
