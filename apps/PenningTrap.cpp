#include <iostream>
#include <random>

#include <eigen3/Eigen/Dense>
#include <highfive/H5File.hpp>
#include <libconfig.hh>

#include "PartSim/PartSim.h"
#include "PartSim/util.h"

const std::string config_filename{"PenTrap.cfg"};

struct config {
    int mass;

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
        config_file.lookup("LennardJonesFast.mass"),       config_file.lookup("LennardJonesFast.dt"),
        config_file.lookup("LennardJonesFast.T"),          config_file.lookup("LennardJonesFast.max_iter"),
        config_file.lookup("LennardJonesFast.save_every"), config_file.lookup("LennardJonesFast.output_filename")};

    PartSim ps{{cfg.mass}, no_interparticle_force, [](const Particle &p, double time) {
                   return Eigen::Vector3d{0, 0, 0};
               }};

    // Open the output file and write metadata
    H5Saver h5saver{cfg.output_filename, cfg.max_iter / cfg.save_every, 1};
    HighFive::File h5_file{h5saver.get_file()};

    h5_file.createAttribute("mass", cfg.mass);
    h5_file.createAttribute("dt", cfg.dt);
    h5_file.createAttribute("T", cfg.T);
    h5_file.createAttribute("save_every", cfg.save_every);
    h5_file.createAttribute("max_iter", cfg.max_iter);

    // Run the simulation
    ps.run(cfg.dt, cfg.T, cfg.max_iter, [&h5saver, &cfg](const PartSim &ps, int i) {
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
