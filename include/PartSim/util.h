#ifndef UTIL_H
#define UTIL_H

#include <string>

#include <eigen3/Eigen/Dense>
#include <highfive/H5File.hpp>

#include "PartSim/Particle.h"
#include "PartSim/PartSim.h"


double angle_to_e1_2d(const Eigen::Vector3d& v_2d);


class LennardJonesForce {
private:
    double m_epsilon;
    double m_sigma;
    double m_sigma6;
    double m_sigma12;
public:
    LennardJonesForce(double epsilon, double sigma) :
        m_epsilon{epsilon},
        m_sigma{sigma},
        m_sigma6{pow(sigma, 6)},
        m_sigma12{pow(m_sigma6, 2)} {};

    Eigen::Vector3d operator()(const Particle& p1, const Particle& p2);
};


class H5Saver {
private:
    HighFive::File m_file;

    // Parameters and an internal counter i
    size_t m_N;
    size_t m_particle_count;
    size_t m_i;

    // The main data datasets, saved to within append_row
    HighFive::DataSpace main_datatype_dataspace;

    HighFive::DataSet m_positions_ds;
    HighFive::DataSet m_velocities_ds;
    HighFive::DataSet m_forces_ds;
    HighFive::DataSet m_times_ds;

    // Used in various places as temporaries to save time allocating them all the time
    std::vector<std::vector<Eigen::Vector3d>> m_temp_positions;
    std::vector<std::vector<Eigen::Vector3d>> m_temp_velocities;
    std::vector<std::vector<Eigen::Vector3d>> m_temp_forces;

public:
    H5Saver(std::string filename, int N, int particle_count) :
        m_file(HighFive::File(filename, HighFive::File::Overwrite)),
        m_N{static_cast<size_t>(N)},
        m_particle_count{static_cast<size_t>(particle_count)},
        m_i{0},
        main_datatype_dataspace{m_N, m_particle_count, 3, 1},
        m_positions_ds {m_file.createDataSet<double>("positions",  main_datatype_dataspace)},
        m_velocities_ds{m_file.createDataSet<double>("velocities", main_datatype_dataspace)},
        m_forces_ds    {m_file.createDataSet<double>("forces",     main_datatype_dataspace)},
        m_times_ds     {m_file.createDataSet<double>("times",      HighFive::DataSpace(m_N))},
        m_temp_positions {1, std::vector<Eigen::Vector3d>(m_particle_count)},
        m_temp_velocities{1, std::vector<Eigen::Vector3d>(m_particle_count)},
        m_temp_forces    {1, std::vector<Eigen::Vector3d>(m_particle_count)}
    {};

    const HighFive::File& get_file() const {return m_file;};

    // These methods append to the main "positions", "velocities", "forces" and "times" datasets
    void append_row(const PartSim& ps);
    bool safe_append_row(const PartSim& ps);

    // A single snapshot can be easily saved with a given prefix
    bool save_extra(std::string prefix, const PartSim& ps);
};

#endif
