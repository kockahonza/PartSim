#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "PartSim/Particle.h"
#include "PartSim/PartSim.h"
#include "PartSim/util.h"


// Angle to get from {1, 0, 0} to the given 2d vector (the third component should be 0)
double angle_to_e1_2d(const Eigen::Vector3d& v_2d) {
    const double v_2d_norm = v_2d.norm();
    return 180 * (atan2(v_2d[1] / v_2d_norm, v_2d[0] / v_2d_norm) / M_PI);
}


// Force functionals method definitions from here

Eigen::Vector3d LennardJonesForce::operator()(const Particle& p1, const Particle& p2) {
    const Eigen::Vector3d r12{p1.get_position() - p2.get_position()};
    const double r{r12.norm()};
    return 24 * m_epsilon * (2 * (m_sigma12 / pow(r, 13)) - (m_sigma6 / pow(r, 7))) * r12;
}

// H5Saver method definitions from here

void H5Saver::append_row(const PartSim& ps) {
    using std::vector, Eigen::Vector3d;

    const vector<Particle>& particles{ps.get_particles()};

    // Collect the newly calculated data
    #pragma omp parallel for default(none) shared(particles, m_temp_positions, m_temp_velocities, m_temp_forces)
    for (size_t j = 0; j < m_particle_count; j++) {
        m_temp_positions[0][j] = particles[j].get_position();
        m_temp_velocities[0][j] = particles[j].get_velocity();
        m_temp_forces[0][j] = particles[j].get_force();
    }

    // And write it to the right place in the hdf5 file
    HighFive::Selection positions_selection{m_positions_ds.select({m_i, 0, 0, 0}, {1, m_particle_count, 3, 1})};
    HighFive::Selection velocities_selection{m_velocities_ds.select({m_i, 0, 0, 0}, {1, m_particle_count, 3, 1})};
    HighFive::Selection forces_selection{m_forces_ds.select({m_i, 0, 0, 0}, {1, m_particle_count, 3, 1})};
    positions_selection.write(m_temp_positions);
    velocities_selection.write(m_temp_velocities);
    forces_selection.write(m_temp_forces);

    // Don't forget time
    m_times_ds.select({m_i}, {1}).write(vector<double>{ps.get_time()});

    m_i += 1;
}

bool H5Saver::safe_append_row(const PartSim& ps) {
    using std::cout, std::endl;
    if (m_i >= m_N) {
        cout << "Error in H5Saver.safe_append_row : m_i >= m_n : the data file is full already" << endl;
        return false;
    } else if (ps.get_particles().size() != m_particle_count) {
        cout << "Error in H5Saver.safe_append_row : the number of particles in PartSim does not equal the given particle_count" << endl;
        return false;
    }
    append_row(ps);
    return true;
}

bool H5Saver::save_extra(std::string prefix, const PartSim& ps) {
    using std::string;
    string positions_path{prefix + "_positions"};
    string velocities_path{prefix + "_velocities"};
    string forces_path{prefix + "_forces"};
    string time_attr_name{prefix + "_time"};

    if (m_file.exist(positions_path) && m_file.exist(velocities_path) && m_file.exist(forces_path) && m_file.hasAttribute(time_attr_name)) {
        std::cout << "data with prefix \"" + prefix + "\" has already been saved or created" << std::endl;
        return false;
    }

    const std::vector<Particle>& particles{ps.get_particles()};
    #pragma omp parallel for default(none) shared(particles, m_temp_positions, m_temp_velocities, m_temp_forces)
    for (size_t j = 0; j < m_particle_count; j++) {
        m_temp_positions[0][j] = particles[j].get_position();
        m_temp_velocities[0][j] = particles[j].get_velocity();
        m_temp_forces[0][j] = particles[j].get_force();
    }
    m_file.createDataSet(positions_path, m_temp_positions);
    m_file.createDataSet(velocities_path, m_temp_velocities);
    m_file.createDataSet(forces_path, m_temp_forces);
    m_file.createAttribute(time_attr_name, ps.get_time());
    return true;
}
