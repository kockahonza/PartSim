#include <iostream>
#include <vector>

#include <omp.h>
#include <eigen3/Eigen/Dense>

#include "PartSim/Particle.h"
#include "PartSim/PartSim.h"


constexpr double G{6.6743e-11}; //m^3 kg^-1 s^-2
Eigen::Vector3d gravitational_force(const Particle& p1, const Particle& p2) {
    Eigen::Vector3d r{p2.get_position() - p1.get_position()};
    return G * p1.get_mass() * p2.get_mass() * r / r.dot(r);
}


const std::vector<Particle>& PartSim::run(double T, double dt, std::function<bool(const PartSim&)> iter_f) {
    while (m_time < T) {
        #pragma omp parallel for
        for (Particle& p : m_particles) {
            p.m_force = m_external_force(p);
        }

        Eigen::Vector3d inter_particle_force_temp;
        #pragma omp parallel default(none) private(inter_particle_force_temp)
        {
            std::vector<Eigen::Vector3d> my_forces(m_particles.size(), {0, 0, 0});
            #pragma omp for
            for (auto i1 = 0; i1 != m_particles.size(); ++i1) {
                for (auto i2 = i1 + 1; i2 != m_particles.size(); ++i2) {
                    inter_particle_force_temp = m_inter_particle_force(m_particles[i1], m_particles[i2]);
                    my_forces[i1] += inter_particle_force_temp;
                    my_forces[i2] -= inter_particle_force_temp;
                }
            }
            #pragma omp critical
            {
                for (auto i = 0; i < m_particles.size(); i++) {
                    m_particles[i].m_force += my_forces[i];
                }
            }
        }

        // Old, non-parallel version, keeping for benchmarking/checking if it still works
        /* Eigen::Vector3d inter_particle_force_temp; */
        /* #pragma omp parallel for default(none) private(inter_particle_force_temp) */
        /* for (auto p1{m_particles.begin()}; p1 != m_particles.end(); ++p1) { */
        /*     for (auto p2{p1 + 1}; p2 != m_particles.end(); ++p2) { */
        /*         inter_particle_force_temp = m_inter_particle_force(*p1, *p2); */
        /*         (*p1).m_force += inter_particle_force_temp; */
        /*         (*p2).m_force -= inter_particle_force_temp; */
        /*     } */
        /* } */

        #pragma omp parallel for
        for (Particle& p : m_particles) {
            p.m_position += dt * p.m_velocity;
            p.m_velocity += dt * p.m_force / p.m_mass;
        }

        m_time += dt;
        if (!iter_f(*this)) {break;};
    }

    return m_particles;
};

void PartSim::print_positions() {
    std::cout << "Printing particle positions from PartSim\n";
    for (size_t i = 0; i < m_particles.size(); i++) {
        std::cout << "Particle " << i << " is at (" <<
            m_particles[i].m_position[0] << ", " <<
            m_particles[i].m_position[1] << ", " <<
            m_particles[i].m_position[2] << ")\n";
    }
    std::cout << "Printing finished" << std::endl;
}
