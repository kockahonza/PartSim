#ifndef PARTSIM_H
#define PARTSIM_H

#include <vector>
#include <functional>

#include <eigen3/Eigen/Dense>

#include "PartSim/Particle.h"


// Default force functions for PartSim
Eigen::Vector3d gravitational_force(const Particle& p1, const Particle& p2);
inline Eigen::Vector3d no_external_force(const Particle&) {return {0, 0, 0};}


class PartSim {
// Types, keep them public so they may be used outside
public:
    using inter_particle_force_t = std::function<Eigen::Vector3d(Particle p1, Particle p2)>;
    using external_force_t = std::function<Eigen::Vector3d(Particle p)>;

    using iter_f_t = std::function<bool(const PartSim&, int)>;

// Internal properties
private:
    std::vector<Particle> m_particles;
    // Calculate the force exerted by p2 on p1, it must be antisymmetric (aka follow Newtons 3rd)!
    inter_particle_force_t m_inter_particle_force;
    // Calculate the external force, only depends on the particle it affects
    external_force_t m_external_force;

    double m_time;

// Public API
public:
    // Constructors
    PartSim(std::vector<Particle> particles={},
            inter_particle_force_t inter_particle_force=gravitational_force,
            external_force_t external_force=no_external_force) :
        m_particles{particles},
        m_inter_particle_force{inter_particle_force},
        m_external_force{external_force},
        m_time{0} {}

    // Simple getters and setters
    const std::vector<Particle>& set_particles(std::vector<Particle> new_particles) {return m_particles = new_particles;};

    const std::vector<Particle>& get_particles() const {return m_particles;};
    const double& get_time() const {return m_time;};

    void print_positions();

    // Particle simulation logic methods
    int run(double dt, double T, int max_iter=std::numeric_limits<int>::max(),
            iter_f_t iter_f=[](PartSim, int) {return true;});

    int run(double dt, double T, iter_f_t iter_f) {return run(dt, T, std::numeric_limits<int>::max(), iter_f);};

    void testing() {
        for (Particle& p : m_particles) {
            p.m_position = {0, 0, 0};
        }
    };

// Private methods
private:
    void step(double dt);
};

#endif
