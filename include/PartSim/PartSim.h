#include <vector>
#include <functional>

#include <eigen3/Eigen/Dense>

#include "PartSim/Particle.h"


// Default force functions for PartSim
Eigen::Vector3d gravitational_force(const Particle& p1, const Particle& p2);
inline Eigen::Vector3d no_external_force(const Particle&) {return {0, 0, 0};}


class PartSim {
public:
    using inter_particle_force_t = std::function<Eigen::Vector3d(Particle p1, Particle p2)>;
    using external_force_t = std::function<Eigen::Vector3d(Particle p)>;

private:
    std::vector<Particle> m_particles;
    // Calculate the force exerted by p2 on p1, it must be antisymmetric (aka follow Newtons 3rd)!
    inter_particle_force_t m_inter_particle_force;
    // Calculate the external force, only depends on the particle it affects
    external_force_t m_external_force;

    double m_time;

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

    // Particle simulation logic methods
    const std::vector<Particle>& run(double T, double dt, std::function<bool(const PartSim&)> iter_f=[](PartSim) {return true;});

    void testing() {
        for (Particle& p : m_particles) {
            p.m_position = {0, 0, 0};
        }
    };

    void print_positions();
};
