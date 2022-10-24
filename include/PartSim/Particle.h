#ifndef PARTICLE_H
#define PARTICLE_H

#include <eigen3/Eigen/Dense>


class Particle {
private:
    Eigen::Vector3d m_position;
    Eigen::Vector3d m_velocity;
    Eigen::Vector3d m_force;

    double m_mass;

public:
    // Constructors
    Particle(double mass=1, Eigen::Vector3d position={0, 0, 0}, Eigen::Vector3d velocity={0, 0, 0}, Eigen::Vector3d force={0, 0, 0}) :
        m_position{position}, m_velocity{velocity}, m_force{force}, m_mass{mass} {}

    // Simple setters and getters
    const Eigen::Vector3d& set_position(Eigen::Vector3d new_position) {return m_position = new_position;}
    const Eigen::Vector3d& set_velocity(Eigen::Vector3d new_velocity) {return m_velocity = new_velocity;}
    const Eigen::Vector3d& set_force(Eigen::Vector3d new_force) {return m_force = new_force;}
    // Mass cannot be changed

    const Eigen::Vector3d& get_position() const {return m_position;}
    const Eigen::Vector3d& get_velocity() const {return m_velocity;}
    const Eigen::Vector3d& get_force() const {return m_force;}
    const double& get_mass() const {return m_mass;}

    friend class PartSim;
};


template<typename T>
class LabeledParticle : public Particle {
private:
    T m_label;

public:
    LabeledParticle(T label, double mass=1, Eigen::Vector3d position={0, 0, 0}, Eigen::Vector3d velocity={0, 0, 0}, Eigen::Vector3d force={0, 0, 0}) :
        Particle(mass, position, velocity, force), m_label{label} {}

    const T& getLabel() const {return m_label;}

    const T& setLabel(T new_label) {return m_label = new_label;}

};


#endif
