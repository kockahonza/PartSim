#ifndef UTIL_H
#define UTIL_H

#include <eigen3/Eigen/Dense>

#include "PartSim/Particle.h"


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

#endif
