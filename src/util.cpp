#include <eigen3/Eigen/Dense>

#include "PartSim/Particle.h"
#include "PartSim/util.h"


// Angle to get from {1, 0, 0} to the given 2d vector (the third component should be 0)
double angle_to_e1_2d(const Eigen::Vector3d& v_2d) {
    const double v_2d_norm = v_2d.norm();
    return 180 * (atan2(v_2d[1] / v_2d_norm, v_2d[0] / v_2d_norm) / M_PI);
}


Eigen::Vector3d LennardJonesForce::operator()(const Particle& p1, const Particle& p2) {
    const Eigen::Vector3d r12{p1.get_position() - p2.get_position()};
    const double r{r12.norm()};
    return 24 * m_epsilon * (2 * (m_sigma12 / pow(r, 13)) - (m_sigma6 / pow(r, 7))) * r12;
}
