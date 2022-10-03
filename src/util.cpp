#include <eigen3/Eigen/Dense>


int gag() {return 42;}

// Angle to get from {1, 0, 0} to the given 2d vector (the third component should be 0)
double angle_to_e1_2d(const Eigen::Vector3d& v_2d) {
    const double v_2d_norm = v_2d.norm();
    return 180 * (atan2(v_2d[1] / v_2d_norm, v_2d[0] / v_2d_norm) / M_PI);
}

