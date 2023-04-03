#include <iostream>

// Robot Handle
#include <franka_control/Franka.hpp>

// Optitrack Handle
#include <optitrack_lib/Optitrack.hpp>

// ZMQ Stream
#include <zmq_stream/Requester.hpp>

// Task Space Manifolds
#include <control_lib/spatial/R.hpp>
#include <control_lib/spatial/SO.hpp>

// Task Space Dynamical System
#include <control_lib/controllers/LinearDynamics.hpp>

// Task Space Derivative Controller
#include <control_lib/controllers/Feedback.hpp>

using namespace franka_control;
using namespace optitrack_lib;
using namespace zmq_stream;
using namespace control_lib;

using R3 = spatial::R<3>;
using SO3 = spatial::SO<3, true>;

struct Params {
    struct controller : public defaults::controller {
        // Integration time step controller
        PARAM_SCALAR(double, dt, 0.01);
    };

    struct feedback : public defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 6);
    };

    struct linear_dynamics : public defaults::linear_dynamics {
    };
};

class StreamController : public control::JointControl {
public:
    StreamController() : control::JointControl()
    {
        // reference
        Eigen::Vector3d position(0.683783, 0.308249, 0.185577);
        Eigen::Matrix3d orientation;
        orientation << 0.922046, 0.377679, 0.0846751,
            0.34527, -0.901452, 0.261066,
            0.17493, -0.211479, -0.9616;

        _r3_ref = R3(position);
        _so3_ref = SO3(orientation);

        // r3 ds
        Eigen::MatrixXd At = 1.0 * Eigen::MatrixXd::Identity(3, 3);
        _r3_ds.setDynamicsMatrix(At);
        _r3_ds.setReference(_r3_ref);

        // r3 feedback
        Eigen::MatrixXd Dt = 10.0 * Eigen::MatrixXd::Identity(3, 3);
        // Dt()
        _r3_feedback.setDamping(Dt);

        // so3 ds
        Eigen::MatrixXd Ar = 1.0 * Eigen::MatrixXd::Identity(3, 3);
        _so3_ds.setDynamicsMatrix(Ar);
        _so3_ds.setReference(_so3_ref);

        // so3 feedback
        Eigen::MatrixXd Dr = 1.0 * Eigen::MatrixXd::Identity(3, 3);
        _so3_feedback.setDamping(Dr);

        // Zmq
        _requester.configure("128.178.145.171", "5511");
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // current task space state
        auto pose = taskPose(state);
        R3 _r3_curr(pose.translation());
        SO3 _so3_curr(pose.linear());

        Eigen::Matrix<double, 6, 7> jac = jacobian(state);
        Eigen::Matrix<double, 6, 1> vel = jac * jointVelocity(state);
        _r3_curr._v = vel.head(3);
        _so3_curr._v = vel.tail(3);

        // ds
        Eigen::Matrix3d mat = Eigen::Matrix3d::Identity();
        mat(0, 0) = 3.0;
        mat(1, 1) = 3.0;
        mat(2, 2) = 3.0;
        _r3_ref._v = mat * _requester.request<Eigen::VectorXd>(_r3_curr._x, 3);
        std::cout << _r3_ref._v.norm() << std::endl;
        // std::cout << _requester.request<Eigen::VectorXd>(_r3_curr._x, 3).transpose() << std::endl;
        // _r3_ref._v = _r3_ds.action(_r3_curr);
        _so3_ref._v = _so3_ds.action(_so3_curr);

        // std::cout << "C++" << std::endl;
        // std::cout << _r3_ref._v.transpose() << std::endl;
        // std::cout << "python" << std::endl;
        // std::cout << _requester.request<Eigen::VectorXd>(_r3_curr._x, 3).transpose() << std::endl;

        Eigen::Matrix<double, 7, 1> tau = jac.transpose() * (Eigen::Matrix<double, 6, 1>() << _r3_feedback.setReference(_r3_ref).action(_r3_curr), _so3_feedback.setReference(_so3_ref).action(_so3_curr)).finished();
        // tau(0) *= 1.5;

        return tau;
    }

protected:
    // reference state
    R3 _r3_ref;
    SO3 _so3_ref;

    // task space ds
    controllers::LinearDynamics<Params, R3> _r3_ds;
    controllers::LinearDynamics<Params, SO3> _so3_ds;

    // task space controller (in this case this space is actually R3 x SO<3>)
    controllers::Feedback<Params, R3> _r3_feedback;
    controllers::Feedback<Params, SO3> _so3_feedback;

    // Zmq
    Requester _requester;
};

int main(int argc, char const* argv[])
{
    Franka robot("franka");

    robot.setJointController(std::make_unique<StreamController>());

    // robot.move();
    robot.torque();

    return 0;
}
