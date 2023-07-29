#include <franka_control/Franka.hpp>

#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/spatial/R.hpp>

#include <beautiful_bullet/bodies/MultiBody.hpp>
#include <zmq_stream/Requester.hpp>

using namespace franka_control;
using namespace control_lib;
using namespace beautiful_bullet;
using namespace zmq_stream;

struct Params {
    struct controller : public defaults::controller {
        // Integration time step controller
        PARAM_SCALAR(double, dt, 0.01);
    };

    struct feedback : public defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 7);
    };
};

class ConfigController : public franka_control::control::JointControl {
public:
    // ConfigController() : control::JointControl(ControlMode::CONFIGURATIONSPACE)
    ConfigController() : franka_control::control::JointControl()
    {
        // step
        _dt = 0.01;

        // gains
        Eigen::MatrixXd K = 3 * Eigen::MatrixXd::Identity(7, 7), D = 1 * Eigen::MatrixXd::Identity(7, 7);

        // goal
        spatial::R<7> ref((Eigen::Matrix<double, 7, 1>() << 0.300325, 0.596986, 0.140127, -1.44853, 0.15547, 2.31046, 0.690596).finished());
        ref._v = Eigen::Matrix<double, 7, 1>::Zero();

        // set controller
        _controller.setStiffness(K).setDamping(D).setReference(ref);

        _orientation << 0.0450009, 0.998953, 0.00821996,
            0.998876, -0.045117, 0.0145344,
            0.01489, 0.00755666, -0.999861;
        _requester.configure("128.178.145.171", "5511");
        _model = std::make_shared<bodies::MultiBody>("models/franka/urdf/panda.urdf");
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // current state
        spatial::R<7> curr(jointPosition(state));
        curr._v = jointVelocity(state);

        auto pose = taskPose(state);
        double dt = 0.01;

        // std::cout << pose.rotation() << std::endl;

        Eigen::Vector3d pos_ref = pose.translation() + dt * _requester.request<Eigen::VectorXd>(pose.translation(), 3);
        _model->setState(curr._x);
        spatial::R<7> ref(_model->inverseKinematics(pos_ref, _orientation));
        ref._v = Eigen::Matrix<double, 7, 1>::Zero();
        // std::cout << ref._x.transpose() << std::endl;
        return _controller.setReference(ref).action(curr);
        // auto tau = _controller.action(curr);

        // std::cout << tau.transpose() << std::endl;

        // return _controller.action(curr);
    }

protected:
    // step
    double _dt;

    // controller
    controllers::Feedback<Params, spatial::R<7>> _controller;

    bodies::MultiBodyPtr _model;

    Eigen::Matrix3d _orientation;

    Requester _requester;
};

int main(int argc, char const* argv[])
{
    Franka robot("franka");

    // franka::RobotState state = robot.robot().readOnce();

    // auto ctr = std::make_unique<ConfigController>();

    // std::cout << ctr->action(state) << std::endl;

    // std::cout << ctr->jointPosition(state).transpose() << std::endl;

    robot.setJointController(std::make_unique<ConfigController>());

    robot.torque();

    return 0;
}
