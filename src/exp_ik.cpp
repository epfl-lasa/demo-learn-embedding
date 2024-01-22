#include <iostream>

// Robot Handle
#include <franka_control/Franka.hpp>

// Task Space Manifolds
#include <control_lib/spatial/R.hpp>
#include <control_lib/spatial/SE.hpp>
#include <control_lib/spatial/SO.hpp>

// Robot Model
#include <beautiful_bullet/bodies/MultiBody.hpp>

// Task Space Dynamical System & Derivative Controller
#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/controllers/QuadraticControl.hpp>

// Reading/Writing Files
#include <utils_lib/FileManager.hpp>

// Stream
#include <zmq_stream/Requester.hpp>

// parse yaml
#include <yaml-cpp/yaml.h>

using namespace franka_control;
using namespace control_lib;
using namespace beautiful_bullet;
using namespace zmq_stream;
using namespace utils_lib;

using R3 = spatial::R<3>;
using R7 = spatial::R<7>;
using SE3 = spatial::SE<3>;
using SO3 = spatial::SO<3, true>;

struct ParamsTask {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 0.01); // Integration time step controller
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 3); // Output dimension
    };
};

struct ParamsConfig {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 0.01); // Integration time step controller
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 7); // Output dimension
    };

    struct quadratic_control : public defaults::quadratic_control {
        PARAM_SCALAR(size_t, nP, 7); // State dimension
        PARAM_SCALAR(size_t, nC, 0); // Control/Input dimension
        PARAM_SCALAR(size_t, nS, 6); // Slack variable dimension
        PARAM_SCALAR(size_t, oD, 1); // derivative order (optimization joint velocity)
    };
};

struct FrankaModel : public bodies::MultiBody {
public:
    FrankaModel() : bodies::MultiBody("rsc/franka/panda.urdf"), _frame("panda_joint_8"), _reference(pinocchio::LOCAL_WORLD_ALIGNED) {}

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q)
    {
        return static_cast<bodies::MultiBody*>(this)->jacobian(q, _frame, _reference);
    }

    Eigen::MatrixXd jacobianDerivative(const Eigen::VectorXd& q, const Eigen::VectorXd& dq)
    {
        return static_cast<bodies::MultiBody*>(this)->jacobianDerivative(q, dq, _frame, _reference);
    }

    Eigen::Matrix<double, 6, 1> framePose(const Eigen::VectorXd& q)
    {
        return static_cast<bodies::MultiBody*>(this)->framePose(q, _frame);
    }

    Eigen::Matrix<double, 6, 1> frameVelocity(const Eigen::VectorXd& q, const Eigen::VectorXd& dq)
    {
        return static_cast<bodies::MultiBody*>(this)->frameVelocity(q, dq, _frame, _reference);
    }

    std::string _frame;
    pinocchio::ReferenceFrame _reference;
};

struct TaskDynamics : public controllers::AbstractController<ParamsTask, SE3> {
    TaskDynamics()
    {
        _d = SE3::dimension(); // adjust output dimension
        _u.setZero(_d);

        // ds
        _pos.setStiffness(5.0 * Eigen::MatrixXd::Identity(3, 3));
        _rot.setStiffness(1.0 * Eigen::MatrixXd::Identity(3, 3));

        // external ds stream
        _external = false;
        _requester.configure("128.178.145.171", "5511");
    }

    TaskDynamics& setReference(const SE3& x)
    {
        _pos.setReference(R3(x._trans));
        _rot.setReference(SO3(x._rot));
        return *this;
    }

    const bool& external() { return _external; }

    TaskDynamics& setExternal(const bool& value)
    {
        _external = value;
        return *this;
    }

    void update(const SE3& x) override
    {
        // if (_external)
        //     std::cout << _requester.request<Eigen::VectorXd>(x._trans, 3).transpose() << std::endl;
        _u.head(3) = _external ? _requester.request<Eigen::VectorXd>(x._trans, 3) : _pos(R3(x._trans));
        // _u.head(3) = _pos(R3(x._trans));
        // _u.tail(3) = _rot(SO3(x._rot));
        _u.tail(3).setZero();
    }

protected:
    using AbstractController<ParamsTask, SE3>::_d;
    using AbstractController<ParamsTask, SE3>::_xr;
    using AbstractController<ParamsTask, SE3>::_u;

    controllers::Feedback<ParamsTask, R3> _pos;
    controllers::Feedback<ParamsTask, SO3> _rot;

    bool _external;
    Requester _requester;
};

class IKController : public franka_control::control::JointControl {
public:
    IKController(const SE3& target_pose) : franka_control::control::JointControl()
    {
        // step
        _dt = 0.03;

        // reference
        _reference = target_pose;

        // model
        _model = std::make_shared<FrankaModel>();

        // task ds
        _task.setReference(target_pose);

        // config ds
        R7 target_state((_model->positionUpper() - _model->positionLower()) * 0.5 + _model->positionLower());
        _config
            .setStiffness(1.0 * Eigen::MatrixXd::Identity(7, 7))
            .setReference(target_state);

        // ik
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(7, 7), S = Eigen::MatrixXd::Zero(6, 6);
        Q.diagonal() << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
        S.diagonal() << 10.0, 10.0, 10.0, 10.0, 10.0, 10.0;
        _ik
            .setModel(_model)
            .stateCost(Q)
            // .stateReference(_config.output())
            .slackCost(S)
            .inverseKinematics(_task.output())
            .positionLimits()
            .velocityLimits()
            .init(target_state);

        // ctr
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(7, 7), D = Eigen::MatrixXd::Zero(7, 7);
        K.diagonal() << 800.0, 1100.0, 800.0, 1100.0, 100.0, 10.0, 10.0;
        D.diagonal() << 50.0, 50.0, 50.0, 50.0, 10.0, 1.0, 1.0;
        _ctr.setStiffness(K).setDamping(D);
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // state
        R7 config_curr(jointPosition(state));
        config_curr._v = jointVelocity(state);
        SE3 task_curr(_model->framePose(config_curr._x));

        // update task ds
        std::cout << (task_curr._trans - _reference._trans).norm() << std::endl;
        if ((task_curr._trans - _reference._trans).norm() <= 0.03 && !_task.external())
            _task.setExternal(true);
        _task.update(task_curr);

        // update config ds
        _config.update(config_curr);

        // ik
        R7 config_ref(config_curr._x + _dt * _ik(config_curr).segment(0, 7));
        config_ref._v = Eigen::VectorXd::Zero(7);

        return _ctr.setReference(config_ref).action(config_curr);
    }

protected:
    // external ds
    double _dt;
    // reference
    SE3 _reference;
    // task space ds
    TaskDynamics _task;
    // configuration space ds
    controllers::Feedback<ParamsConfig, R7> _config;
    // inverse kinematics
    controllers::QuadraticControl<ParamsConfig, FrankaModel> _ik;
    // task space controller
    controllers::Feedback<ParamsConfig, R7> _ctr;
    // model
    std::shared_ptr<FrankaModel> _model;
};

int main(int argc, char const* argv[])
{
    // trajectory
    std::string demo = (argc > 1) ? "demo_" + std::string(argv[1]) : "demo_1";
    YAML::Node config = YAML::LoadFile("rsc/demos/" + demo + "/dynamics_params.yaml");
    auto offset = config["offset"].as<std::vector<double>>();

    FileManager mng;
    std::vector<Eigen::MatrixXd> trajectories;
    for (size_t i = 1; i <= 1; i++) {
        trajectories.push_back(mng.setFile("rsc/demos/" + demo + "/trajectory_" + std::to_string(i) + ".csv").read<Eigen::MatrixXd>());
        trajectories.back().rowwise() += Eigen::Map<Eigen::Vector3d>(&offset[0]).transpose();
    }

    // task space target
    Eigen::Vector3d xDes = trajectories[0].row(0);
    Eigen::Matrix3d oDes = (Eigen::Matrix3d() << 0.768647, 0.239631, 0.593092, 0.0948479, -0.959627, 0.264802, 0.632602, -0.147286, -0.760343).finished();
    // Eigen::Vector3d xDes(0.683783, 0.308249, 0.185577);
    // Eigen::Matrix3d oDes = (Eigen::Matrix3d() << 0.922046, 0.377679, 0.0846751, 0.34527, -0.901452, 0.261066, 0.17493, -0.211479, -0.9616).finished();
    SE3 tDes(oDes, xDes);

    Franka robot("franka");
    robot.setJointController(std::make_unique<IKController>(tDes));
    robot.torque();

    return 0;
}
