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
using SE3 = spatial::SE<3>;
using SO3 = spatial::SO<3, true>;

struct ParamsDS {
    struct controller : public defaults::controller {
        // Integration time step controller
        PARAM_SCALAR(double, dt, 0.01);
    };

    struct feedback : public defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 3);
    };
};

struct ParamsCTR {
    struct controller : public defaults::controller {
        // Integration time step controller
        PARAM_SCALAR(double, dt, 0.01);
    };

    struct feedback : public defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 6);
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

struct TaskDynamics : public controllers::AbstractController<ParamsDS, SE3> {
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
        _u.tail(3) = _rot(SO3(x._rot));
    }

protected:
    using AbstractController<ParamsDS, SE3>::_d;
    using AbstractController<ParamsDS, SE3>::_xr;
    using AbstractController<ParamsDS, SE3>::_u;

    controllers::Feedback<ParamsDS, R3> _pos;
    controllers::Feedback<ParamsDS, SO3> _rot;

    bool _external;
    Requester _requester;
};

class OperationSpaceController : public franka_control::control::JointControl {
public:
    OperationSpaceController(const SE3& target_pose) : franka_control::control::JointControl()
    {
        // reference
        _reference = target_pose;

        // ds
        _ds.setReference(target_pose);

        // ctr
        Eigen::Matrix<double, 6, 6> damping;
        damping.setZero();
        damping.diagonal() << 20.0, 20.0, 20.0, 1.0, 1.0, 1.0;
        _ctr.setDamping(damping);

        // model
        _model = std::make_shared<FrankaModel>();
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // state
        Eigen::Matrix<double, 7, 1> q = jointPosition(state), dq = jointVelocity(state);
        SE3 curr(_model->framePose(q));
        // std::cout << (curr._trans - _reference._trans).norm() << std::endl;
        if ((curr._trans - _reference._trans).norm() <= 0.05 && !_ds.external())
            _ds.setExternal(true);
        Eigen::Matrix<double, 6, 7> jac = _model->jacobian(q);
        curr._v = jac * dq;
        _reference._v = _ds(curr);
        _ctr.setReference(_reference);

        return jac.transpose() * _ctr(curr);
    }

protected:
    // reference
    SE3 _reference;
    // task space ds
    TaskDynamics _ds;
    // task space controller
    controllers::Feedback<ParamsCTR, SE3> _ctr;
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
    robot.setJointController(std::make_unique<OperationSpaceController>(tDes));
    robot.torque();

    return 0;
}
