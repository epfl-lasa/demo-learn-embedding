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
        PARAM_SCALAR(double, dt, 1.0e-2);
    };

    struct feedback : public defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 3);
    };
};

struct ParamsCTR {
    struct controller : public defaults::controller {
        // Integration time step controller
        PARAM_SCALAR(double, dt, 1.0e-2);
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
        _rot.setStiffness(0.0 * Eigen::MatrixXd::Identity(3, 3));

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
        // position ds
        // if (_external)
        //     std::cout << _requester.request<Eigen::VectorXd>(x._trans, 3).transpose() << std::endl;
        _u.head(3) = _external ? _requester.request<Eigen::VectorXd>(x._trans, 3) : _pos(R3(x._trans));
        // _u.head(3) = _pos(R3(x._trans));
        if (_u.head(3).norm() >= 0.3)
            _u.head(3) /= _u.head(3).norm() / 0.3;

        // orientation ds
        // _u.tail(3) = _rot(SO3(x._rot));
        // if (_u.tail(3).norm() >= 0.5)
        //     _u.tail(3) /= _u.tail(3).norm() / 0.5;
        _u.tail(3).setZero();
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
    OperationSpaceController(const SE3& ref_pose)
        : franka_control::control::JointControl(), _ref_pose(ref_pose), _model(std::make_shared<FrankaModel>())
    {
        // ds
        _ds.setReference(_ref_pose);

        // ctr
        Eigen::Matrix<double, 6, 6> damping = Eigen::Matrix<double, 6, 6>::Zero();
        damping.diagonal() << 120.0, 120.0, 120.0, 5.0, 5.0, 5.0;
        _ctr.setDamping(damping);

        // writer
        _writer.setFile("exp_os_7.csv");
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // state
        Eigen::Matrix<double, 7, 1> q = jointPosition(state), dq = jointVelocity(state);
        SE3 curr_pose(_model->framePose(q));

        // if (_ds.external())
        //     _writer.append(curr_pose._trans.transpose());

        std::cout << (curr_pose._trans - _ref_pose._trans).norm() << std::endl;
        if ((curr_pose._trans - _ref_pose._trans).norm() <= 0.03 && !_ds.external())
            _ds.setExternal(true);
        Eigen::Matrix<double, 6, 7> jac = _model->jacobian(q);
        curr_pose._v = jac * dq;
        _ref_pose._v = _ds(curr_pose);
        _ctr.setReference(_ref_pose);

        return jac.transpose() * _ctr(curr_pose);
    }

protected:
    // reference
    SE3 _ref_pose;
    // task space ds
    TaskDynamics _ds;
    // task space controller
    controllers::Feedback<ParamsCTR, SE3> _ctr;
    // model
    std::shared_ptr<FrankaModel> _model;
    // file manager
    FileManager _writer;
};

int main(int argc, char const* argv[])
{
    // trajectory
    std::string demo = (argc > 1) ? "demo_" + std::string(argv[1]) : "demo_1";
    YAML::Node config = YAML::LoadFile("rsc/demos/" + demo + "/dynamics_params.yaml");
    auto offset = config["offset"].as<std::vector<double>>();

    FileManager mng;
    std::vector<Eigen::MatrixXd> trajectories;
    for (size_t i = 1; i <= 7; i++) {
        trajectories.push_back(mng.setFile("rsc/demos/" + demo + "/trajectory_" + std::to_string(i) + ".csv").read<Eigen::MatrixXd>());
        trajectories.back().rowwise() += Eigen::Map<Eigen::Vector3d>(&offset[0]).transpose();
    }

    // task space target
    Eigen::Vector3d ref_pos = trajectories[0].row(0);
    Eigen::Matrix3d ref_rot = (Eigen::Matrix3d() << 0.768647, 0.239631, 0.593092, 0.0948479, -0.959627, 0.264802, 0.632602, -0.147286, -0.760343).finished();
    SE3 ref_pose(ref_rot, ref_pos);

    Franka robot("franka");
    robot.setJointController(std::make_unique<OperationSpaceController>(ref_pose));
    robot.torque();

    return 0;
}
