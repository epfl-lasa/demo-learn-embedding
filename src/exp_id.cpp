/*
    This file is part of beautiful-bullet.

    Copyright (c) 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

// Robot Handle
#include <franka_control/Franka.hpp>

// Task Space Manifolds
#include <control_lib/spatial/R.hpp>
#include <control_lib/spatial/SE.hpp>
#include <control_lib/spatial/SO.hpp>

// Robot Model
#include <beautiful_bullet/bodies/MultiBody.hpp>

// Controllers
#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/controllers/QuadraticControl.hpp>

// CPP Utils
#include <utils_lib/FileManager.hpp>
#include <utils_lib/Timer.hpp>

// Stream
#include <zmq_stream/Requester.hpp>

// parse yaml
#include <yaml-cpp/yaml.h>

using namespace franka_control;
using namespace beautiful_bullet;
using namespace control_lib;
using namespace utils_lib;
using namespace zmq_stream;

using R3 = spatial::R<3>;
using R7 = spatial::R<7>;
using SE3 = spatial::SE<3>;
using SO3 = spatial::SO<3, true>;

struct ParamsConfig {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 5.0e-3);
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 7);
    };

    struct quadratic_control : public defaults::quadratic_control {
        PARAM_SCALAR(size_t, nP, 7); // State dimension
        PARAM_SCALAR(size_t, nC, 7); // Control/Input dimension (optimization torques)
        PARAM_SCALAR(size_t, nS, 6); // Slack variable dimension (optimization slack)
        PARAM_SCALAR(size_t, oD, 2); // derivative order (optimization joint acceleration)
    };
};

struct ParamsTask {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 5.0e-3);
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 3);
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
        _d = SE3::dimension();
        _u.setZero(_d);

        // position ds weights
        double k = 30.0, d = 2.0 * std::sqrt(k);
        _pos
            .setStiffness(k * Eigen::MatrixXd::Identity(3, 3))
            .setDamping(d * Eigen::MatrixXd::Identity(3, 3));

        // orientation ds weights
        _rot.setStiffness(k * Eigen::MatrixXd::Identity(3, 3))
            .setDamping(d * Eigen::MatrixXd::Identity(3, 3));

        // external ds stream
        _external = false;
        _requester.configure("128.178.145.171", "5511");
    }

    TaskDynamics& setReference(const SE3& x)
    {
        auto p = R3(x._trans);
        p._v = x._v.head(3);
        _pos.setReference(p);

        auto r = SO3(x._rot);
        r._v = x._v.tail(3);
        _rot.setReference(r);

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
        auto p = R3(x._trans);
        p._v = x._v.head(3);
        Eigen::Matrix<double, 6, 1> state;
        state << p._x, p._v;
        if (_external)
            _u.head(3) = _requester.request<Eigen::VectorXd>(state, 3);
        else {
            _u.head(3) = _pos(p);
            if (_u.head(3).norm() >= 5.0)
                _u.head(3) /= _u.head(3).norm() / 5.0;
        }
        // _u.head(3) = _external ? _requester.request<Eigen::VectorXd>(state, 3) : _pos(p);
        // // _u.head(3) = _pos(p);
        // if (_u.head(3).norm() >= 5.0)
        //     _u.head(3) /= _u.head(3).norm() / 5.0;

        // orientation ds
        auto r = SO3(x._rot);
        r._v = x._v.tail(3);
        _u.tail(3) = _rot(r);
        // _u.tail(3).setZero();
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

struct IDController : public franka_control::control::JointControl {
    IDController(const franka::RobotState& state, const SE3& ref_pose)
        : franka_control::control::JointControl(), _ref_pose(ref_pose), _model(std::make_shared<FrankaModel>())
    {
        // configuration ds
        R7 curr_state(jointPosition(state)),
            ref_state((_model->positionUpper() + _model->positionLower()) * 0.5);
        curr_state._v = jointVelocity(state);
        ref_state._v.setZero();
        double k = 1.0, d = 2.0 * std::sqrt(k);
        _config
            .setStiffness(k * Eigen::MatrixXd::Identity(7, 7))
            .setDamping(d * Eigen::MatrixXd::Identity(7, 7))
            .setReference(ref_state)
            .update(curr_state);

        // input reference
        _ref_input = _model->gravityVector(curr_state._x);

        // task ds
        SE3 curr_pose(_model->framePose(curr_state._x));
        curr_pose._v = _model->jacobian(curr_state._x) * curr_state._v;
        _task
            .setReference(_ref_pose)
            .update(curr_pose);

        // inverse dynamics
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(7, 7), R = Eigen::MatrixXd::Zero(7, 7), S = Eigen::MatrixXd::Zero(6, 6);
        Q.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
        R.diagonal() << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
        S.diagonal() << 5000.0, 5000.0, 5000.0, 10.0, 10.0, 10.0;
        _id
            .setModel(_model)
            .stateCost(Q)
            .inputCost(R)
            .inputReference(_ref_input)
            .stateReference(_config.output())
            .slackCost(S)
            .modelConstraint()
            .inverseDynamics(_task.output())
            .positionLimits()
            .velocityLimits()
            .accelerationLimits()
            .effortLimits()
            .init(curr_state);

        // writer
        _writer.setFile("exp_id_7.csv");
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // curr
        R7 curr_state(jointPosition(state));
        curr_state._v = jointVelocity(state);
        SE3 curr_pose(_model->framePose(curr_state._x));
        curr_pose._v = _model->frameVelocity(curr_state._x, curr_state._v);

        if (_task.external())
            _writer.append(curr_pose._trans.transpose());

        // task ds
        // std::cout << (curr_pose._trans - _ref_pose._trans).norm() << std::endl;
        if ((curr_pose._trans - _ref_pose._trans).norm() <= 0.04 && !_task.external())
            _task.setExternal(true);
        _task.update(curr_pose);

        // config ds
        _config.update(curr_state);

        // input reference
        _ref_input = _model->gravityVector(curr_state._x);

        // std::cout << "pinocchio" << std::endl;
        // std::cout << _model->jacobian(curr_state._x) << std::endl;
        // std::cout << "franka" << std::endl;
        // std::cout << jacobian(state) << std::endl;
        // std::cout << "-" << std::endl;

        // _id.update(curr_state);
        // auto ref_vel = curr_state._v + ParamsConfig::controller::dt() * _id.output().segment(0, 7);
        // auto ref_pos = curr_state._x + ParamsConfig::controller::dt() * ref_vel;

        // Eigen::MatrixXd K = Eigen::MatrixXd::Zero(7, 7), D = Eigen::MatrixXd::Zero(7, 7);
        // K.diagonal() << 2500.0, 2500.0, 2500.0, 2500.0, 1000.0, 1000.0, 1000.0;
        // K *= 2.0;
        // D.diagonal() << 30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 10.0;
        // D *= 0.5;

        // ctr
        return _id(curr_state).segment(7, 7) - _ref_input;
    }

    // pose reference
    SE3 _ref_pose;
    // input reference
    Eigen::Matrix<double, 7, 1> _ref_input;
    // task space ds
    TaskDynamics _task;
    // configuration space ds
    controllers::Feedback<ParamsConfig, R7> _config;
    // inverse dynamics
    controllers::QuadraticControl<ParamsConfig, FrankaModel> _id;
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
    ref_pose._v.setZero();

    Franka robot("franka");
    robot.setJointController(std::make_unique<IDController>(robot.state(), ref_pose));
    robot.torque();

    return 0;
}

// std::cout << "local" << std::endl;
// std::cout << static_cast<bodies::MultiBodyPtr>(franka)->jacobian("", pinocchio::LOCAL) << std::endl;

// std::cout << "world" << std::endl;
// std::cout << static_cast<bodies::MultiBodyPtr>(franka)->jacobian("", pinocchio::WORLD) << std::endl;

// std::cout << "local aligned" << std::endl;
// std::cout << static_cast<bodies::MultiBodyPtr>(franka)->jacobian("", pinocchio::LOCAL_WORLD_ALIGNED) << std::endl;

// std::cout << static_cast<bodies::MultiBodyPtr>(franka)->pose().transpose() << std::endl;

// pinocchio::SE3 ref_pose(oDes, xDes);

// std::cout << pinocchio::log6(ref_pose).toVector().transpose() << std::endl;