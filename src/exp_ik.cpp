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

struct ParamsConfig {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 1.0e-3); // Integration time step controller
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

struct ParamsTask {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 1.0e-3); // Integration time step controller
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 3); // Output dimension
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
        if (_u.head(3).norm() >= 0.3)
            _u.head(3) /= _u.head(3).norm() / 0.3;

        _u.tail(3) = _rot(SO3(x._rot));
        if (_u.tail(3).norm() >= 0.5)
            _u.tail(3) /= _u.tail(3).norm() / 0.5;
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

class IKController : public franka_control::control::JointControl {
public:
    IKController(const franka::RobotState& state, const SE3& ref_pose)
        : franka_control::control::JointControl(), _ref_pose(ref_pose), _model(std::make_shared<FrankaModel>())
    {
        // config ds
        R7 curr_state(jointPosition(state)),
            ref_state((_model->positionUpper() + _model->positionLower()) * 0.5);
        _config
            .setStiffness(1.0 * Eigen::MatrixXd::Identity(7, 7))
            .setReference(ref_state)
            .update(curr_state);

        // task ds
        SE3 curr_pose(_model->framePose(curr_state._x));
        _task.setReference(_ref_pose)
            .update(curr_pose);

        // ik
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(7, 7), S = Eigen::MatrixXd::Zero(6, 6);
        Q.diagonal() << 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0;
        S.diagonal() << 1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0;
        _ik
            .setModel(_model)
            .stateCost(Q)
            .stateReference(_config.output())
            .slackCost(S)
            .inverseKinematics(_task.output())
            .positionLimits()
            .velocityLimits()
            .init(curr_state);

        // ctr
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(7, 7), D = Eigen::MatrixXd::Zero(7, 7);
        K.diagonal() << 2500.0, 2500.0, 2500.0, 2500.0, 1000.0, 1000.0, 1000.0;
        K *= 20;
        // K.diagonal() << 700.0, 700.0, 700.0, 700.0, 500.0, 500.0, 300.0;
        D.diagonal() << 30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 10.0;
        _ctr
            .setStiffness(K)
            .setDamping(D);

        // writer
        _writer.setFile("exp_ik_7.csv");

        _open = false;
        _ik_state = curr_state;
    }

    Eigen::Matrix<double, 7, 1> action(const franka::RobotState& state) override
    {
        // curr
        R7 curr_state(jointPosition(state));
        curr_state._v = jointVelocity(state);
        SE3 curr_pose(_model->framePose(_open ? _ik_state._x : curr_state._x));

        // if (_task.external())
        //     _writer.append(curr_pose._trans.transpose());

        // config ds
        _config.update(_open ? _ik_state : curr_state);

        // task ds
        // std::cout << (curr_pose._trans - _ref_pose._trans).norm() << std::endl;
        if ((curr_pose._trans - _ref_pose._trans).norm() <= 0.03 && !_task.external())
            _task.setExternal(true);
        _task.update(curr_pose);

        // ik
        auto state_vel = _ik(curr_state).segment(0, 7);
        // R7 ref_state(curr_state._x + ParamsConfig::controller::dt() * state_vel);
        R7 ref_state;
        if (_open) {
            ref_state._x = _ik_state._x + ParamsConfig::controller::dt() * _ik(_ik_state).segment(0, 7);
            _ik_state = ref_state;
        }
        else
            ref_state._x = curr_state._x + ParamsConfig::controller::dt() * _ik(curr_state).segment(0, 7);
        ref_state._v.setZero();
        // std::cout << state_vel.transpose() << std::endl;

        auto tau = _ctr.setReference(ref_state).action(curr_state);

        std::cout << "tau" << std::endl;
        std::cout << tau.transpose() << std::endl;
        std::cout << "ref" << std::endl;
        std::cout << ref_state._x.transpose() << std::endl;
        std::cout << "-" << std::endl;

        return tau;
    }

protected:
    // reference
    SE3 _ref_pose;
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
    // file manager
    FileManager _writer;

    // prev state
    bool _open;
    R7 _ik_state;
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
    robot.setJointController(std::make_unique<IKController>(robot.state(), ref_pose));
    robot.torque();

    return 0;
}
