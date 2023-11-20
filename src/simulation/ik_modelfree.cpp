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

// Simulator
#include <beautiful_bullet/Simulator.hpp>

// Graphics
#include <beautiful_bullet/graphics/MagnumGraphics.hpp>

// Spaces
#include <control_lib/spatial/SE.hpp>
#include <control_lib/spatial/SO.hpp>

// Controllers
#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/controllers/QuadraticControl.hpp>

// Reading/Writing Files
#include <utils_lib/FileManager.hpp>

// Stream
#include <zmq_stream/Requester.hpp>

// parse yaml
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <iostream>
#include <thread>

using namespace beautiful_bullet;
using namespace control_lib;
using namespace utils_lib;
using namespace std::chrono;
using namespace zmq_stream;

using R3 = spatial::R<3>;
using R7 = spatial::R<7>;
using SE3 = spatial::SE<3>;
using SO3 = spatial::SO<3, true>;

struct ParamsConfig {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 1.0);
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 7);
    };

    struct quadratic_control : public defaults::quadratic_control {
        // State dimension
        PARAM_SCALAR(size_t, nP, 7);

        // Control/Input dimension
        PARAM_SCALAR(size_t, nC, 0);

        // Slack variable dimension
        PARAM_SCALAR(size_t, nS, 6);

        // derivative order (optimization joint velocity)
        PARAM_SCALAR(size_t, oD, 1);
    };
};

struct ParamsTask {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 1.0);
    };

    struct feedback : public defaults::feedback {
        PARAM_SCALAR(size_t, d, 3);
    };
};

struct TaskDynamics : public controllers::AbstractController<ParamsTask, SE3> {
    TaskDynamics()
    {
        _d = SE3::dimension();
        _u.setZero(_d);

        // position ds weights
        _pos
            .setStiffness(1.0 * Eigen::MatrixXd::Identity(3, 3));

        // orientation ds weights
        _rot
            .setStiffness(1.0 * Eigen::MatrixXd::Identity(3, 3));

        // external ds stream
        _external = false;
        _requester.configure("localhost", "5511");
    }

    TaskDynamics& setReference(const SE3& x)
    {
        _pos.setReference(R3(x._trans));
        _rot.setReference(SO3(x._rot));
        return *this;
    }

    TaskDynamics& setExternal(const bool& value)
    {
        _external = value;
        return *this;
    }

    void update(const SE3& x) override
    {
        // position ds
        _u.head(3) = _external ? _requester.request<Eigen::VectorXd>(x._trans, 3) : _pos(R3(x._trans));

        // orientation ds
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

struct FrankaModel : public bodies::MultiBody {
public:
    FrankaModel() : bodies::MultiBody("rsc/franka/panda.urdf"), _frame("panda_joint_8"), _reference(pinocchio::LOCAL_WORLD_ALIGNED) {}

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q) { return static_cast<bodies::MultiBody*>(this)->jacobian(q, _frame, _reference); }

    std::string _frame;
    pinocchio::ReferenceFrame _reference;
};

struct IKController : public control::MultiBodyCtr {
    IKController(const std::shared_ptr<FrankaModel>& model, const SE3& target_pose) : control::MultiBodyCtr(ControlMode::CONFIGURATIONSPACE)
    {
        // integration step
        _dt = 0.03;

        // reference frame for inverse kinematics
        _frame = model->_frame;

        // configuration target
        R7 state, target_state;
        state._x = model->state();
        target_state._x = (model->positionUpper() - model->positionLower()) * 0.5 + model->positionLower();
        _config
            .setStiffness(1.0 * Eigen::MatrixXd::Identity(7, 7))
            .setReference(target_state)
            .update(state);

        // task target
        SE3 pose(model->framePose());
        _task
            .setReference(target_pose)
            .update(pose);

        // inverse kinematics
        Eigen::MatrixXd Q = 1.0 * Eigen::MatrixXd::Identity(7, 7),
                        S = 10.0 * Eigen::MatrixXd::Identity(6, 6);

        _ik
            .setModel(model)
            .stateCost(Q)
            // .stateReference(_config.output())
            .slackCost(S)
            .inverseKinematics(_task.output())
            .positionLimits()
            .velocityLimits()
            .init(state);

        // joints controller
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(7, 7), D = Eigen::MatrixXd::Zero(7, 7);
        K.diagonal() << 700.0, 700.0, 700.0, 700.0, 500.0, 500.0, 50.0;
        D.diagonal() << 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0;
        _ctr
            .setStiffness(K)
            .setDamping(D);
    }

    IKController& setTarget(const SE3& target_pose)
    {
        _task.setReference(target_pose);
        return *this;
    }

    IKController& setExternalDynamics(const bool& value)
    {
        _task.setExternal(value);
        return *this;
    }

    Eigen::VectorXd action(bodies::MultiBody& body) override
    {
        // config
        R7 state;
        state._x = body.state();
        state._v = body.velocity();
        _config.update(state);

        // task
        SE3 pose(body.framePose(_frame));
        _task.update(pose);

        // ik
        auto ref = R7(state._x + _dt * _ik(state).segment(0, 7));
        ref._v = Eigen::VectorXd::Zero(7);

        return _ctr.setReference(ref).action(state) + body.gravityVector(state._x);
    }

    // step
    double _dt;

    // configuration space ds
    controllers::Feedback<ParamsConfig, R7> _config;

    // task space ds
    TaskDynamics _task;

    // inverse dynamics
    controllers::QuadraticControl<ParamsConfig, FrankaModel> _ik;

    // joint space controller
    controllers::Feedback<ParamsConfig, R7> _ctr;
};

int main(int argc, char const* argv[])
{
    // Create simulator
    Simulator simulator;

    // Add graphics
    simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());

    // Add ground
    simulator.addGround();

    // Multi Bodies
    auto franka = std::make_shared<FrankaModel>();
    Eigen::VectorXd state_ref = (franka->positionUpper() - franka->positionLower()) * 0.5 + franka->positionLower();
    franka->setState(state_ref);

    // trajectory
    std::string demo = (argc > 1) ? "demo_" + std::string(argv[1]) : "demo_1";
    YAML::Node config = YAML::LoadFile("rsc/demos/" + demo + "/dynamics_params.yaml");
    auto offset = config["dynamics"]["offset"].as<std::vector<double>>();

    FileManager mng;
    std::vector<Eigen::MatrixXd> trajectories;
    for (size_t i = 1; i <= 1; i++) {
        trajectories.push_back(mng.setFile("rsc/demos/" + demo + "/trajectory_" + std::to_string(i) + ".csv").read<Eigen::MatrixXd>());
        trajectories.back().rowwise() += Eigen::Map<Eigen::Vector3d>(&offset[0]).transpose();
        static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(trajectories.back(), i >= 4 ? "red" : "green");
    }

    // Eigen::MatrixXd traj = mng.setFile("rsc/demos/" + demo + "/trajectory_1.csv").read<Eigen::MatrixXd>();
    // traj.rowwise() += offset.transpose();

    // task space target
    Eigen::Vector3d xDes = trajectories[0].row(0);
    Eigen::Matrix3d oDes = (Eigen::Matrix3d() << 0.768647, 0.239631, 0.593092, 0.0948479, -0.959627, 0.264802, 0.632602, -0.147286, -0.760343).finished();
    SE3 tDes(oDes, xDes);

    auto controller = std::make_shared<IKController>(franka, tDes);

    // Set controlled robot
    (*franka)
        // .activateGravity()
        .addControllers(controller);

    // Add robots and run simulation
    simulator.add(static_cast<bodies::MultiBodyPtr>(franka));

    // plot trajectory
    // static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(trajectories[0]);

    // run
    // simulator.run();
    simulator.initGraphics();

    size_t index = 0;
    double t = 0.0, dt = 1e-3, T = 40.0;

    auto next = steady_clock::now();
    auto prev = next - 1ms;

    bool enter = true;
    auto limits_up = franka->positionUpper(), limits_down = franka->positionLower();

    while (t <= T) {
        auto now = steady_clock::now();

        if (!simulator.step(size_t(t / dt)))
            break;

        t += dt;

        // if ((xDes - franka->framePosition("panda_joint8")).norm() <= 0.01 && enter) {
        //     std::cout << "Activating DS" << std::endl;
        //     controller->setExternalDynamics(true);
        //     enter = false;
        // }

        // std::cout << (xDes - franka->framePosition("panda_link7")).norm() << std::endl;

        prev = now;
        next += 1ms;
        std::this_thread::sleep_until(next);
    }

    return 0;
}