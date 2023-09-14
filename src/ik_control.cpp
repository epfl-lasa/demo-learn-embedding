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

// Controllers
#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/controllers/InverseKinematics.hpp>
#include <control_lib/controllers/LinearDynamics.hpp>

#include <utils_lib/FileManager.hpp>

#include <chrono>
#include <iostream>
#include <thread>

using namespace beautiful_bullet;
using namespace control_lib;
using namespace utils_lib;
using namespace std::chrono;

struct Params {
    struct controller : public defaults::controller {
        PARAM_SCALAR(double, dt, 1.0);
    };

    struct linear_dynamics : public defaults::linear_dynamics {
    };

    struct feedback : public defaults::feedback {
        /* data */
    };

    struct inverse_kinematics : public defaults::inverse_kinematics {
        // State dimension
        PARAM_SCALAR(size_t, nP, 7);

        // Slack variable dimension
        PARAM_SCALAR(size_t, nS, 6);
    };
};

struct IKController : public control::MultiBodyCtr {
    IKController(const bodies::MultiBodyPtr& model, const spatial::SE<3>& target_pose) : control::MultiBodyCtr(ControlMode::CONFIGURATIONSPACE)
    {
        // integration step
        _dt = 1.0;

        // reference frame for inverse kinematics
        _frame = "panda_link7";

        // configuration target
        spatial::R<7> state, target_state;
        state._x = model->state();
        target_state._x = (model->positionUpper() - model->positionLower()) * 0.5 + model->positionLower();
        _config
            .setDynamicsMatrix(1 * Eigen::MatrixXd::Identity(7, 7))
            .setReference(target_state)
            .update(state);

        // task target
        spatial::SE<3> pose(model->frameOrientation(), model->framePosition());
        _task
            .setDynamicsMatrix(1 * Eigen::MatrixXd::Identity(6, 6))
            .setReference(target_pose)
            .update(pose);

        // inverse kinematics
        Eigen::MatrixXd Q = 1.0 * Eigen::MatrixXd::Identity(7, 7),
                        W = 10.0 * Eigen::MatrixXd::Identity(6, 6);

        _ctr
            .setModel(model)
            .velocityMinimization(Q)
            // .velocityTracking(Q, _config)
            .slackVariable(W)
            .inverseKinematics(state, _task)
            .positionLimits(state)
            .velocityLimits()
            .init();
    }

    IKController& setTarget(const spatial::SE<3>& target_pose)
    {
        _task.setReference(target_pose);
        return *this;
    }

    Eigen::VectorXd action(bodies::MultiBody& body) override
    {
        // update config
        spatial::R<7> state;
        state._x = body.state();
        _config.update(state);

        // update task
        spatial::SE<3> pose(body.framePose(_frame));
        _task.update(pose);

        return state._x + _dt * _ctr.action(state).segment(0, 7);
    }

    // step
    double _dt;

    // configuration/task space target
    controllers::LinearDynamics<Params, spatial::R<7>> _config;
    controllers::LinearDynamics<Params, spatial::SE<3>> _task;

    // inverse dynamics controller
    controllers::InverseKinematics<Params, bodies::MultiBody> _ctr;
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
    bodies::MultiBodyPtr franka = std::make_shared<bodies::MultiBody>("rsc/franka/panda.urdf");
    Eigen::VectorXd state_ref = (franka->positionUpper() - franka->positionLower()) * 0.5 + franka->positionLower();
    franka->setState(state_ref);

    // trajectory
    FileManager mng;
    Eigen::MatrixXd traj = mng.setFile("rsc/trajectory.csv").read<Eigen::MatrixXd>();
    traj.rowwise() += Eigen::RowVector3d(0.5, -0.5, 0.5);

    // task space target
    Eigen::Vector3d xDes = traj.row(0);
    Eigen::Matrix3d oDes = (Eigen::Matrix3d() << 0.768647, 0.239631, 0.593092, 0.0948479, -0.959627, 0.264802, 0.632602, -0.147286, -0.760343).finished();
    spatial::SE<3> tDes(oDes, xDes);

    Eigen::VectorXd state(7);
    auto ctr = std::make_unique<IKController>(franka, tDes);
    for (size_t i = 0; i < 5; i++) {
        state = ctr->action(*franka);
        franka->setState(state);
    }

    // Set controlled robot
    (*franka)
        .activateGravity();
    // .addControllers(std::make_unique<IKController>(franka, tDes));

    // Add robots and run simulation
    simulator.add(franka);

    // plot trajectory
    static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(traj);

    // // plot mesh
    // Eigen::MatrixXd vertices = mng.setFile("rsc/mesh_points.csv").read<Eigen::MatrixXd>(),
    //                 indices = mng.setFile("rsc/mesh_faces.csv").read<Eigen::MatrixXd>();
    // Eigen::VectorXd fun = mng.setFile("rsc/mesh_values.csv").read<Eigen::MatrixXd>();
    // static_cast<graphics::MagnumGraphics&>(simulator.graphics())
    //     .app()
    //     .surface(vertices, fun, indices)
    //     .setTransformation(Matrix4::scaling({0.2, 0.2, 0.2}) * Matrix4::translation({2.0, -2.3, 0.0}) * Matrix4::rotationX(2.5_radf));

    // // run
    // simulator.run();
    simulator.initGraphics();
    // static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().camera3D().setPose(Vector3{2., -2.5, 2.});

    size_t index = 0;
    double t = 0.0, dt = 1e-3, T = 10.0;

    auto next = steady_clock::now();
    auto prev = next - 1ms;

    bool enter = true;
    auto limits_up = franka->positionUpper(), limits_down = franka->positionLower();

    while (t <= T) {
        auto now = steady_clock::now();

        if (!simulator.step(size_t(t / dt)))
            break;

        t += dt;

        if (size_t(t / dt) % 10 == 0) {
            if (index < traj.rows() - 1) {
                index++;
                xDes = traj.row(index);
                ctr->setTarget(spatial::SE<3>(oDes, xDes));
                for (size_t i = 0; i < 5; i++) {
                    state = ctr->action(*franka);
                    franka->setState(state);
                }
                for (size_t i = 0; i < state.rows(); i++) {
                    if (state(i) >= limits_up(i) || state(i) <= limits_down(i))
                        std::cout << "error" << std::endl;
                }
            }
        }

        prev = now;
        next += 1ms;
        std::this_thread::sleep_until(next);
    }

    return 0;
}
