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
#include <control_lib/spatial/R.hpp>
#include <control_lib/spatial/SE.hpp>

// Controllers
#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/controllers/LinearDynamics.hpp>
#include <control_lib/controllers/QuadraticProgramming.hpp>

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

    struct quadratic_programming : public defaults::quadratic_programming {
        // State dimension
        PARAM_SCALAR(size_t, nP, 7);

        // Control input dimension
        PARAM_SCALAR(size_t, nC, 7);

        // Slack variable dimension
        PARAM_SCALAR(size_t, nS, 6);
    };
};

struct JointAcceleration {
    JointAcceleration()
    {
        _Kp = 0.01 * Eigen::MatrixXd::Identity(7, 7);
        _Kd = 0.03 * Eigen::MatrixXd::Identity(7, 7);

        _ref << 0., 0.7, 0.4, 0.6, 0.3, 0.5, 0.1;
    }

    size_t dimension() const { return 7; };

    JointAcceleration& setModel(const bodies::MultiBodyPtr& model)
    {
        _model = model;
        return *this;
    }

    void update(const spatial::R<7>& state)
    {
        _a = _Kp * (_ref - state._x) - _Kd * state._v;
        _f = _model->nonLinearEffects(state._x, state._v);
    }

    const Eigen::Matrix<double, 7, 1>& acceleration() const { return _a; }

    const Eigen::Matrix<double, 7, 1>& effort() const { return _f; }

protected:
    bodies::MultiBodyPtr _model;
    Eigen::Matrix<double, 7, 1> _ref, _a, _f;
    Eigen::MatrixXd _Kp, _Kd;
};

struct JointEffort {
    JointEffort()
    {
    }

    size_t dimension() const { return 7; };

    JointEffort& setModel(const bodies::MultiBodyPtr& model)
    {
        _model = model;
        return *this;
    }

    void update(const spatial::R<7>& state)
    {
        _f = _model->nonLinearEffects(state._x, state._v);
    }

    const Eigen::Matrix<double, 7, 1>& effort() const { return _f; }

protected:
    bodies::MultiBodyPtr _model;
    Eigen::Matrix<double, 7, 1> _f;
};

struct ConfigurationSpaceQP : public control::MultiBodyCtr {
    ConfigurationSpaceQP(const bodies::MultiBodyPtr& model, const spatial::SE<3>& target) : control::MultiBodyCtr(ControlMode::CONFIGURATIONSPACE)
    {
        // step
        _dt = 0.1;

        // set controlled frame
        _frame = "panda_link7";

        // Robot state
        spatial::R<7> state;
        state._x = model->state();
        state._v = model->velocity();
        state._a = model->acceleration();
        state._f = model->effort();

        // Task state
        spatial::SE<3> pose(model->frameOrientation(), model->framePosition());

        // set ds
        Eigen::MatrixXd A = 1 * Eigen::MatrixXd::Identity(6, 6);
        _ds
            .setDynamicsMatrix(A)
            .setReference(target)
            .update(pose);

        // set qp
        Eigen::MatrixXd Q = 10.0 * Eigen::MatrixXd::Identity(7, 7),
                        R = 0.01 * Eigen::MatrixXd::Identity(7, 7),
                        W = 100.0 * Eigen::MatrixXd::Identity(6, 6);

        _qp.setModel(model)
            .accelerationMinimization(Q)
            .effortMinimization(R)
            .slackVariable(W)
            .modelDynamics(state)
            .inverseKinematics(state, _ds)
            .positionLimits(state)
            .init();
    }

    Eigen::VectorXd action(bodies::MultiBody& body) override
    {
        // task state
        spatial::SE<3> sCurr(body.framePose(_frame));

        // robot state
        spatial::R<7> state;
        state._x = body.state();
        state._v = body.velocity();
        state._a = body.acceleration();
        state._f = body.effort();

        // update ds
        _ds.update(sCurr);

        auto tau = _qp.action(state);
        // std::cout << _ds.output().transpose() << std::endl;

        return tau;
    }

    // step
    double _dt;

    // ds
    controllers::LinearDynamics<Params, spatial::SE<3>> _ds;

    // qp
    controllers::QuadraticProgramming<Params, bodies::MultiBody> _qp;
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

    // Set controlled robot
    (*franka)
        // .activateGravity();
        .addControllers(std::make_unique<ConfigurationSpaceQP>(franka, tDes));

    // Add robots and run simulation
    simulator.add(franka);

    // plot trajectory
    static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(traj);

    // run
    simulator.run();

    return 0;
}
