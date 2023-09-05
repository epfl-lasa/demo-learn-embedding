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

using namespace beautiful_bullet;
using namespace control_lib;
using namespace utils_lib;

struct Params {
    struct controller : public defaults::controller {
    };

    struct feedback : public defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 6);
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

struct OperationSpaceFeedback : public control::MultiBodyCtr {
    OperationSpaceFeedback(const spatial::SE<3>& target) : control::MultiBodyCtr(ControlMode::OPERATIONSPACE)
    {
        // set controlled frame
        _frame = "lbr_iiwa_link_7";

        // set ds gains and reference
        Eigen::MatrixXd A = 0.1 * Eigen::MatrixXd::Identity(6, 6);
        _ds.setDynamicsMatrix(A);
        _ds.setReference(target);

        // set controller gains
        Eigen::MatrixXd D = 1 * Eigen::MatrixXd::Identity(6, 6);
        _feedback.setDamping(D);
    }

    Eigen::VectorXd action(bodies::MultiBody& body) override
    {
        // current state
        spatial::SE<3> sCurr(body.framePose(_frame));
        sCurr._v = body.frameVelocity(_frame);

        // reference state
        spatial::SE<3> sRef;
        sRef._v = _ds.action(sCurr);

        return _feedback.setReference(sRef).action(sCurr);
    }

    // ds & feedback
    controllers::LinearDynamics<Params, spatial::SE<3>> _ds;
    controllers::Feedback<Params, spatial::SE<3>> _feedback;
};

struct ReferenceConfiguration {
    ReferenceConfiguration()
    {
        _Kp = 0.01 * Eigen::MatrixXd::Identity(7, 7);
        _Kd = 0.03 * Eigen::MatrixXd::Identity(7, 7);

        // _f.setZero(7);
        // _a.setZero(7);

        // _ref.setZero(7);
        _ref << 0., 0.7, 0.4, 0.6, 0.3, 0.5, 0.1;
    }

    size_t dimension() const { return 7; };

    ReferenceConfiguration& setModel(const bodies::MultiBodyPtr& model)
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

struct ConfigurationSpaceQP : public control::MultiBodyCtr {
    ConfigurationSpaceQP(const bodies::MultiBodyPtr& model, const spatial::SE<3>& target) : control::MultiBodyCtr(ControlMode::CONFIGURATIONSPACE)
    {
        // step
        _dt = 0.1;

        // set controlled frame
        _frame = "lbr_iiwa_link_7";

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

        // set ref tracking
        _ref
            .setModel(model)
            .update(state);

        // set qp
        Eigen::MatrixXd Q = 50 * Eigen::MatrixXd::Identity(7, 7),
                        Qt = 10 * Eigen::MatrixXd::Identity(7, 7),
                        R = 1 * Eigen::MatrixXd::Identity(7, 7),
                        Rt = 0.5 * Eigen::MatrixXd::Identity(7, 7),
                        W = 0 * Eigen::MatrixXd::Identity(6, 6);
        // W.diagonal() << 10, 10, 10, 5, 5, 5;

        _qp.setModel(model)
            .accelerationMinimization(Q)
            // .accelerationTracking(Qt, _ref)
            .effortMinimization(R)
            .effortTracking(Rt, _ref)
            .slackVariable(W)
            .modelDynamics(state)
            // .inverseKinematics(state, _ds)
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

        // update robot tracker
        _ref.update(state);

        // std::cout << "reference" << std::endl;
        // // std::cout << _ref.acceleration().transpose() << std::endl;
        // // std::cout << _ref.effort().transpose() << std::endl;
        // std::cout << _ds.velocity().transpose() << std::endl;

        // auto u = _qp.action(state);

        // std::cout << "=====" << std::endl;

        return _qp.action(state);
    }

    // step
    double _dt;

    // ds
    controllers::LinearDynamics<Params, spatial::SE<3>> _ds;

    // reference configuration
    ReferenceConfiguration _ref;

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
    bodies::MultiBodyPtr franka = std::make_shared<bodies::MultiBody>("models/franka/panda.urdf");

    // state
    Eigen::VectorXd state(7);
    state << 0, 0.2, 0.1, 0.2, 0.3, 0.7, 0.2;

    // task space target
    Eigen::Vector3d xDes(0.365308, -0.0810892, 1.13717);
    Eigen::Matrix3d oDes = (Eigen::Matrix3d() << 0.591427, -0.62603, 0.508233, 0.689044, 0.719749, 0.0847368, -0.418848, 0.300079, 0.857041).finished();
    spatial::SE<3> tDes(oDes, xDes);

    // trajectory
    FileManager mng;
    Eigen::MatrixXd traj = mng.setFile("rsc/trajectory.csv").read<Eigen::MatrixXd>();
    traj.rowwise() += Eigen::RowVector3d(0.5, -0.5, 0.5);

    static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(traj);

    // Set controlled robot
    (*franka)
        .setState(state)
        // .setPosition(0, -1, 0)
        // .addControllers(std::make_unique<OperationSpaceFeedback>(tDes))
        // .addControllers(std::make_unique<ConfigurationSpaceQP>(iiwaBullet, tDes));
        .activateGravity();

    // Add robots and run simulation
    simulator.add(franka);
    simulator.run();

    return 0;
}
