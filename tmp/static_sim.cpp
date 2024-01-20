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
#include <beautiful_bullet/graphics/MagnumGraphics.hpp>
#include <utils_lib/FileManager.hpp>

using namespace beautiful_bullet;
using namespace utils_lib;

int main(int argc, char const* argv[])
{
    // Create simulator
    Simulator simulator;

    // Add graphics
    simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());

    // Add ground
    // simulator.addGround();
    bodies::BoxParams paramsBox;
    paramsBox.setSize(0.3, 0.3, 0.01).setMass(0.0).setFriction(0.5).setColor("grey");
    bodies::RigidBodyPtr box = std::make_shared<bodies::RigidBody>("box", paramsBox);
    box->setPosition(0.7, 0.0, 0.4);

    // Multi Bodies
    bodies::MultiBodyPtr franka = std::make_shared<bodies::MultiBody>("rsc/franka/panda.urdf");
    Eigen::VectorXd state_ref(7);
    state_ref << 0.3, 0.3, 0.0, -1.5, 0.0, 1.8, 0.0;
    franka->setState(state_ref);

    // trajectory
    FileManager mng;
    Eigen::MatrixXd traj = mng.setFile("rsc/trajectory.csv").read<Eigen::MatrixXd>();
    traj.rowwise() += Eigen::RowVector3d(0.5, -0.5, 0.5);

    // Set controlled robot
    (*franka)
        .activateGravity();

    // Add robots and run simulation
    simulator.add(franka, box);

    // // plot trajectory
    // static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(traj);

    // // plot mesh
    // Eigen::MatrixXd vertices = mng.setFile("rsc/mesh_points.csv").read<Eigen::MatrixXd>(),
    //                 indices = mng.setFile("rsc/mesh_faces.csv").read<Eigen::MatrixXd>();
    // Eigen::VectorXd fun = mng.setFile("rsc/mesh_values.csv").read<Eigen::MatrixXd>();
    // static_cast<graphics::MagnumGraphics&>(simulator.graphics())
    //     .app()
    //     .surface(vertices, fun, indices)
    //     .setTransformation(Matrix4::scaling({0.2, 0.2, 0.2}) * Matrix4::translation({2.0, -2.3, 0.0}) * Matrix4::rotationX(2.5_radf));

    // run
    // simulator.run();
    simulator.initGraphics();
    static_cast<graphics::MagnumGraphics&>(simulator.graphics())
        .app()
        .setBackground("black")
        .camera3D()
        .setPose(Vector3{1.3, 1.2, 1.});

    double t = 0.0, dt = 1e-3;
    while (true) {
        if (!simulator.step(size_t(t / dt)))
            break;
        t += dt;
    }

    return 0;
}