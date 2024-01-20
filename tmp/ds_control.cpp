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

#include <utils_lib/FileManager.hpp>

#include <chrono>
#include <iostream>
#include <thread>

using namespace beautiful_bullet;
using namespace utils_lib;
using namespace std::chrono;

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

    Eigen::VectorXd state_ref(7);
    state_ref << 0, 0.2, 0.1, 0.2, 0.3, 0.7, 0.2;
    franka->setState(state_ref);

    // trajectory
    FileManager mng;
    Eigen::MatrixXd traj = mng.setFile("rsc/trajectory.csv").read<Eigen::MatrixXd>();
    traj.rowwise() += Eigen::RowVector3d(0.5, -0.5, 0.5);

    // task space target
    Eigen::Vector3d xDes = traj.row(0);
    Eigen::Matrix3d oDes = (Eigen::Matrix3d() << 0.768647, 0.239631, 0.593092, 0.0948479, -0.959627, 0.264802, 0.632602, -0.147286, -0.760343).finished();
    Eigen::VectorXd state = franka->inverseKinematics(xDes, oDes, "panda_link7");

    // Set controlled robot
    (*franka)
        .setState(state)
        .activateGravity();

    // Add robots and run simulation
    simulator.add(franka);
    simulator.initGraphics();
    static_cast<graphics::MagnumGraphics&>(simulator.graphics()).app().trajectory(traj);

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
                state = franka->inverseKinematics(xDes, oDes, "panda_link7");
                franka->setState(state);
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

    std::cout << "Num rows: " << traj.rows() << std::endl;

    return 0;
}
