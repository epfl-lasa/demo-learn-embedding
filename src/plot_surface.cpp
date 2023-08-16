/*
    This file is part of graphics-lib.

    Copyright (c) 2020, 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>

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

#include <graphics_lib/Graphics.hpp>
#include <utils_lib/FileManager.hpp>

#include <filesystem>
namespace fs = std::filesystem;

using namespace graphics_lib;
using namespace utils_lib;

int main(int argc, char** argv)
{
    Graphics app({argc, argv});

    // Load mesh
    FileManager io_manager;
    Eigen::MatrixXd vertices = io_manager.setFile("rsc/mesh.obj").read<Eigen::MatrixXd>("$Nodes", 1),
                    indices = io_manager.read<Eigen::MatrixXd>("$Elements", 1).array() - 1;

    Eigen::VectorXd fun = Eigen::VectorXd::Random(vertices.rows());

    // std::filesystem::path p = "foo.c";
    // std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    // std::cout << "Absolute path for " << p << " is " << fs::absolute(p) << '\n';

    // app.import("rsc/mesh.vtk");

    app
        .setBackground("white")
        .surface(vertices, fun, indices)
        .setTransformation(Matrix4::scaling({0.05, 0.05, 0.05}));

    return app.exec();
}