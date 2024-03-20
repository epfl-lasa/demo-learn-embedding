# Learning Embedding via Demonstrations
Repositories containing source code to reproduce the robotics in demo in https://arxiv.org/abs/2403.11948

## Some reminder
Generate trace
```sh
apitrace trace ./build/src/examples/ik_control
```
Create frames
```sh
apitrace dump-images -o outputs/ ik_control.trace
```
Generate urdf from xacro
```sh
xacro [in.urdf.xacro] > [out.urdf]
```