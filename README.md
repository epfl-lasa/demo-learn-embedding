# demo-learn-embedding
Repositories containing source code to reproduce demonstrations in the paper...

Generate trace
```sh
apitrace trace ./build/src/examples/ik_control
```
Create frames
```sh
apitrace dump-images -o outputs/ ik_control.trace
```
