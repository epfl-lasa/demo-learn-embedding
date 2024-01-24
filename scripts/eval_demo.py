#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import yaml
import matplotlib.pyplot as plt
from dtaidistance import dtw_ndim


def subsample(x, num_samples):
    total_length = np.sum(np.linalg.norm(x[1:] - x[:-1], axis=1))
    average_distance = total_length / (num_samples - 1)
    idx = [0]

    for j in range(1, x.shape[0]):
        distance = np.linalg.norm(x[j] - x[idx[-1]])
        if distance >= average_distance:
            idx.append(j)

    if x.shape[0] - 1 not in idx:
        idx = np.append(idx, x.shape[0] - 1)

    return np.array(idx)


demo_name = "demo_ik"

with open("rsc/demos/demo_2/dynamics_params.yaml", "r") as yamlfile:
    offset = np.array(yaml.load(yamlfile, Loader=yaml.SafeLoader)["offset"])

dtwd = np.zeros(3)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i in range(5, 8):
    traj_1 = np.loadtxt("rsc/demos/demo_2/trajectory_"+str(i)+".csv") + offset
    idx = subsample(traj_1, 1000)
    traj_1 = traj_1[idx]
    traj_2 = np.loadtxt(demo_name+"_"+str(i)+".csv")
    idx = subsample(traj_2, 1000)
    traj_2 = traj_2[idx]
    dtwd[i-5] = dtw_ndim.distance(traj_1, traj_2)
    ax.scatter(traj_1[:, 0], traj_1[:, 1], traj_1[:, 2], c="r")
    ax.scatter(traj_2[:, 0], traj_2[:, 1], traj_2[:, 2], c="b")

print(np.round(dtwd.mean(), 2), "\pm", np.round(dtwd.std(), 2))
plt.show()
