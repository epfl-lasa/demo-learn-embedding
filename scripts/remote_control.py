#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np
import torch
import yaml

from zmq_stream.replier import Replier
from learn_embedding.utils import *
from learn_embedding.dynamics import FirstGeometry
from learn_embedding.embedding import Embedding
from learn_embedding.covariances import *
from learn_embedding.approximators import *

demo_number = sys.argv[1] if len(sys.argv) == 2 else "1"

# load params
with open("rsc/demos/demo_" + demo_number + "/dynamics_params.yaml", "r") as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.SafeLoader)["dynamics"]

# model
approximator = FeedForward(params['dimension'], [params['approximator']['num_neurons']]*params['approximator']['num_layers'], 1)
embedding = Embedding(approximator)
stiffness = SPD(params['dimension'])
model = FirstGeometry(embedding, torch.tensor([0.0, 0.0, 0.0]), stiffness)
TorchHelper.load(model, "rsc/demos/demo_" + demo_number + "/dynamics_model")

# callback
offset = torch.from_numpy(np.loadtxt("rsc/demos/demo_" + demo_number + "/offset.csv")).float()


def dynamics(x):
    x = torch.tensor(x[np.newaxis, :]).float().requires_grad_(True) - offset
    return model(x).detach().squeeze().to(dtype=torch.float64).numpy()


# communicator
rep = Replier()
rep.configure("0.0.0.0", "5511")
while True:
    x = rep.reply(dynamics, np.float64, 3)
