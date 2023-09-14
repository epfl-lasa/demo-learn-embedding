#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch

from zmq_stream.replier import Replier
from learn_embedding.utils import *
from learn_embedding.dynamics import FirstGeometry
from learn_embedding.embedding import Embedding
from learn_embedding.covariances import *
from learn_embedding.approximators import *

approximator = FeedForward(3, [32, 32, 32], 1)
embedding = Embedding(approximator)
stiffness = SPD(3)
offset = torch.tensor([0.5, -0.5, 0.5])
model = FirstGeometry(embedding, torch.tensor([0.0, 0.0, 0.0]), stiffness)
TorchHelper.load(model, 'models/robotic_demo_5_1')

rep = Replier()
rep.configure("0.0.0.0", "5511")


def dynamics(x):
    x = torch.tensor(x[np.newaxis, :]).float().requires_grad_(True) - offset
    xdot = model(x).detach().squeeze().to(dtype=torch.float64).numpy()

    return xdot


while True:
    x = rep.reply(dynamics, np.float64, 3)
