#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np
import torch
import yaml
import time

from zmq_stream.replier import Replier
from learn_embedding.utils import *
from learn_embedding.dynamics import FirstGeometry
from learn_embedding.embedding import Embedding
from learn_embedding.covariances import *
from learn_embedding.approximators import *

demo_number = sys.argv[1] if len(sys.argv) == 2 else "1"

# load params
with open("rsc/demos/demo_" + demo_number + "/dynamics_params.yaml", "r") as yamlfile:
    p = yaml.load(yamlfile, Loader=yaml.SafeLoader)

# model
use_cuda = torch.cuda.is_available()
device = "cpu" # torch.device("cuda" if use_cuda else "cpu")
if p['first_order']['embedding']['type'] == "network":
    approximator = FeedForward(p['dimension'], 
                               [p['first_order']['embedding']['params'][0]]*p['first_order']['embedding']['params'][1], 1)
embedding = Embedding(approximator)
if p['first_order']['stiffness'] == "full":
    stiffness = SPD(p['dimension'])
model = FirstGeometry(embedding, torch.zeros(p["dimension"]).to(device), stiffness).to(device)
TorchHelper.load(model, "rsc/demos/demo_" + demo_number + "/models/"+p['first_order']['name'], device)

# callback
offset = torch.from_numpy(np.array(p["offset"])).float().to(device)
def dynamics(x):
    t0 = time.time()
    x = torch.tensor(x[np.newaxis, :]).float().requires_grad_(True).to(device) - offset
    y = model(x).cpu().detach().squeeze().to(dtype=torch.float64).numpy()
    print(time.time()-t0)
    return y

# communicator
rep = Replier()
rep.configure("0.0.0.0", "5511")
while True:
    x = rep.reply(dynamics, np.float64, 3)
