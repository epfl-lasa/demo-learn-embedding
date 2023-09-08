#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch

from learn_embedding.utils import *
from learn_embedding.dynamics import FirstGeometry
from learn_embedding.embedding import Embedding
from learn_embedding.covariances import *
from learn_embedding.approximators import *

approximator = FeedForward(3, [32, 32, 32], 1)
embedding = Embedding(approximator)
stiffness = SPD(3)
offset = torch.tensor([0.56, -0.44, 0.034])
model = FirstGeometry(embedding, torch.tensor([0.0, 0.0, 0.0]), stiffness)
TorchHelper.load(model, 'models/robotic_demo_5_1')
