import collections
import copy
import functools
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import server

import jacinle.random as random
import jacinle.io as io
import constants
import utils
import traceback
from utils import State

from difflogic.cli import format_args
from difflogic.nn.baselines import MemoryNet
from difflogic.nn.neural_logic import LogicMachine, LogitsInference, LogicLayer
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.rl.reinforce import REINFORCELoss
from difflogic.train import MiningTrainerBase
from difflogic.nn.neural_logic import InputTransform

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
from args import args


class Model(nn.Module):

  def __init__(self):
    super().__init__()

    self.feature_axis = 2
    input_dims = [0 for i in range(args.nlm_breadth + 1)]

    input_dims[2] = 5

    self.features = LogicMachine.from_args(
        input_dims, args.nlm_attributes, args, prefix='nlm')

    current_dim = self.features.output_dims[self.feature_axis]
    self.pred = LogitsInference(current_dim, 1, [])
    self.loss = REINFORCELoss()
    self.pred_loss = nn.BCELoss()

  def forward(self, feed_dict):
    feed_dict = GView(feed_dict)
    states = None
    states = feed_dict.states.float()

    def get_features(states, depth=None):
      inp = [None for i in range(args.nlm_breadth + 1)]
      inp[2] = states
      features = self.features(inp, depth=depth)
      return features

    f = get_features(states)[self.feature_axis]
    logits = self.pred(f).view(states.size(0), -1)
    policy = F.softmax(logits, dim=-1).clamp(min=1e-20)

    if self.training:
      loss, monitors = self.loss(policy, feed_dict.actions,
                                 feed_dict.discount_rewards,
                                 feed_dict.entropy_beta)
      return loss, monitors, dict()
    else:
      return dict(policy=policy, logits=logits)