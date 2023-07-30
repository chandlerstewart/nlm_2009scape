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




parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm', 'memnet'],
    help='model choices, nlm: Neural Logic Machine, memnet: Memory Networks')

# NLM parameters, works when model is 'nlm'.
nlm_group = parser.add_argument_group('Neural Logic Machines')
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 8,
        'breadth': 2,
        'residual': True,
        'exclude_self': False,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=16,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

# MemNN parameters, works when model is 'memnet'.
memnet_group = parser.add_argument_group('Memory Networks')
MemoryNet.make_memnet_parser(memnet_group, {}, prefix='memnet')

data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
  '--gen-grid-nr-empty',
  type=int,
  default=1,
  metavar='N',
  help='number of empty cell inside grid'
)
data_gen_group.add_argument(
  '--gen-grid-dim',
  type=int,
  default=9,
  metavar='N',
  help='dimension of grid'
)
# data_gen_group.add_argument(


MiningTrainerBase.make_trainer_parser(
    parser, {
        'epochs': 100,
        'epoch_size': 100,
        #'test_epoch_size': 1000,
        'test_epoch_size': 1,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
        'curriculum_start': 3,
        'curriculum_step': 1,
        'curriculum_graduate': 12,
        'curriculum_thresh_relax': 0.005,
        'sample_array_capacity': 3,
        'enable_mining': True,
        'mining_interval': 6,
        'mining_epoch_size': 1,
        #'mining_epoch_size': 3000,
        'mining_dataset_size': 300,
        'inherit_neg_data': True,
        'prob_pos_data': 0.5
    })

train_group = parser.add_argument_group('Train')
train_group.add_argument('--seed', type=int, default=None, metavar='SEED')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=0.9,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient (default: 1)')
train_group.add_argument(
    '--candidate-relax',
    type=int,
    default=0,
    metavar='N',
    help='number of thresh relaxation for candidate')

rl_group = parser.add_argument_group('Reinforcement Learning')
rl_group.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='F',
    help='discount factor for accumulated reward function in reinforcement learning'
)
rl_group.add_argument(
    '--penalty',
    type=float,
    default=-0.01,
    metavar='F',
    help='a small penalty each step')
rl_group.add_argument(
    '--entropy-beta',
    type=float,
    default=0.1,
    metavar='F',
    help='entropy loss scaling factor')
rl_group.add_argument(
    '--entropy-beta-decay',
    type=float,
    default=0.8,
    metavar='F',
    help='entropy beta exponential decay factor')

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--dump-play',
    action='store_true',
    help='dump the trajectory of the plays for visualization')
io_group.add_argument(
    '--dump-fail-only', action='store_true', help='dump failure cases only')
io_group.add_argument(
    '--load-checkpoint',
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--early-drop-epochs',
    type=int,
    default=40,
    metavar='N',
    help='epochs could spend for each lesson, early drop')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')
schedule_group.add_argument(
    '--test-not-graduated',
    action='store_true',
    help='test not graduated models also')

args = parser.parse_args()

args.test_not_graduated = True
args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.dump_play = args.dump_play and (args.dump_dir is not None)
args.goal_loc_multi = 2