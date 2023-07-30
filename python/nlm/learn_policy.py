"""The script for sudoku experiments."""

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
from model import Model
from trainer import MyTrainer
from server import Server

if args.dump_dir is not None:
  io.mkdir(args.dump_dir)
  args.log_file = os.path.join(args.dump_dir, 'log.log')
  set_output_file(args.log_file)
else:
  args.checkpoints_dir = None
  args.summary_file = None

if args.seed is not None:
  import jacinle.random as random
  random.reset_global_seed(args.seed)


from difflogic.envs.runescape import make as make_env

logger = get_logger(__file__)





#def make_data(traj, gamma):
#  Q = 0
#  discount_rewards = []
#  for reward in traj['rewards'][::-1]:
#    Q = Q * gamma + reward
#    discount_rewards.append(Q)
#  discount_rewards.reverse()
#
#  traj['states'] = as_tensor(np.array(traj['states']))
#  traj['actions'] = as_tensor(np.array(traj['actions']))
#  traj['discount_rewards'] = as_tensor(np.array(discount_rewards)).float()
#  return traj








def main(run_id):
  if args.dump_dir is not None:
    if args.runs > 1:
      args.current_dump_dir = os.path.join(args.dump_dir,
                                           'run_{}'.format(run_id))
      io.mkdir(args.current_dump_dir)
    else:
      args.current_dump_dir = args.dump_dir
    args.checkpoints_dir = os.path.join(args.current_dump_dir, 'checkpoints')
    io.mkdir(args.checkpoints_dir)
    args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')

  logger.info(format_args(args))

  model = Model()
  if args.use_gpu:
    model.cuda()
  optimizer = get_optimizer(args.optimizer, model, args.lr)
  if args.accum_grad > 1:
    optimizer = AccumGrad(optimizer, args.accum_grad)

  trainer = MyTrainer.from_args(model, optimizer, args)

  if args.load_checkpoint is not None:
    trainer.load_checkpoint(args.load_checkpoint)

  if args.test_only:
    trainer.current_epoch = 0
    return None, trainer.test()

  graduated = trainer.train()
  trainer.save_checkpoint('last')
  test_meters = trainer.test() if graduated or args.test_not_graduated else None
  return graduated, test_meters



if __name__ == '__main__':
  Server.start()

  stats = []
  nr_graduated = 0

  for i in range(args.runs):
    graduated, test_meters = main(i)
    logger.info('run {}'.format(i + 1))

    if test_meters is not None:
      for j, meters in enumerate(test_meters):
        if len(stats) <= j:
          stats.append(GroupMeters())
        stats[j].update(
            number=meters.avg['number'], test_succ=meters.avg['succ'])

      for meters in stats:
        logger.info('number {}, test_succ {}'.format(meters.avg['number'],
                                                     meters.avg['test_succ']))

    if not args.test_only:
      nr_graduated += int(graduated)
      logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
      if graduated:
        for j, meters in enumerate(test_meters):
          stats[j].update(grad_test_succ=meters.avg['succ'])
      if nr_graduated > 0:
        for meters in stats:
          logger.info('number {}, grad_test_succ {}'.format(
              meters.avg['number'], meters.avg['grad_test_succ']))
          
  Server.close()
  utils.plot(MyTrainer.succ_lst, "success_rate", "episode", "success", f"success_rate_{args.goal_loc_multi}.png")
  utils.plot(MyTrainer.rewards_lst, "rewards", "episode", "reward", f"rewards_{args.goal_loc_multi}.png")