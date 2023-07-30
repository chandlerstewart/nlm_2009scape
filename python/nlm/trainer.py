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
from difflogic.envs.runescape import make as make_env
import jacinle.random as random

from args import args
from model import Model
import numpy as np

import collections
import copy
import functools
import json
import os
import torch
from server import Server




class MyTrainer(MiningTrainerBase):
  rewards_lst = []
  succ_lst = []
  def save_checkpoint(self, name):
    if args.checkpoints_dir is not None:
      checkpoint_file = os.path.join(args.checkpoints_dir,
                                     'checkpoint_{}.pth'.format(name))
      super().save_checkpoint(checkpoint_file)

  def _dump_meters(self, meters, mode):
    if args.summary_file is not None:
      meters_kv = meters._canonize_values('avg')
      meters_kv['mode'] = mode
      meters_kv['epoch'] = self.current_epoch
      with open(args.summary_file, 'a') as f:
        f.write(io.dumps_json(meters_kv))
        f.write('\n')

  def _prepare_dataset(self, epoch_size, mode):
    pass

  def _get_player(self, number, mode):
    player = make_env(None, nr_empty=number)
    player.restart()
    return player

  def _get_result_given_player(self, index, meters, number, player, mode):
    assert mode in ['train', 'test', 'mining', 'inherit']
    params = dict(
        eval_only=True,
        number=number,
        play_name='{}_epoch{}_episode{}'.format(mode, self.current_epoch,
                                                index))
    backup = None
    if mode == 'train':
      params['eval_only'] = False
      params['entropy_beta'] = self.entropy_beta
      meters.update(lr=self.lr, entropy_beta=self.entropy_beta)
    elif mode == 'test':
      params['dump'] = True
      params['use_argmax'] = True
    else:
      backup = copy.deepcopy(player)
      params['use_argmax'] = self.is_candidate
    succ, score, traj, length, optimal = \
        run_episode(player, **params, model=self.model)
    meters.update(
        number=number, succ=succ, score=score, length=length, optimal=optimal)
    
    if mode == 'train':
      feed_dict = make_data(traj, args.gamma)
      feed_dict['entropy_beta'] = as_tensor(self.entropy_beta).float()

      if args.use_gpu:
        feed_dict = as_cuda(feed_dict)
      return feed_dict
    else:
      message = '> {} iter={iter}, number={number}, succ={succ}, \
score={score:.4f}, length={length}, optimal={optimal}'.format(
          mode, iter=index, **meters.val)
      return message, dict(succ=succ, number=number, backup=backup)

  def _extract_info(self, extra):
    return extra['succ'], extra['number'], extra['backup']

  def _get_accuracy(self, meters):
    return meters.avg['succ']

  def _get_threshold(self):
    candidate_relax = 0 if self.is_candidate else args.candidate_relax
    return super()._get_threshold() - \
        self.curriculum_thresh_relax * candidate_relax

  def _upgrade_lesson(self):
    super()._upgrade_lesson()
    # Adjust lr & entropy_beta w.r.t different lesson progressively.
    self.lr *= args.lr_decay
    self.entropy_beta *= args.entropy_beta_decay
    self.set_learning_rate(self.lr)

  def _train_epoch(self, epoch_size):
    meters = super()._train_epoch(epoch_size)

    i = self.current_epoch
    if args.save_interval is not None and i % args.save_interval == 0:
      self.save_checkpoint(str(i))
    if args.test_interval is not None and i % args.test_interval == 0:
      self.test()

    return meters

  def _early_stop(self, meters):
    t = args.early_drop_epochs
    if t is not None and self.current_epoch > t * (self.nr_upgrades + 1):
      return True
    return super()._early_stop(meters)

  def train(self):
    self.lr = args.lr
    self.entropy_beta = args.entropy_beta
    return super().train()
  


def run_episode(env,
                number,
                model,
                play_name='',
                dump=False,
                eval_only=False,
                use_argmax=False,
                need_restart=False,
                entropy_beta=0.0):
  """Run one episode using the model with $number nodes/numbers."""
  is_over = False
  traj = collections.defaultdict(list)
  score = 0
  moves = []
  # If dump_play=True, store the states and actions in a json file
  # for visualization.
  dump_play = args.dump_play and dump

  if need_restart:
    env.restart()


  optimal = abs(constants.SPAWN_LOCATION[0] - env.unwrapped.goal[0]) \
          + abs(constants.SPAWN_LOCATION[1] - env.unwrapped.goal[1])
  nodes_trajectory = []
  policies = []

  while not is_over:
    if Server.MESSAGE_IN_UPDATED:
      Server.update_message()
    if Server.STATE != State.SEND_ACTION:
      continue
    
    
    bots = utils.json_to_bot(Server.last_response)
    bot = bots[0]
    env.unwrapped.update_bot(bot)
    state = env.current_state

    feed_dict = dict(states=np.array([state]))
    feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
    feed_dict = as_tensor(feed_dict)
    if args.use_gpu:
      feed_dict = as_cuda(feed_dict)

    with torch.set_grad_enabled(not eval_only):
      output_dict = model(feed_dict)

    policy = output_dict['policy']
    p = as_numpy(policy.data[0])
    action = p.argmax() if use_argmax else random.choice(len(p), p=p)
    reward, (is_over, success) = env.action(action)

    # collect moves information
    if dump_play:
      mapped_row, mapped_col, mapped_num = env.mapping[action]
      moves.append([mapped_row, mapped_col, mapped_num])

    if reward == 0 and args.penalty is not None:
      reward = args.penalty
    succ = 1 if success else 0

    score += reward
    traj['states'].append(state)
    traj['rewards'].append(reward)
    traj['actions'].append(action)


    Server.step(utils.bot_to_json(bots))

  # dump json file storing information of playing
  if dump_play and not (args.dump_fail_only and succ):
    num = env.unwrapped.nr_empty
    json_str = json.dumps(dict(grid=cur, moves=moves))
    dump_file = os.path.join(args.current_dump_dir,
                             '{}_size{}.json'.format(play_name, num))
    with open(dump_file, 'w') as f:
      f.write(json_str)

  length = len(traj['rewards'])
  Server.STATE = State.RESET_EPISDOE
  MyTrainer.succ_lst.append(succ)
  MyTrainer.rewards_lst.append(score)
  return succ, score, traj, length, optimal


def make_data(traj, gamma):
  Q = 0
  discount_rewards = []
  for reward in traj['rewards'][::-1]:
    Q = Q * gamma + reward
    discount_rewards.append(Q)
  discount_rewards.reverse()

  traj['states'] = as_tensor(np.array(traj['states']))
  traj['actions'] = as_tensor(np.array(traj['actions']))
  traj['discount_rewards'] = as_tensor(np.array(discount_rewards)).float()
  return traj