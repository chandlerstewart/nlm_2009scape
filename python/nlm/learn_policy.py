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
        'epochs': 400,
        'epoch_size': 100,
        'test_epoch_size': 1000,
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
        'mining_epoch_size': 3000,
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

args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.dump_play = args.dump_play and (args.dump_dir is not None)

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


def run_episode(env,
                model,
                number,
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
    
    
    bots = json_to_bots(Server.last_response)
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
  return succ, score, traj, length, optimal


class MyTrainer(MiningTrainerBase):
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
        run_episode(player, self.model, **params)
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


def json_to_bots(jsondata):
  if isinstance(jsondata, utils.Message):
    return
        
  bots = utils.json_to_bot(jsondata) 
  return bots


if __name__ == '__main__':
  Server = server.Server()
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
