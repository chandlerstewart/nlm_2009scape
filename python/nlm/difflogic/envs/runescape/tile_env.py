"""The environment class for grid tasks."""

import copy
import numpy as np

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase
import constants
import math

__all__ = ['RunescapeEnv']




class RunescapeEnv(SimpleRLEnvBase):
  def __init__(self, bot):
    super().__init__()
    self.episode_steps = 0
    self._bot = bot


  def update_bot(self, bot):
    self._bot = bot
    self._current_state = self._bot.nearby_nodes_to_state()
    botx, boty = self._bot.get_absolute_state()

    goalx, goaly = self._absolute_to_relative(botx, boty)
    state_length = constants.GRID_SIDE_LENGTH
    self.add_state_relations()


  def _restart(self):
    def random_goal_tile():
      print("RESTARTING")

      prob = random.uniform()
      (x,y) = constants.SPAWN_LOCATION
      size = constants.NODES_RANGE*2

      if prob > 0.9:
        size *= 5*2
      elif prob > 0.8:
        size *= 4*2
      elif prob > 0.5:
        size *= 3*2
      


      if constants.GOAL_LOC is not None:
        self.goal = constants.GOAL_LOC
      else:
        self.goal = (random.randint(x-(size), x+(size)),
                    random.randint(y-(size), y+(size)))
      
      self.episode_steps = 0

      
      
    random_goal_tile()
    self._steps = 0

  def is_blocked(self, x, y):
    return self._current_state[x,y,0] != True #True is not blocked, False is blocked
  

  def add_state_relations(self):
    def get_goal_is_north():
      return self.goal[1] > Y
    
    def get_goal_is_south():
      return self.goal[1] < Y
    
    def get_goal_is_east():
      return self.goal[0] > X
    
    def get_goal_is_west():
      return self.goal[0] < X
    
    (botx, boty) = self._bot.get_absolute_state()
    X,Y = self._relative_to_absolute_env(botx, boty)

    is_north = get_goal_is_north()
    is_south = get_goal_is_south()
    is_east = get_goal_is_east()
    is_west = get_goal_is_west()

    self._current_state = np.stack([self._current_state, is_north, is_south, is_east, is_west], axis=2)

    
    return get_goal_is_north()

  def _relative_to_absolute_env(self, botx, boty):
    middle_tile = int(math.sqrt(constants.STATE_SIZE)) // 2
    env_startx = botx - middle_tile
    env_starty = boty - middle_tile
    env_endx = botx + middle_tile
    env_endy = boty + middle_tile
    X,Y = np.mgrid[env_startx:env_endx+1, env_starty:env_endy+1]
    

    return X,Y
  
  def _absolute_to_relative(self, x, y):
    middle_tile = int(math.sqrt(constants.STATE_SIZE)) // 2
    relative_x = self.goal[0] - x + middle_tile
    relative_y = self.goal[1] - y + middle_tile

    return relative_x, relative_y

  def _action(self, action):
    def reward():
        # reward based on euclidean distance from goal
      bot_loc = self._bot.get_absolute_state()
      start_dist = abs(constants.SPAWN_LOCATION[0] - self.goal[0]) \
                 + abs(constants.SPAWN_LOCATION[1] - self.goal[1])
      
      new_dist = abs(bot_loc[0] - self.goal[0]) \
          + abs(bot_loc[1] - self.goal[1])
      
      r = start_dist - new_dist

      if list(bot_loc) == list(self.goal):
        r = 10000
      
      return r
    
    def is_over():
      goal = list(self.goal)
      made_it = self._bot.get_absolute_state() == goal
      
      if made_it or self.episode_steps > constants.EPISODE_STEPS:
        if made_it:
          print("GOOOAAALLLLLLLLLLLLLL")

          return True, True #done, won
        
        return True, False

      
      return False, False

  
    middle_tile = int(math.sqrt(constants.STATE_SIZE)) // 2
    action_arr = np.full(constants.STATE_SIZE, -1)
    action_arr[action] = 1
    action_arr = action_arr.reshape((constants.GRID_SIDE_LENGTH, constants.GRID_SIDE_LENGTH))
    action_ind = np.where(action_arr == 1)
    x,y = action_ind[0][0], action_ind[1][0]
    
    
    if not self.is_blocked(x, y):
      # x,y is relative to the middle grid, so we need to translate relative back to absolute
      x += self._bot.get_absolute_state()[0] - middle_tile
      y += self._bot.get_absolute_state()[1] - middle_tile
      self._bot.take_action((x,y))
    
    r = reward()
    self.episode_steps += 1

    done, success = is_over()
    return r, (done, success)
    

    




      
