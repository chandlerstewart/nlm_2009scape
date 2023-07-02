"""Quick access for graph environments."""

from jaclearn.rl.proxy import LimitLengthProxy

from .tile_env import RunescapeEnv
from ..utils import get_action_mapping_2009scape
from ..utils import MapActionProxy

__all__ = ['get_tile_env', 'make']


def get_tile_env(nr_empty):
  env_cls = RunescapeEnv
  env = env_cls(nr_empty)
  p = LimitLengthProxy(env, 400)
  mapping = get_action_mapping_2009scape()
  # print(mapping)
  p = MapActionProxy(p, mapping)
  return p

def make(task, *args, **kwargs):
  return get_tile_env(*args, **kwargs)
