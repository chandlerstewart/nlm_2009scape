import math

HOST = 'localhost'
PORT = 5000

NUM_BOTS = 1
NUM_EPISDOES = 1000


EPISODE_NUM_STEPS_MIN = 20
EPISODE_NUM_STEPS_MAX = 20
EPISODE_STEPS = 50


NODES_RANGE = 2
STATE_SIZE = (NODES_RANGE*2 + 1)** 2 #x,y,free space, nearby nodes
GRID_SIDE_LENGTH = int(math.sqrt(STATE_SIZE))

ACTION_SIZE = 4 #move in each direction, interact in each direction

SPAWN_LOCATION = (2727, 3478)

#GOAL_LOC = (2737, 3478)
GOAL_LOC = None

MODEL_SAVE_PATH = "./models/model.pt"

