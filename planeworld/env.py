import gym
from gym import spaces
import numpy as np

class Planeworld(gym.Env):
    metadata = {"render_modes": [None, "rgb_array"], "render_fps": 1}
    def __init__(self):
        pass
