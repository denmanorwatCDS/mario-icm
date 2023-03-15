from gym_minigrid.envs import FourRoomsEnv
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, FlatObsWrapper
from PIL import Image
from minigrid_wrappers.imagedirection import RGBImgObsDirectionWrapper, FullyObsDirectionWrapper
from minigrid_wrappers.movementactions import MovementActions
from gym.wrappers import TimeLimit, ResizeObservation

from gym.wrappers import GrayScaleObservation
import gym

env = gym.make("MiniGrid-FourRooms-v0")
env = MovementActions(env)
env = FullyObsDirectionWrapper(env)
env = RGBImgObsDirectionWrapper(env)
env = TimeLimit(env, 1000)



