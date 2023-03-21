from environments.crossing_limited import FourRoomsEnvLimited
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, FlatObsWrapper, StateBonus
from PIL import Image
from minigrid_wrappers.imagedirection import RGBImgObsDirectionWrapper
from minigrid_wrappers.movementactions import MovementActions
from gym.wrappers import TimeLimit, ResizeObservation
from stable_baselines3.common.atari_wrappers import WarpFrame

from gym.wrappers import GrayScaleObservation
import gym

grid_size = 15
env = gym.make("MiniGrid-DoorKey-8x8-v0")
env = RGBImgObsDirectionWrapper(env, grid_size, restriction=None)
env = MovementActions(env)
env = TimeLimit(env, 100)

env = WarpFrame(env, width=126, height=126)
obs_start = env.reset()
#obs_next, *_ = env.step(0)
#obs_next, *_ = env.step(2)
#obs_next, *_ = env.step(1)
obs_next, rewards, *_ = env.step(1)
obs_next, rewards, *_ = env.step(1)
obs_next, rewards, *_ = env.step(1)
obs_next, rewards, *_ = env.step(1)
obs_next, rewards, *_ = env.step(2)
#obs_next, *_ = env.step(0)
#obs = env.step()
print(obs_start)
print(obs_start.shape)
print(obs_next.shape)

im = Image.fromarray(obs_start.squeeze(), mode='L')
im.save("obs_start.jpeg")
im = Image.fromarray(obs_next.squeeze(), mode='L')
im.save("obs_next.jpeg")
