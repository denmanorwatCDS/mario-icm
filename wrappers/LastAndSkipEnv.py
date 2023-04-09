import gym
import numpy as np
from gym import spaces

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class LastAndSkipEnv(MaxAndSkipEnv):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env, skip)

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs) -> GymObs:
        return self.env.reset(**kwargs)