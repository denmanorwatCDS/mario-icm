# Used to deceit gymnasium robotics: It thinks env is GoalEnv, whilst it behaves as Env with ordinary observations

import gymnasium as gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        # Create new observation space with the new shape
        self.observation_space = env.observation_space["observation"]
        self.observation_space.dtype = np.uint8

    def observation(self, observation):
        obs = observation["observation"]

        return obs