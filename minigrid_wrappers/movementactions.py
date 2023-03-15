import gym
import numpy as np
from gym.spaces import Discrete, Box

class MovementActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )
    
    def step(self, action):
        TURN_LEFT = 0
        FORWARD = 2

        new_obs, _, _, _ = self.env.step(TURN_LEFT)
        direction = new_obs["direction"]
        while direction != action:
            new_obs, _, _, _ = self.env.step(TURN_LEFT)
            direction = new_obs["direction"]
        new_obs, rewards, dones, infos = self.env.step(FORWARD)
        new_obs = np.array(new_obs["image"])
        return new_obs, rewards, dones, infos

    def reset(self):
        obs = super().reset()
        return np.array(obs["image"])
