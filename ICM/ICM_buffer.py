import numpy as np
import torch
from Config.ENV_CFG import DEVICE
from Config.A2C_CFG import NUM_AGENTS, NUM_STEPS
import pickle

class ICMBuffer():
    
    def __init__(self, sample_size = None, buffer_size = None, save_latest = True):
        if save_latest:
            self.buffer_size, self.sample_size = NUM_STEPS*NUM_AGENTS, NUM_STEPS*NUM_AGENTS
        else:
            assert buffer_size is not None and save_latest is not None, "Buffer size and save_latest must not be none!"
            self.sample_size = sample_size
            self.buffer_size = buffer_size
        self.save_as_train = True
        self.save_latest = save_latest
        self.buffer = []

    def add_triplet(self, observation, action, next_observation):
        print("Current buffer size: {}/{}".format(len(self.buffer), self.buffer_size))
        for i in range(observation.shape[0]):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((observation[i], action[i], next_observation[i]))

            elif not self.save_latest:
                swap_index = np.random.choice(self.buffer_size)
                self.buffer[swap_index] = (observation[i], action[i], next_observation[i])

            elif self.save_latest:
                del self.buffer[0]
                self.buffer.append((observation[i], action[i], next_observation[i]))

    def get_triplets(self):
        indicies = np.random.choice(len(self.buffer), size = self.sample_size)
        previous_observations, actions, next_observations = [], [], []
        for i in indicies:
            previous_observation, action, next_observation = self.buffer[i]
            previous_observations.append(previous_observation)
            actions.append(action)
            next_observations.append(next_observation)
        previous_observations = torch.stack(previous_observations)
        actions = torch.tensor(actions).to(DEVICE).to(torch.long)
        next_observations = torch.stack(next_observations)
        return previous_observations, actions, next_observations

