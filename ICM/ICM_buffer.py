import numpy as np
import torch
import pickle
from torch import nn
from Config.ENV_CFG import DEVICE


class ICMBuffer():
    
    def __init__(self, sample_size, buffer_size):
        self.sample_size = sample_size
        self.buffer_size = buffer_size
        self.save_as_train = True
        self.buffer = []

    def add_triplet(self, observation, action, next_observation):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((observation, action, next_observation))
        else:
            if self.save_as_train:
                self.__save_as_train()
                self.save_as_train=False

            swap_index = np.random.choice(self.buffer_size)
            self.buffer[swap_index] = (observation, action, next_observation)

    def get_triplets(self):
        indicies = np.random.choice(len(self.buffer), size = self.sample_size)
        #print("Random indicies are: {}".format(indicies))
        previous_observations, actions, next_observations = [], [], []
        for i in indicies:
            previous_observation, action, next_observation = self.buffer[i]
            previous_observations.append(previous_observation)
            actions.append(action)
            next_observations.append(next_observation)
        previous_observations = torch.cat(previous_observations, dim=0)
        actions = torch.tensor(actions).to(DEVICE).to(torch.long)
        next_observations = torch.cat(next_observations, dim=0)
        return previous_observations, actions, next_observations

    def __save_as_train(self):
        path_to_file = "/home/dvasilev/mario_icm/debug/train_set/pickled_train_list.pkl"
        with open(path_to_file, "wb") as train_loader_file:
            pickle.dump(self.buffer, train_loader_file)

