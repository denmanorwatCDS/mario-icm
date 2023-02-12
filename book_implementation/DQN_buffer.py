from random import shuffle
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

class ExperienceReplay:
    def __init__(self, override_memory, N=500, batch_size=100):
        self.N = N
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0
        self.override_memory = override_memory

    def add_memory(self, state1, action, reward, state2):
        self.counter +=1
        if self.counter % 500 == 0:
            self.shuffle_memory()
        if len(self.memory) < self.N:
            self.memory.append( (state1, action, reward, state2) )
        elif self.override_memory:
            rand_index = np.random.randint(0,self.N-1)
            self.memory[rand_index] = (state1, action, reward, state2)

    def shuffle_memory(self):
        shuffle(self.memory)

    def set_random_ind(self):
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        if len(self.memory) < 1:
            print("Error: No data in memory.")
            return None
        self.ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
    

    def get_batch(self):
        batch = [self.memory[i] for i in self.ind] #batch is a list of tuples
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch],dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch],dim=0)
        return state1_batch, action_batch, reward_batch, state2_batch