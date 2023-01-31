import multiprocessing as mp

import asyncio
from Config import A2C_CFG
import torch
from torch import nn
import torch.functional as F

class MotivationProcess(mp.Process):
    def __init__(self, motivation_model, optimizer_of_model, connection_list):
        super().__init__()
        self.motivation_model = motivation_model
        self.optimizer_of_model = optimizer_of_model
        self.connection_list = connection_list
        self.current_gradient_step = 0

    def run(self):
        asyncio.get_event_loop().run_until_complete(self.run_logic())

    async def run_logic(self):
        print("Run started!")
        print("Readers added!")
        while True:
            agent_ready = []
            for i in range(A2C_CFG.NUM_PROCESSES):
                agent_ready.append(asyncio.Event())

                agent = self.connection_list[i]
                asyncio.get_event_loop().add_reader(agent.fileno(), agent_ready[i].set)
            print("Wait started!")
            await asyncio.gather(*[agent.wait() for agent in agent_ready])
            print("Wait ended!")

            agents_output = [data.recv() for data in self.connection_list]
            observations = [observation for observation in agents_output[0]]
            actions = [action for action in agents_output[1]]
            next_observations = [next_observation for next_observation in agents_output[2]]
            observations, actions, next_observations =\
                torch.stack(observations), torch.stack(actions), torch.stack(next_observations)
            reward = self.motivation_model.intrinsic_reward(observations, actions, next_observations)
            print(reward)
        
            for i in range(A2C_CFG.NUM_PROCESSES):
                self.connection_list[i].send(reward[i])
            
            self.grad_step(observations, actions, next_observations)
            for i in range(A2C_CFG.NUM_PROCESSES):
                agent_ready[i].clear()


    def grad_step(self, observations, actions, next_observations):
        predicted_actions, predicted_states, next_states =\
                self.motivation_model.forward(observations, actions, next_observations)
        CE_loss = nn.CrossEntropyLoss()
        action_one_hot = F.one_hot(actions.flatten(), num_classes = self.ACTION_SPACE_SIZE)

        state_prediction_loss =\
            (1/2*(next_states-predicted_states)**2).sum(dim = 1).mean()
        action_prediction_loss =\
            CE_loss(predicted_actions, action_one_hot.argmax(dim = 1)).mean()
        icm_loss = (self.BETA*state_prediction_loss + 
                        (1-self.BETA)*action_prediction_loss)

        self.current_gradient_step += 1

        self.optimizer_of_model.zero_grad()
        icm_loss.backward()
        self.optimizer_of_model.step()