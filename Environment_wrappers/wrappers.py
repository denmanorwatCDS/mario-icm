import torchvision.transforms as T
import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import torch
from torch import nn
from torch.nn import functional as F
import wandb

from Config.environment_config import ACTION_SPACE_SIZE, DEVICE
from Config.ICM_config import BETA

class ResizeAndGrayscale(gym.ObservationWrapper):
    def __init__(self, env, new_image_size, max_temporal_channels):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=new_image_size + (3,), low=0, high=255, dtype=np.uint8)
        self.new_image_size = new_image_size
        self.max_temporal_channels = max_temporal_channels
        self.observation_buffer = []
        self.is_empty = True


    def observation(self, observation):
        observation_compressed = T.ToPILImage()(observation)
        observation_compressed = T.Grayscale()(observation_compressed)
        observation_compressed = T.Resize(size = self.new_image_size)(observation_compressed)
        observation_compressed = np.array(observation_compressed)
        self.__update_buffer(observation_compressed)
        observation_in_time = self.__get_compressed_observation()

        return observation_in_time


    def __update_buffer(self, observation):
        if self.is_empty:
            self.observation_buffer =\
            [np.expand_dims(observation.copy(), axis=2) for i in range(self.max_temporal_channels)]
            self.is_empty = False
        del self.observation_buffer[0]
        self.observation_buffer.append(np.expand_dims(observation.copy(), axis=2))


    def __get_compressed_observation(self):
        return np.concatenate(self.observation_buffer, axis = 2)


    def reset_buffer(self):
        self.observation_buffer = []
        self.is_empty = True


class IntrinsicWrapper(gym.Wrapper):
    def __init__(self, env, motivation_model=None, optimizer_of_model=None,  buffer=None, 
        motivation_only=True):
        super().__init__(env)
        self.motivation_model = motivation_model
        self.optimizer_of_model = optimizer_of_model
        self.buffer = buffer
        self.motivation_only = motivation_only
        self.prev_obs = None
        self.current_gradient_step = 1
        self.logger = Logger()
    

    def __train_model(self):
        if self.motivation_model is not None:
            observations, actions, next_observations = self.buffer.get_triplets()
            predicted_actions, predicted_states, next_states =\
            self.motivation_model.forward(observations, actions, next_observations)
            CE_loss = nn.CrossEntropyLoss()
            action_one_hot = F.one_hot(actions.flatten(), num_classes = ACTION_SPACE_SIZE)

            state_prediction_loss =\
                (1/2*(next_states-predicted_states)**2).sum(dim = 1).mean()
            action_prediction_loss =\
                CE_loss(predicted_actions, action_one_hot.argmax(dim = 1)).mean()
            icm_loss =(BETA*state_prediction_loss + (1-BETA)*action_prediction_loss)

            self.logger.log_losses(state_prediction_loss, action_prediction_loss, icm_loss,
                                   self.current_gradient_step)
            self.current_gradient_step += 1
            #print("ICM loss value: {}".format(icm_loss_value))
            self.optimizer_of_model.zero_grad()
            icm_loss.backward()
            self.optimizer_of_model.step()


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torch_obs = torchify_observation(obs)
        torch_action = torch.tensor([action]).to(DEVICE)
        intrinsic_reward = 0
        if self.prev_obs is not None:
            self.buffer.add_triplet(self.prev_obs, torch_action, torch_obs)
            #print("Previous observation shape: {}".format(self.prev_obs.shape))
            #print("Action shape: {}".format(torch_action.shape))
            #print("New observation shape: {}".format(torch_obs.shape))
            intrinsic_reward =\
                self.motivation_model.intrinsic_reward(self.prev_obs, torch_action, torch_obs)
            #print("ICM reward: {}".format(intrinsic_reward))
            #print("Shape of ICM reward {}".format(intrinsic_reward.shape))
        if self.motivation_only:
            reward = intrinsic_reward
        else:
            reward += intrinsic_reward
        if self.prev_obs is not None:
            self.__train_model()
        self.prev_obs = torch_obs if not done else None
        self.logger.log_info(obs, info, reward, done, self.current_gradient_step)
        return obs, reward, done, info


def torchify_observation(observation):
    observation = observation.transpose(2, 0, 1)
    observation = torch.tensor(observation).to(DEVICE)
    observation = observation.to(torch.float32)
    observation = torch.unsqueeze(observation, dim=0)
    return observation

class JoypadSpaceWithObservation(JoypadSpace):
    def __init__(self, env, actions):
        super().__init__(env, actions)
        self.state = self.env.env.state

    def step(self, action):
        return_value = super().step(action)
        self.state = self.env.state


class MaxAndSkipEnvWithObservation(MaxAndSkipEnv):
    def __init__(self, env, skip=4):
        super().__init__(env, skip)
        self.state = self.env.state

    def step(self, action):
        return_value = super().step(action)
        self.state = self.env.state

class Logger():
    def __init__(self):
        self.buffer = []


    def log_losses(self, state_prediction_loss, action_prediction_loss, icm_loss,
                   current_gradient_step):
        state_value, action_value, icm_value =\
            state_prediction_loss.item(), action_prediction_loss.item(), icm_loss.item()
        wandb.log({"State prediction loss": state_value,
                       "Action prediction loss": action_value,
                       "Total icm loss": icm_value,
                       "Number of ICM gradient steps": current_gradient_step})


    def log_info(self, obs, info, reward, done, current_gradient_step):
        wandb.log({"X coordinate": info["x_pos"],
                   "Number of ICM gradient steps": current_gradient_step})
        wandb.log({"Reward gained": reward,
                   "Number of ICM gradient steps": current_gradient_step})
        self.__save_and_log_video(obs, done)


    def __save_and_log_video(self, obs, done):
        self.buffer.append(np.expand_dims(obs[:, :, 0], 0))
        if done:
            video_sequence = np.stack(self.buffer)
            wandb.log({"Train video": wandb.Video(video_sequence, fps = 10, format = "gif")})
            self.buffer = []
