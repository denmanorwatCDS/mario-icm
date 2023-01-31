import torchvision.transforms as T
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import wandb
import time

from Config.ENV_CFG import DEVICE

class ResizeAndGrayscale(gym.ObservationWrapper):
    def __init__(self, env, new_image_size, max_temporal_channels):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=new_image_size + (max_temporal_channels,), low=0, high=255, dtype=np.float32)
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


class ExtrinsicWrapper(gym.Wrapper):
    def __init__(self, env, fps = 10):
        super().__init__(env)
        self.environment_steps = 0
        self.logger = Logger(fps)
    
    def step(self, action):
        self.environment_steps += 1
        obs, reward, done, info = self.env.step(action)
        self.logger.log_info(obs, info, reward, reward, 
                             done, self.environment_steps)
        return obs, reward, done, info


class IntrinsicWrapper(gym.Wrapper):

    def __init__(self, env, action_space_size, connection_to_model,
    motivation_only=True, beta_coef = 0.5, fps = 10
    ):
        super().__init__(env)
        self.connection_to_model = connection_to_model
        self.motivation_only = motivation_only
        self.prev_obs = None
        self.current_gradient_step = 1
        self.environment_steps = 0
        self.BETA = beta_coef
        self.ACTION_SPACE_SIZE = action_space_size
        self.logger = Logger(fps)
        #print("Buffer id: {}".format(id(buffer)))
        #print("Motivation model id: {}".format(id(motivation_model)))
        #print("Optimizer id: {}".format(id(optimizer_of_model)))
    

    """ def __train_model(self, observation, action, next_action):
        if self.motivation_model is not None:
            start = time.time()
            predicted_actions, predicted_states, next_states =\
            self.motivation_model.forward(observation, action, next_action)
            br1 = time.time()
            CE_loss = nn.CrossEntropyLoss()
            action_one_hot = F.one_hot(action.flatten(), num_classes = self.ACTION_SPACE_SIZE)

            state_prediction_loss =\
                (1/2*(next_states-predicted_states)**2).sum(dim = 1).mean()
            action_prediction_loss =\
                CE_loss(predicted_actions, action_one_hot.argmax(dim = 1)).mean()
            icm_loss = (self.BETA*state_prediction_loss + 
                        (1-self.BETA)*action_prediction_loss)
            #self.logger.log_losses(state_prediction_loss, action_prediction_loss, icm_loss,
            #                       self.current_gradient_step)
            self.current_gradient_step += 1
            #print("ICM loss value: {}".format(icm_loss_value))
            self.optimizer_of_model.zero_grad()
            br2 = time.time()
            icm_loss.backward()
            br3 = time.time()
            self.optimizer_of_model.step()
            end = time.time()
            #print("Br1: {}".format(br1-start))
            #print("Br2: {}".format(br2-br1))
            #print("Br3: {}".format(br3-br2))
            #print("End: {}".format(end-br3)) """


    def step(self, action):
        start = time.time()
        print("Connection to model: {}".format(self.connection_to_model))
        self.environment_steps += 1
        obs, reward, done, info = self.env.step(action)
        #_save_numpy_pic(obs)
        torch_obs = self.torchify_observation(obs)
        torch_action = torch.tensor([action]).to(DEVICE)
        intrinsic_reward = 0
        unclipped_intrinsic_reward = 0
        #print(obs.shape)
        #print(torch_obs.shape)
        if self.prev_obs is not None:
            print("Message sent!")
            self.connection_to_model.send((self.prev_obs, torch_action, torch_obs))
            print("Observation sent!")
            unclipped_intrinsic_reward = self.connection_to_model.recv()
            intrinsic_reward = np.clip(unclipped_intrinsic_reward, -1, 1)
        if self.motivation_only:
            reward = intrinsic_reward
        else:
            reward += intrinsic_reward
        if self.prev_obs is not None:
            self.__train_model(self.prev_obs, torch_action, torch_obs)
        self.prev_obs = torch_obs if not done else None
        #print(3)
        #self.logger.log_info(obs, info, unclipped_intrinsic_reward, reward, 
        #                     done, self.environment_steps)
        end = time.time()
        print("Step!")
        #print("Total time of step function {}".format(end-start))
        return obs, reward, done, info

    def torchify_observation(self, observation):
        torch.cuda.empty_cache()
        observation = observation.transpose(2, 0, 1).astype(np.float32)/255.0
        print("Right way: {}".format(torch.cuda.mem_get_info()))
        observation = torch.tensor(observation).to(DEVICE)
        #_save_torch_pic(observation)
        # observation = observation.float()/255.0
        observation = torch.unsqueeze(observation, dim=0)
        return observation


class Logger():
    def __init__(self, fps):
        self.buffer = []
        self.fps = fps


    def log_losses(self, state_prediction_loss, action_prediction_loss, icm_loss,
                   current_gradient_step):
        state_value, action_value, icm_value =\
            state_prediction_loss.item(), action_prediction_loss.item(), icm_loss.item()
        wandb.log({"State prediction loss": state_value,
                       "Action prediction loss": action_value,
                       "Total icm loss": icm_value,
                       "Number of ICM gradient steps": current_gradient_step})


    def log_info(self, obs, info, unclipped_reward, reward, done, current_gradient_step):
        wandb.log({"X coordinate": info["x_pos"],
                   "Number of environment steps": current_gradient_step})
        wandb.log({"Unclipped reward": unclipped_reward,
                   "Number of environment steps": current_gradient_step})
        wandb.log({"Reward gained": reward,
                   "Number of environment steps": current_gradient_step})
        self.__save_and_log_video(obs, done)


    def __save_and_log_video(self, obs, done):
        self.buffer.append(np.expand_dims(obs[:, :, 0], 0))
        if done:
            video_sequence = np.stack(self.buffer)
            wandb.log({"Train video": wandb.Video(video_sequence, fps = self.fps, format = "gif")})
            self.buffer = []
