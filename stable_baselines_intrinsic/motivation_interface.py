import torch as th
import numpy as np
from torch.nn.utils import clip_grad_norm_
from stable_baselines3.common.utils import obs_as_tensor

class MotivationInterface():
    def __init__(self):
        self.batch_for_icm = {}

        self.model_optimizer = None
        self.motivation_model = None
        self.motivation_device = None
        self.motivation_grad_norm = None

        self.global_counter = None
        self.warmup_steps = None

        self.intrinsic_reward_coef = None
        self.extrinsic_reward_coef = None

        self.logger = None

    def calculate_new_reward(self, obs, action, new_obs, rewards, dones):
        new_obs = obs_as_tensor(new_obs, self.motivation_device)
        obs, action, new_obs = (obs.to(th.float).to(self.motivation_device),
                                th.from_numpy(action).to(self.motivation_device), new_obs.to(th.float))
        self.save_batch_for_icm(obs, action, new_obs, dones)
        int_reward = np.zeros(rewards.shape)
        ext_reward = rewards
        if self.global_counter.get_count() < self.warmup_steps:
            rewards = rewards
            return rewards, 0, rewards
        else:
            int_reward = self.motivation_model.intrinsic_reward(obs, action, new_obs)
            int_reward = np.clip(int_reward, 0, 1)
            int_reward[dones == True] = 0
            rewards = int_reward * self.intrinsic_reward_coef + ext_reward * self.extrinsic_reward_coef
        return rewards, int_reward * self.intrinsic_reward_coef, ext_reward * self.extrinsic_reward_coef, int_reward, ext_reward

    def save_batch_for_icm(self, obs, action, new_obs, dones):
        relevant_obs = obs[dones == False]
        relevant_action = action[dones == False]
        relevant_new_obs = new_obs[dones == False]
        self.batch_for_icm["old obs"].append(relevant_obs)
        self.batch_for_icm["action"].append(relevant_action)
        self.batch_for_icm["new obs"].append(relevant_new_obs)

    def get_batch_for_icm(self):
        old_obs = th.concat(self.batch_for_icm["old obs"], dim=0)
        action_batch = th.concat(self.batch_for_icm["action"], dim=0)
        new_obs = th.concat(self.batch_for_icm["new obs"], dim=0)
        self.batch_for_icm["old obs"] = []
        self.batch_for_icm["action"] = []
        self.batch_for_icm["new obs"] = []
        return old_obs, action_batch, new_obs

    def train_motivation(self):
        old_obs, action_batch, new_obs = self.get_batch_for_icm()
        icm_loss = self.motivation_model.get_icm_loss(old_obs, action_batch, new_obs)
        icm_loss_value = icm_loss.clone().detach().item()
        self.model_optimizer.zero_grad()
        icm_loss.backward()
        self.logger.record("train/final/icm_loss", icm_loss_value)
        self.logger.record("train/final/forward_loss", self.motivation_model.forward_loss.item())
        self.logger.record("train/final/inverse_loss", self.motivation_model.inverse_loss.item())
        self.logger.record("train/raw/forward_loss", self.motivation_model.raw_forward_loss.item())
        self.logger.record("train/raw/inverse_loss", self.motivation_model.raw_inverse_loss.item())
        self.logger.record("train/grads/ICM grad norm (Before clipping)",
                           self.calculate_grad_norm(self.motivation_model))
        clip_grad_norm_(self.motivation_model.parameters(), self.motivation_grad_norm)
        self.model_optimizer.step()

    def calculate_grad_norm(self, model):
        total_norm = 0
        for name, p in model.named_parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

