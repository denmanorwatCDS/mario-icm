from stable_baselines3.a2c.a2c import A2C
import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.utils import obs_as_tensor

class intrinsic_A2C(A2C):
    def __init__(self, policy, env, motivation_model, motivation_lr, intrinsic_reward_coef, learning_rate = 7e-4, n_steps = 5, 
                 gamma = 0.99, gae_lambda = 1.0, ent_coef = 0.0, vf_coef = 0.5, max_grad_norm = 0.5, rms_prop_eps = 1e-5, 
                 use_rms_prop = True, use_sde = False, sde_sample_freq = -1, normalize_advantage = False, 
                 tensorboard_log = None, create_eval_env = False, policy_kwargs = None, verbose = 0, seed = None,
                 device = "auto", _init_setup_model = True):
        super().__init__(policy, env, learning_rate, n_steps, gamma, gae_lambda, 
                       ent_coef, vf_coef, max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, 
                       sde_sample_freq, normalize_advantage, tensorboard_log, create_eval_env, policy_kwargs, verbose, seed,
                       device, _init_setup_model,)
        self.motivation_model = motivation_model
        self.batch_for_icm = {"old obs": [], "action": [], "new obs": []}
        self.intrinsic_reward_coef = intrinsic_reward_coef
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.model_optimizer = th.optim.RMSprop(params=motivation_model.parameters(), 
                                                    alpha=0.99, eps=rms_prop_eps, weight_decay=0, lr=motivation_lr)
        else:
            self.model_optimizer = th.optim.Adam(params=motivation_model.parameters(), lr=motivation_lr)
        
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps,):
        
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            rewards, int_reward, ext_reward = self.calculate_new_reward(obs_tensor, clipped_actions, new_obs, rewards)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        return True
    
    def calculate_new_reward(self, obs, action, new_obs, rewards):
        new_obs = obs_as_tensor(new_obs, self.device)
        obs, action, new_obs = obs.to(th.float), th.from_numpy(action).to(self.device), new_obs.to(th.float)
        self.save_batch_for_icm(obs, action, new_obs)
        int_reward = self.motivation_model.intrinsic_reward(obs, action, new_obs)
        ext_reward = rewards
        rewards = int_reward*self.intrinsic_reward_coef + ext_reward*(1-self.intrinsic_reward_coef)
        return rewards, int_reward, ext_reward

    def save_batch_for_icm(self, obs, action, new_obs):
        self.batch_for_icm["old obs"].append(obs)
        self.batch_for_icm["action"].append(action)
        self.batch_for_icm["new obs"].append(new_obs)

    def get_batch_for_icm(self):
        old_obs = th.concat(self.batch_for_icm["old obs"], dim=0)
        action_batch = th.concat(self.batch_for_icm["action"], dim=0)
        new_obs = th.concat(self.batch_for_icm["new obs"], dim=0)
        self.batch_for_icm["old obs"] = []
        self.batch_for_icm["action"] = []
        self.batch_for_icm["new obs"] = []
        return old_obs, action_batch, new_obs
    
    def train(self):
        super().train()
        old_obs, action_batch, new_obs = self.get_batch_for_icm()
        icm_loss = self.motivation_model.get_icm_loss(old_obs, action_batch, new_obs)
        icm_loss_value = icm_loss.detach().item()
        forward_loss, inverse_loss = self.motivation_model.get_losses(old_obs, action_batch, new_obs)
        forward_loss, inverse_loss = forward_loss.detach().item(), inverse_loss.detach().item()
        self.model_optimizer.zero_grad()
        icm_loss.backward()
        self.model_optimizer.step()
        self.logger.record("train/icm_loss", icm_loss_value)
        self.logger.record("train/forward_loss", forward_loss)
        self.logger.record("train/inverse_loss", inverse_loss)
        self.logger.record("train/ICM grad norm", self.calculate_grad_norm(self.motivation_model))
        
    def calculate_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
