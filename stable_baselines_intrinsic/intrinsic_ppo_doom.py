import torch as th
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.ppo.ppo import PPO
import numpy as np

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.utils import obs_as_tensor

from torch.nn.utils import clip_grad_norm_
class intrinsic_PPO(PPO):
    def __init__(self, policy, env, motivation_model, motivation_lr, motivation_grad_norm,
                 intrinsic_reward_coef, warmup_steps, global_counter, learning_rate = 3e-4,
                 n_steps = 2048, batch_size = 64, n_epochs = 10, gamma = 0.99, gae_lambda = 0.95,
                 clip_range = 0.2, clip_range_vf = None, normalize_advantage = True, ent_coef = 0.0,
                 vf_coef = 0.5, max_grad_norm = 0.5, use_sde = False, sde_sample_freq = -1,
                 target_kl = None, stats_window_size = 100, tensorboard_log = None,
                 policy_kwargs = None, verbose = 0, seed = None, device = "auto",
                 motivation_device="cuda:0", _init_setup_model = True,):
        super().__init__(policy=policy,
                     env=env,
                     learning_rate=learning_rate,
                     n_steps=n_steps,
                     batch_size=batch_size,
                     n_epochs=n_epochs,
                     gamma=gamma,
                     gae_lambda=gae_lambda,
                     clip_range=clip_range,
                     clip_range_vf=clip_range_vf,
                     normalize_advantage = normalize_advantage,
                     ent_coef=ent_coef,
                     vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm,
                     use_sde=use_sde,
                     sde_sample_freq=sde_sample_freq,
                     target_kl=target_kl,
                     stats_window_size=stats_window_size,
                     tensorboard_log=tensorboard_log,
                     policy_kwargs=policy_kwargs,
                     verbose=verbose,
                     seed=seed,
                     device=device,
                     _init_setup_model=_init_setup_model,)

        self.motivation_model = motivation_model
        self.motivation_device = motivation_device
        self.batch_for_icm = {"old obs": [], "action": [], "new obs": []}
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.warmup_steps = warmup_steps
        self.global_counter = global_counter
        self.motivation_grad_norm = motivation_grad_norm
        self.step_counter = np.zeros(self.env.num_envs)
        self.model_optimizer = th.optim.Adam(params=motivation_model.parameters(), lr=motivation_lr)

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
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
            self.debug(dones, rewards)
            # Reset doom reward

            rewards, int_reward, ext_reward, raw_int_reward, raw_ext_reward =\
                self.calculate_new_reward(obs_tensor, clipped_actions, new_obs, rewards, dones)

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

    def train(self):
        super().train()
        old_obs, action_batch, new_obs = self.get_batch_for_icm()
        icm_loss = self.motivation_model.get_icm_loss(old_obs, action_batch, new_obs)
        icm_loss_value = icm_loss.detach().item()
        self.model_optimizer.zero_grad()
        icm_loss.backward()
        self.logger.record("train/final/icm_loss", icm_loss_value)
        self.logger.record("train/final/forward_loss", self.motivation_model.forward_loss.item())
        self.logger.record("train/final/inverse_loss", self.motivation_model.inverse_loss.item())
        self.logger.record("train/raw/forward_loss", self.motivation_model.raw_forward_loss.item())
        self.logger.record("train/raw/inverse_loss", self.motivation_model.raw_inverse_loss.item())
        self.logger.record("train/grads/ICM grad norm (Before clipping)",
                           self.calculate_grad_norm(self.motivation_model))
        self.logger.record("train/grads/A2C grad norm (After clipping)", self.calculate_grad_norm(self.policy))
        clip_grad_norm_(self.motivation_model.parameters(), self.motivation_grad_norm)
        self.model_optimizer.step()

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
            int_reward[dones==True] = 0
            rewards = int_reward * self.intrinsic_reward_coef + ext_reward
        return rewards, int_reward * self.intrinsic_reward_coef, ext_reward, int_reward, ext_reward

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

    def calculate_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def debug(self, dones, rewards):
        self.step_counter += 1
        if (rewards > 1).any():
            print("Step number: {}, global steps: {}".format(self.step_counter, self.global_counter.get_count()))
            print("Dones: {}, Rewards: {}".format(dones, rewards))