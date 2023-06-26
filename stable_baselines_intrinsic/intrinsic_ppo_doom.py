# Slightly modified version from SB3

import torch as th
from gymnasium import spaces
from stable_baselines_intrinsic.motivation_interface import MotivationInterface
from stable_baselines3.ppo.ppo import PPO
import numpy as np

from stable_baselines3.common.utils import obs_as_tensor

from torch.nn.utils import clip_grad_norm_
class intrinsic_PPO(PPO, MotivationInterface):
    def __init__(self, policy, env, motivation_model, motivation_lr, motivation_grad_norm,
                 intrinsic_reward_coef, extrinsic_reward_coef, warmup_steps, global_counter, learning_rate = 3e-4,
                 n_steps = 2048, batch_size = 64, n_epochs = 10, gamma = 0.99, gae_lambda = 0.95,
                 clip_range = 0.2, clip_range_vf = None, normalize_advantage = True, ent_coef = 0.0,
                 vf_coef = 0.5, max_grad_norm = 0.5, use_sde = False, sde_sample_freq = -1,
                 target_kl = None, stats_window_size = 100, tensorboard_log = None,
                 policy_kwargs = None, verbose = 0, seed = None, device = "auto",
                 motivation_device="cuda:0", _init_setup_model = True, icm_n_steps=20):
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
        self.extrinsic_reward_coef = extrinsic_reward_coef
        self.warmup_steps = warmup_steps
        self.global_counter = global_counter
        self.motivation_grad_norm = motivation_grad_norm
        self.step_counter = np.zeros(self.env.num_envs)
        self.model_optimizer = th.optim.Adam(params=motivation_model.parameters(), lr=motivation_lr)
        self.icm_n_steps = icm_n_steps

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
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            if n_steps % self.icm_n_steps == 0 and n_steps > 0:
                self.train_motivation()

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        return True

    def train(self):
        super().train()
        self.logger.record("train/grads/Agent grad norm (After clipping)", self.calculate_grad_norm(self.policy))

    def debug(self, dones, rewards):
        self.step_counter += 1
        if (rewards > 1).any():
            print("Step number: {}, global steps: {}".format(self.step_counter, self.global_counter.get_count()))
            print("Dones: {}, Rewards: {}".format(dones, rewards))