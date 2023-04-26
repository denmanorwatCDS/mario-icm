from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import pickle
from torchvision.transforms import Grayscale


class LoggerCallback(BaseCallback):
    def __init__(self, verbose, wandb_project_name, config, global_counter, quantity_of_agents, device,
                 log_frequency=500, video_submission_frequency=10, fps=30):
        super(LoggerCallback, self).__init__(verbose)
        self.wandb_project_name = wandb_project_name
        self.config = config
        self.global_counter = global_counter
        self.log_frequency = log_frequency
        self.video_submission_frequency = video_submission_frequency
        self.quantity_of_agents = quantity_of_agents
        self.device = device
        self.fps = fps
        self.gray = Grayscale(1)

        # Theese are calculated on each step
        self.step_characteristics = {"Media/Agent #0 observations": [],
                                     "Media/Environment state of agent #0": [],
                                     "mean/train/raw/Estimated probability of ground truth action": [],
                                     "Raw/Intrinsic reward of agent #0": [], "Raw/Extrinsic reward of agent #0": [],
                                     "Mix/Intrinsic reward of agent #0": [], "Mix/Extrinsic reward of agent #0": [],
                                     "Mix/Total reward of agent #0": [],
                                     "Raw/Intrinsic reward": [], "Raw/Extrinsic reward": [],
                                     "Raw/Mean episode steps of agent #0": [], "Raw/Mean episode steps": [],
                                     "Mix/Intrinsic reward": [], "Mix/Extrinsic reward": [],
                                     "Mix/Total reward": []}

        # Simple characteristics - this characteristics can be logged after mean operation
        self.simple_characteristics = ["Raw/Intrinsic reward of agent #0", "Raw/Extrinsic reward of agent #0",
                                       "Mix/Intrinsic reward of agent #0", "Mix/Extrinsic reward of agent #0",
                                       "Mix/Total reward of agent #0",
                                       "mean/train/raw/Estimated probability of ground truth action",
                                       "Raw/Intrinsic reward", "Raw/Extrinsic reward",
                                       "Raw/Mean episode steps of agent #0", "Raw/Mean episode steps",
                                       "Mix/Intrinsic reward", "Mix/Extrinsic reward", "Mix/Total reward"]
        # Videos array
        self.video_names = ["Media/Agent #0 observations", "Media/Environment state of agent #0"]
        self.fully_observable_frames = []
        self.episode_of_agent_0_counter = 0

        # Probability of performing ground truth action
        self.mean_probabilites_from_step = []

        # Set of all visited states through train
        self.all_visited_states = set()

        # Previous time
        self.previous_time = None

        # Episode lengths
        self.episode_lengths = np.zeros(self.quantity_of_agents)

    def _on_training_start(self):
        print(self.model.device)

    def log_timer(self, log):
        current_time = time.perf_counter()
        if self.previous_time is not None:
            log["Performance/Step time"] = current_time - self.previous_time
            log["Elapsed time"] = current_time - self.previous_time
        self.previous_time = current_time

    def update_estimated_probability_of_ground_truth(self, old_obs, new_obs, performed_actions):
        motivation_model = self.model.motivation_model
        old_obs, new_obs = (old_obs.to(torch.float32).to(self.model.motivation_device),
                            new_obs.to(torch.float32).to(self.model.motivation_device))
        predicted_probabilities = motivation_model.get_probability_distribution(old_obs, new_obs)
        self.step_characteristics["mean/train/raw/Estimated probability of ground truth action"].append(
            predicted_probabilities[torch.arange(0, performed_actions.shape[-1]), performed_actions].mean().item())

    def update_mean_episode_length(self, dones):
        self.episode_lengths[dones==0] += 1
        for episode_length in self.episode_lengths[dones==1]:
            self.step_characteristics["Raw/Mean episode steps"].append(episode_length)
        if dones[0] == 1:
            self.step_characteristics["Raw/Mean episode steps of agent #0"].append(self.episode_lengths[0])
        self.episode_lengths[dones==1] = 0

    def append_characteristrics(self, **kwargs):
        for statistic, value in kwargs.items():
            self.step_characteristics[statistic].append(value)

    def process_mean_characteristics(self, log):
        for simple_characteristic in self.simple_characteristics:
            print(simple_characteristic)
            if simple_characteristic == "Raw/Mean episode steps":
                print(self.step_characteristics[simple_characteristic])
            log[simple_characteristic] = np.mean(self.step_characteristics[simple_characteristic])
            self.step_characteristics[simple_characteristic] = []

    def log_video_if_ready(self, env_state, done, agent_obs=None):
        logs = {}
        frame = env_state[0]
        self.step_characteristics["Media/Environment state of agent #0"].append(frame.cpu())
        if agent_obs is not None:
            self.step_characteristics["Media/Agent #0 observations"].append(agent_obs.cpu()[0][-1:])
        if done:
            for name in self.video_names:
                if self.episode_of_agent_0_counter % self.video_submission_frequency == 0:
                    if len(self.step_characteristics[name]) > 0:
                        video = np.stack(self.step_characteristics[name])
                        if name == "Media/Environment state of agent #0":
                            logs[name] = wandb.Video(video, fps=self.fps)
                        else:
                            logs[name] = wandb.Video(video)
                self.step_characteristics[name] = []
            self.episode_of_agent_0_counter += 1
        if len(logs) > 0:
            wandb.log(logs, step=self.global_counter.get_count())

    def _on_step(self):
        # Gather logging info
        self.global_counter.count()
        raw_int_rewards, raw_ext_rewards = self.locals["raw_int_reward"], self.locals["raw_ext_reward"]
        int_rewards, ext_rewards = self.locals["int_reward"], self.locals["ext_reward"]
        total_rewards = int_rewards + ext_rewards
        raw_int_reward_agent_0, raw_ext_reward_agent_0 = raw_int_rewards[0], raw_ext_rewards[0]
        int_reward_agent_0, ext_reward_agent_0 = int_rewards[0], ext_rewards[0]
        total_reward_agent_0 = total_rewards[0]
        reward_info = {"Raw/Intrinsic reward of agent #0": raw_int_reward_agent_0,
                       "Raw/Extrinsic reward of agent #0": raw_ext_reward_agent_0,
                       "Mix/Intrinsic reward of agent #0": int_reward_agent_0,
                       "Mix/Extrinsic reward of agent #0": ext_reward_agent_0,
                       "Mix/Total reward of agent #0": total_reward_agent_0,
                       "Raw/Intrinsic reward": raw_int_rewards.mean(), "Raw/Extrinsic reward": raw_ext_rewards.mean(),
                       "Mix/Intrinsic reward": int_rewards.mean(), "Mix/Extrinsic reward": ext_rewards.mean(),
                       "Mix/Total reward": total_rewards.mean()}
        self.append_characteristrics(**reward_info)

        previous_observation, current_observation = \
            self.locals["obs_tensor"], \
                torch.as_tensor(self.locals["new_obs"]).to(self.device)
        actions = self.locals["clipped_actions"]
        self.update_estimated_probability_of_ground_truth(previous_observation, current_observation, actions)

        done = self.locals["dones"][0]
        self.log_video_if_ready(previous_observation, done, None)

        dones = self.locals["dones"]
        self.update_mean_episode_length(dones)

        if self.global_counter.get_count() % self.log_frequency == 0 and self.global_counter.get_count() > 0:
            log = {}
            self.process_mean_characteristics(log)
            self.log_timer(log)
            wandb.log(log, step=self.global_counter.get_count())
