from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
import numpy as np
from torchvision.transforms import Grayscale

class LoggerCallback(BaseCallback):
    def __init__(self, log_freq, verbose, wandb_project_name, config, global_counter, num_agents, logged_agents = 3):
        super(LoggerCallback, self).__init__(verbose)
        self.wandb_project_name = wandb_project_name
        self.config = config
        self.logged_agents = logged_agents
        self.global_counter = global_counter
        self.log_freq = log_freq
        self.num_agents = num_agents


    def _on_training_start(self):
        wandb.init(project = self.wandb_project_name, config=self.config)
        # Global stats
        self.video_array = [[] for i in range(self.logged_agents)]
        self.episode_reward = {"extrinsic": [0 for i in range(self.num_agents)], 
                               "intrinsic": [0 for i in range(self.num_agents)]}
        # Local stats
        self.local_mean_reward = {"intrinsic": [], "extrinsic": []}
        self.local_prob_of_right_action = []
        self.agent_episode = [0 for i in range(self.num_agents)]
        # Mean vest findings
        self.closest_empty_cell_by_agent = np.zeros(self.num_agents)
        self.episode_to_agent_to_vest = [[None for i in range(self.num_agents)]]
        

    def _log_local(self, ext_reward, int_reward):
        self.local_mean_reward["intrinsic"].append(int_reward)
        self.local_mean_reward["extrinsic"].append(ext_reward)
        self.log_probabilities(self.model.motivation_model, self.locals["obs_tensor"], self.locals["new_obs"], self.locals["clipped_actions"])
        if self.global_counter.get_count()%self.log_freq==0 and self.global_counter.get_count()>0:
            mean_agent_reward_int = np.array(self.local_mean_reward["intrinsic"])
            mean_agent_reward_ext = np.array(self.local_mean_reward["extrinsic"])
            for agent in range(self.logged_agents):
                wandb.log({"Dynamics/Mean intrinsic reward from {} previous steps. Agent #{}".format(self.log_freq, agent): 
                           np.mean(mean_agent_reward_int[:, agent]),
                           "Dynamics/Mean extrinsic reward from {} previous steps. Agent #{}".format(self.log_freq, agent):
                           np.mean(mean_agent_reward_ext[:, agent])}, step = self.global_counter.get_count())
            self.local_mean_reward["intrinsic"] = []
            self.local_mean_reward["extrinsic"] = []


    def log_probabilities(self, motivation_model, old_obs, new_obs, target_actions):
        with torch.no_grad():
            new_obs = torch.from_numpy(new_obs).to(torch.float32).to("cuda:0" if torch.cuda.is_available() else "cpu")
            old_obs = old_obs.to(torch.float32).to("cuda:0" if torch.cuda.is_available() else "cpu")
            probabilities = motivation_model.get_probability_distribution(old_obs, new_obs)
            mean_probability_of_right_action = probabilities[torch.arange(0, target_actions.shape[-1]), target_actions].mean().item()
        self.local_prob_of_right_action.append(mean_probability_of_right_action)
        if self.global_counter.get_count()%self.log_freq==0 and self.global_counter.get_count()>0:
            wandb.log({"mean/train/raw/mean probability of right action from {} previous steps".format(self.log_freq): 
                       np.mean(mean_probability_of_right_action)}, step = self.global_counter.get_count())
            mean_probability_of_right_action = []


    def _on_step(self):
        self.global_counter.count()
        int_rewards = self.locals["int_reward"]
        ext_rewards = self.locals["ext_reward"]
        self._log_local(ext_rewards, int_rewards)
        
        observations = self.locals["obs_tensor"][:self.logged_agents]
        train_dones = self.locals["dones"]
        infos = self.locals["infos"]
        self.update_globals(observations, int_rewards, ext_rewards)
        

        for agent_idx in range(self.num_agents):
            if train_dones[agent_idx]:
                found_vest = self.episode_reward["extrinsic"]
                self.update_found_vest(agent_idx, found_vest)
            if train_dones[agent_idx] and agent_idx < self.logged_agents:
                video_of_agent = torch.stack(self.video_array[agent_idx]).detach().cpu().numpy()
                wandb.log({"Global/Target/Agent #{} found vest ".format(agent_idx):
                           found_vest,
                           "Global/Rewards/Intrinsic reward per episode of agent #{}".format(agent_idx):
                           self.episode_reward["intrinsic"][agent_idx],
                           "Global/Rewards/Extrinsic reward per episode of agent #{}".format(agent_idx):
                           self.episode_reward["extrinsic"][agent_idx],
                           "Global/Target/Total_reward of agent #{}".format(agent_idx):
                           self.episode_reward["intrinsic"][agent_idx] + self.episode_reward["extrinsic"][agent_idx],
                           "Video/Video of train evaluation of agent #{}".format(agent_idx): 
                           wandb.Video(video_of_agent, fps=30)},
                           step = self.global_counter.get_count())
            if train_dones[agent_idx]:
                self.nullify_globals(agent_idx)
        return True


    def update_found_vest(self, agent, found_vest):
        closest_empty_cell = int(self.closest_empty_cell_by_agent[agent])
        self.closest_empty_cell_by_agent[agent] += 1
        self.episode_to_agent_to_vest[closest_empty_cell][agent] = found_vest
        if len(self.episode_to_agent_to_vest) == self.closest_empty_cell_by_agent[agent]:
            self.episode_to_agent_to_vest.append([None for i in range(self.num_agents)])
        lower_layer_filled = True
        for agent_found_vest in self.episode_to_agent_to_vest[0]:
            if agent_found_vest is None:
                lower_layer_filled = False
                break
        if lower_layer_filled:
            mean_of_findings = np.array(self.episode_to_agent_to_vest[0]).mean()
            del self.episode_to_agent_to_vest[0]
            self.closest_empty_cell_by_agent -= 1
            wandb.log({"Global/Target/Mean of findings of vest": mean_of_findings}, step = self.global_counter.get_count())
        

    def update_globals(self, observations, int_rewards, ext_rewards):
        for agent_idx in range(self.num_agents):
            self.episode_reward["intrinsic"][agent_idx] += int_rewards[agent_idx]
            self.episode_reward["extrinsic"][agent_idx] += ext_rewards[agent_idx]
            if agent_idx < self.logged_agents:
                self.video_array[agent_idx].append(observations[agent_idx][-1:])


    def nullify_globals(self, agent_idx):
        self.episode_reward["intrinsic"][agent_idx] = 0
        self.episode_reward["extrinsic"][agent_idx] = 0
        if agent_idx < self.logged_agents:
            self.video_array[agent_idx] = []
