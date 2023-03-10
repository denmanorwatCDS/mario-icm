from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
import numpy as np

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
        self.episode_max_x_pos = [-1 for i in range(self.num_agents)]
        # Local stats
        self.local_mean_reward = {"intrinsic": [], "extrinsic": []}
        self.local_max_x_pos = np.array([-1 for i in range(self.num_agents)])
        self.local_prob_of_right_action = []
        # Mean x_pos by episode
        self.agent_episode = [0 for i in range(self.num_agents)]
        self.x_positions = [[None for i in range(self.num_agents)]]
        

    def _log_local(self, x_pos, ext_reward, int_reward):
        self.local_max_x_pos = np.max(np.stack((self.local_max_x_pos, x_pos)), axis=0)
        self.local_mean_reward["intrinsic"].append(int_reward)
        self.local_mean_reward["extrinsic"].append(ext_reward)
        self.log_probabilities(self.model.motivation_model, self.locals["obs_tensor"], self.locals["new_obs"], self.locals["actions"])
        if self.global_counter.get_count()%self.log_freq==0 and self.global_counter.get_count()>0:
            mean_agent_reward_int = np.array(self.local_mean_reward["intrinsic"])
            mean_agent_reward_ext = np.array(self.local_mean_reward["extrinsic"])
            for agent in range(self.logged_agents):
                wandb.log({"Dynamics/Max x pos from {} previous steps. Agent #{}".format(self.log_freq, agent): 
                           self.local_max_x_pos[agent],
                           "Dynamics/Mean intrinsic reward from {} previous steps. Agent #{}".format(self.log_freq, agent): 
                           np.mean(mean_agent_reward_int[:, agent]),
                           "Dynamics/Mean extrinsic reward from {} previous steps. Agent #{}".format(self.log_freq, agent):
                           np.mean(mean_agent_reward_ext[:, agent])}, step = self.global_counter.get_count())
        self.local_max_x_pos = np.zeros(self.local_max_x_pos.shape)-1
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
        x_pos = np.array([info["x_pos"] for info in self.locals["infos"]])
        self._log_local(x_pos, ext_rewards, int_rewards)
        
        observations = self.locals["obs_tensor"][:self.logged_agents]
        train_dones = self.locals["dones"]
        self.update_globals(observations, int_rewards, ext_rewards, x_pos)

        for agent_idx in range(self.num_agents):
            if train_dones[agent_idx]:
                self.log_mean_x_pos_by_episode(agent_idx, self.episode_max_x_pos[agent_idx])
                
        for agent_idx in range(self.logged_agents):
            if train_dones[agent_idx] and agent_idx < self.logged_agents:
                video_of_agent = torch.stack(self.video_array[agent_idx]).detach().cpu().numpy()
                wandb.log({"Global/Extrisnic reward per episode of agent #{}".format(agent_idx):
                           self.episode_reward["extrinsic"][agent_idx],
                           "Global/Intrinsic reward per episode of agent #{}".format(agent_idx):
                           self.episode_reward["intrinsic"][agent_idx],
                           "Global/Max x pos per episode of agent #{}".format(agent_idx):
                           self.episode_max_x_pos[agent_idx],
                           "Video/Video of train evaluation of agent #{}".format(agent_idx): wandb.Video(video_of_agent, fps=30)},
                           step = self.global_counter.get_count())
            if train_dones[agent_idx]:
                self.nullify_globals(agent_idx)
        return True


    def update_globals(self, observations, int_rewards, ext_rewards, x_pos):
        for agent_idx in range(self.num_agents):
            self.episode_reward["intrinsic"][agent_idx] += int_rewards[agent_idx]
            self.episode_reward["extrinsic"][agent_idx] += ext_rewards[agent_idx]
            self.episode_max_x_pos = np.max((self.episode_max_x_pos, x_pos), axis=0)
            if agent_idx < self.logged_agents:
                self.video_array[agent_idx].append(observations[agent_idx][-1:])


    def log_mean_x_pos_by_episode(self, agent_idx, x_pos):
        self.insert_xpos(agent_idx, x_pos)
        self.send_mean_x_pos_if_ready()


    def insert_xpos(self, agent_idx, x_pos):
        add_new_array = True
        for i in range(len(self.x_positions)):
            if self.x_positions[i][agent_idx] is None:
                add_new_array = False
                self.x_positions[i][agent_idx] = x_pos
                break
        if add_new_array:
            self.x_positions.append([None for i in range(self.num_agents)])
            self.x_positions[-1][agent_idx] = x_pos


    def send_mean_x_pos_if_ready(self):
        all_not_none = True
        for agent_idx in range(self.num_agents):
            if self.x_positions[0][agent_idx] is None:
                all_not_none = False
        if all_not_none:
            mean_x_pos = np.mean(self.x_positions[0])
            del self.x_positions[0]
            wandb.log({"Global/Mean max x pos of agents per episode": mean_x_pos}, 
                      step = self.global_counter.get_count())


    def nullify_globals(self, agent_idx):
        self.episode_reward["intrinsic"][agent_idx] = 0
        self.episode_reward["extrinsic"][agent_idx] = 0
        self.video_array[agent_idx] = []
        self.episode_max_x_pos[agent_idx] = -1