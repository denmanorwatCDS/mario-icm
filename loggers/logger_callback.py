from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class LoggerCallback(BaseCallback):
    def __init__(self, log_freq, verbose, wandb_project_name, config, global_counter, num_agents, all_available_states,
                 grid_size, logged_agents = 3):
        super(LoggerCallback, self).__init__(verbose)
        self.wandb_project_name = wandb_project_name
        self.config = config
        self.logged_agents = logged_agents
        self.global_counter = global_counter
        self.log_freq = log_freq
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.all_available_states = all_available_states


    def _on_training_start(self):
        wandb.init(project = self.wandb_project_name, config=self.config)
        # Global stats
        self.local_video_array = [[] for i in range(self.logged_agents)]
        self.global_video_array = [[] for i in range(self.logged_agents)]
        self.episode_reward = {"extrinsic": [0 for i in range(self.num_agents)], 
                               "intrinsic": [0 for i in range(self.num_agents)]}
        # Local stats
        self.local_mean_reward = {"intrinsic": [], "extrinsic": []}
        self.agent_episode = [0 for i in range(self.num_agents)]

        # Exploration states
        self.unique_states_by_agent = [np.zeros((self.grid_size, self.grid_size)) for i in range(self.num_agents)]
        self.all_states_by_agent = [0 for i in range(self.num_agents)]

        # Mean statistics by state
        self.episode_statistics = {"Unique visits to all visits": [[None for i in range(self.num_agents)]],
                                   "Unique visits to all states": [[None for i in range(self.num_agents)]],
                                   "Extrinsic reward": [[None for i in range(self.num_agents)]]}

        # Local and global obs
        self.log_local_obs = True

        # Quantity of dones from monitored agents
        self.monitored_agents_done_quantity = 0

        # Visited states across learning
        self.visited_states = set()

        # Probabilities of right actions
        self.probabilities_of_right_action = []

    def log_sizes(self):
        print("Local video array: {}, {}".format(len(self.local_video_array), len(self.local_video_array[0])))
        print("Global video array: {}, {}".format(len(self.global_video_array), len(self.global_video_array[0])))
        print("Episode reward: {}, {}".format(len(self.episode_reward["extrinsic"]), len(self.episode_reward["intrinsic"])))
        print("Local mean reward: {}".format(len(self.local_mean_reward["extrinsic"]), len(self.local_mean_reward["intrinsic"])))
        print("Agent episode length: {}".format(len(self.agent_episode)))
        print("Unique states by agent: {}, {}".format(len(self.unique_states_by_agent), len(self.unique_states_by_agent[0])))
        print("All states by agent: {}".format(len(self.all_states_by_agent)))
        print("Episode statistics: {}, {}, {}".format(len(self.episode_statistics), len(self.episode_statistics["Unique visits to all visits"]), len(self.episode_statistics["Unique visits to all visits"][0])))

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
            self.probabilities_of_right_action.append(probabilities[torch.arange(0, target_actions.shape[-1]), target_actions].mean().item())
        if self.global_counter.get_count()%self.log_freq==0 and self.global_counter.get_count()>0:
            wandb.log({"mean/train/raw/mean probability of right action from {} previous steps".format(self.log_freq): 
                       np.mean(self.probabilities_of_right_action)}, step = self.global_counter.get_count())
            self.probabilities_of_right_action = []
            mean_probability_of_right_action = []


    def _on_step(self):
        self.global_counter.count()
        int_rewards = self.locals["int_reward"]
        ext_rewards = self.locals["ext_reward"]
        self._log_local(ext_rewards, int_rewards)
        
        loc_observations = self.locals["obs_tensor"][:self.logged_agents]
        glob_observations = [info["full_img"] for info in self.locals["infos"][:self.logged_agents]]
        train_dones = self.locals["dones"]
        infos = self.locals["infos"]
        self.update_globals(loc_observations, glob_observations, int_rewards, ext_rewards, infos)
                
        for agent_idx in range(self.num_agents):
            if train_dones[agent_idx] and agent_idx < self.logged_agents:
                self.monitored_agents_done_quantity += 1
                logs = {"Global/Rewards/Extrisnic reward per episode of agent #{}".format(agent_idx):
                           self.episode_reward["extrinsic"][agent_idx],
                           "Global/Rewards/Intrinsic reward per episode of agent #{}".format(agent_idx):
                           self.episode_reward["intrinsic"][agent_idx]}
                if agent_idx == 0 and self.global_counter.get_count() % self.log_freq == 0:
                    global_video_of_agent = np.stack(self.global_video_array[agent_idx])
                    logs["Video/Global video of train evaluation of agent #{}".format(agent_idx)] =\
                        wandb.Video(global_video_of_agent, fps=10)
                if self.log_local_obs:
                    local_video_of_agent = torch.stack(self.local_video_array[agent_idx]).detach().cpu().numpy()
                    logs["Video/Local video of train evaluation of agent #{}".format(agent_idx)] =\
                        wandb.Video(local_video_of_agent, fps=30)
                wandb.log(logs, step=self.global_counter.get_count())
            if train_dones[agent_idx]:
                self.nullify_globals(agent_idx)
            if self.global_counter.get_count() % self.log_freq == 0 and self.global_counter.get_count() > 0:
                logs["Global/Exploration/Unique visits to all visits of agent #{}".format(agent_idx)] = \
                    (self.unique_states_by_agent[agent_idx]>0).sum()/self.unique_states_by_agent[agent_idx].sum()
                print("Unique states by agent: {}".format(self.unique_states_by_agent[agent_idx].sum()))
                logs["Global/Exploration/Unique visits to all states of env of agent #{}".format(agent_idx)] = \
                    (self.unique_states_by_agent[agent_idx] > 0).sum() / self.all_available_states
                print("All available states: {}".format(self.all_available_states))
                unique_visits_to_all_visits = (self.unique_states_by_agent[agent_idx]>0).sum() / self.unique_states_by_agent[agent_idx].sum()
                unique_visits_to_all_states = (self.unique_states_by_agent[agent_idx]>0).sum() / self.all_available_states
                extrinsic_reward_per_episode = self.episode_reward["extrinsic"][agent_idx]
                self.save_episode_info("Unique visits to all visits", agent_idx, unique_visits_to_all_visits)
                self.save_episode_info("Unique visits to all states", agent_idx, unique_visits_to_all_states)
                self.save_episode_info("Extrinsic reward", agent_idx, extrinsic_reward_per_episode)
                wandb.log({"Metrics/Visited state": len(self.visited_states)/self.all_available_states}, step =self.global_counter.get_count())
                self.log_episode_info_when_ready()
                self.nullify_metrics(agent_idx)
        return True


    def update_globals(self, local_observations, global_observations, int_rewards, ext_rewards, infos):
        for agent_idx in range(self.num_agents):
            self.episode_reward["intrinsic"][agent_idx] += int_rewards[agent_idx]
            self.episode_reward["extrinsic"][agent_idx] += ext_rewards[agent_idx]
            y_pos, x_pos = tuple(infos[agent_idx]["position"])
            self.unique_states_by_agent[agent_idx][y_pos, x_pos] += 1
            self.visited_states.add(tuple(infos[agent_idx]["position"]))
            if agent_idx < self.logged_agents:
                if self.log_local_obs:
                    self.local_video_array[agent_idx].append(local_observations[agent_idx][-1:])
                if agent_idx == 0:
                    self.global_video_array[agent_idx].append(np.transpose(global_observations[agent_idx], axes = (2, 0, 1)))


    def nullify_globals(self, agent_idx):
        self.episode_reward["intrinsic"][agent_idx] = 0
        self.episode_reward["extrinsic"][agent_idx] = 0
        if agent_idx < self.logged_agents:
            self.local_video_array[agent_idx] = []
            self.log_local_obs = False
            self.global_video_array[agent_idx] = []

    def nullify_metrics(self, agent_idx):
        self.unique_states_by_agent[agent_idx] = np.zeros((self.grid_size, self.grid_size))


    def save_episode_info(self, key, agent_idx, ratio):
        found_empty = False
        for row in self.episode_statistics[key]:
            if row[agent_idx] == None:
                row[agent_idx] = ratio
                found_empty = True
                break
        
        if not found_empty:
            self.episode_statistics[key].append([None for i in range(self.num_agents)])
            self.episode_statistics[key][-1][agent_idx] = ratio


    def log_episode_info_when_ready(self):
        ready_to_log = True
        for agent_idx in range(self.num_agents):
            if self.episode_statistics["Unique visits to all visits"][0][agent_idx] is None:
                ready_to_log = False
                break
        if ready_to_log:
            wandb_log = {}
            for key in self.episode_statistics.keys():
                wandb_log["Metrics/"+key] = float(np.mean(self.episode_statistics[key][0]))
                if len(self.episode_statistics[key])==1:
                    self.episode_statistics[key].append([None for i in range(self.num_agents)])
                del self.episode_statistics[key][0]
                print("Episode statictics length: {}".format(len(self.episode_statistics)))
            print(np.mean(self.unique_states_by_agent, axis=0).shape)
            sns.heatmap(np.mean(self.unique_states_by_agent, axis=0))
            plt.savefig("tmp/Heatmap.png")
            plt.clf()
            wandb_log["Metrics/Heatmap"] = wandb.Image("tmp/Heatmap.png")
            wandb.log(wandb_log, step = self.global_counter.get_count())
