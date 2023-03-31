from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import time
import pickle
from scipy.stats import entropy

from environments.fast_grid import DIRECTIONS_ORDER
from loggers.prepare_maps import FOUR_ROOMS_OBSTACLES


class LoggerCallback(BaseCallback):
    def __init__(self, verbose, wandb_project_name, config, global_counter, quantity_of_agents, grid_size, device,
                 log_frequency=500, video_submission_frequency=10):
        super(LoggerCallback, self).__init__(verbose)
        self.wandb_project_name = wandb_project_name
        self.config = config
        self.global_counter = global_counter
        self.log_frequency = log_frequency
        self.video_submission_frequency = video_submission_frequency
        self.quantity_of_agents = quantity_of_agents
        self.grid_size = grid_size
        self.device = device

        # Theese are calculated on each step
        self.step_characteristics = {"Media/Agent #0 observations": [],
                                     "Media/Environment state of agent #0": [],
                                     "Media/Heatmap": [np.zeros((grid_size, grid_size)) for i in range(quantity_of_agents)],
                                     "mean/train/raw/Estimated probability of ground truth action": [],
                                     "Raw/Intrinsic reward of agent #0": [], "Raw/Extrinsic reward of agent #0": [],
                                     "Mix/Intrinsic reward of agent #0": [], "Mix/Extrinsic reward of agent #0": [],
                                     "Mix/Total reward of agent #0": [],
                                     "Raw/Intrinsic reward": [], "Raw/Extrinsic reward": [],
                                     "Mix/Intrinsic reward": [], "Mix/Extrinsic reward": [],
                                     "Mix/Total reward": []}

        # Simple characteristics - this characteristics can be logged after mean operation
        self.simple_characteristics = ["Raw/Intrinsic reward of agent #0", "Raw/Extrinsic reward of agent #0",
                                       "Mix/Intrinsic reward of agent #0", "Mix/Extrinsic reward of agent #0",
                                       "Mix/Total reward of agent #0", "mean/train/raw/Estimated probability of ground truth action",
                                       "Raw/Intrinsic reward", "Raw/Extrinsic reward",
                                       "Mix/Intrinsic reward", "Mix/Extrinsic reward", "Mix/Total reward"]
        # Videos array
        self.sanity_checked = False
        self.video_names = ["Media/Agent #0 observations", "Media/Environment state of agent #0"]
        self.sanity_check_frames = []
        self.fully_observable_frames = []
        self.episode_of_agent_0_counter = 0

        # Probability of performing ground truth action
        self.mean_probabilites_from_step = []

        # Set of all visited states through train
        self.all_visited_states = set()

        # Previous time
        self.previous_time = None
    

    def _on_training_start(self):
        wandb.init(project=self.wandb_project_name, config=self.config)
        print(self.model.device)

    def log_array_as_heatmap(self, array, name, logs):
        heatmap = sns.heatmap(array)
        heatmap.set_facecolor("green")
        plt.savefig("tmp/"+name+".png")
        plt.clf()
        logs["Media/"+name] = wandb.Image("tmp/"+name+".png")

    def get_triplets_and_coordinates(self):
        with open("tmp/image_array.pkl", "rb") as file:
            unpickled_data = pickle.load(file)
            prev_images, actions, new_images, coordinates, grid_size, obs_matrix =\
                unpickled_data["prev_obs"].to(self.device), unpickled_data["action"].to(self.device), \
                    unpickled_data["next_obs"].to(self.device), unpickled_data["position"], \
                    unpickled_data["grid_size"], unpickled_data["obs_matrix"]
        return prev_images, actions, new_images, coordinates, grid_size, obs_matrix

    def update_heatmap(self, x_positions, y_positions):
        i=0
        for y_pos, x_pos in zip(y_positions, x_positions):
            self.step_characteristics["Media/Heatmap"][i][y_pos, x_pos] += 1
            self.all_visited_states.add((y_pos, x_pos))
            i += 1

    def log_timer(self, log):
        current_time = time.perf_counter()
        if self.previous_time is not None:
            log["Performance/Step time"] = current_time-self.previous_time
        self.previous_time = current_time

    def update_estimated_probability_of_ground_truth(self, old_obs, new_obs, performed_actions):
        motivation_model = self.model.motivation_model
        old_obs, new_obs = (old_obs.to(torch.float32).to(self.model.motivation_device),
                            new_obs.to(torch.float32).to(self.model.motivation_device))
        predicted_probabilities = motivation_model.get_probability_distribution(old_obs, new_obs)
        self.step_characteristics["mean/train/raw/Estimated probability of ground truth action"].append(
            predicted_probabilities[torch.arange(0, performed_actions.shape[-1]), performed_actions].mean().item())

    def append_characteristrics(self, **kwargs):
        for statistic, value in kwargs.items():
            self.step_characteristics[statistic].append(value)

    def process_mean_characteristics(self, log):
        for simple_characteristic in self.simple_characteristics:
            log[simple_characteristic] = np.mean(self.step_characteristics[simple_characteristic])
            self.step_characteristics[simple_characteristic] = []

    def process_heatmap(self, logs):
        visitations = np.array(self.step_characteristics["Media/Heatmap"])
        mean_unique_visited_states = (visitations>0).sum(axis=(1, 2)).mean()
        print(entropy(visitations.reshape((-1, self.quantity_of_agents))).shape)
        mean_entropy_of_all_states = entropy(visitations.reshape((-1, self.quantity_of_agents))).mean()
        average_heatmap = np.mean(self.step_characteristics["Media/Heatmap"], axis=0)
        all_states = average_heatmap.shape[0]*average_heatmap.shape[1]
        mean_steps = average_heatmap.sum()
        logs["Metrics/Mean unique visited states to all states"] = mean_unique_visited_states/all_states
        logs["Metrics/Mean unique visited states to mean episode length"] = mean_unique_visited_states/mean_steps
        logs["Metrics/Mean entropy"] = mean_entropy_of_all_states
        logs["Metrics/All visited by now states (through training)"] = len(self.all_visited_states)
        self.log_array_as_heatmap(average_heatmap, "Visitation heatmap", logs)
        self.step_characteristics["Media/Heatmap"] =\
            [np.zeros((self.grid_size, self.grid_size)) for i in range(self.quantity_of_agents)]

    def log_video_if_ready(self, env_state, done, agent_obs = None):
        logs = {}
        self.step_characteristics["Media/Environment state of agent #0"].append(env_state.cpu()[0][-1:])
        if agent_obs is not None:
            self.step_characteristics["Media/Agent #0 observations"].append(agent_obs.cpu()[0][-1:])
        if done:
            for name in self.video_names:
                if self.episode_of_agent_0_counter % self.video_submission_frequency == 0:
                    if len(self.step_characteristics[name]) > 0:
                        video = np.stack(self.step_characteristics[name])
                        if name == "Media/Environment state of agent #0":
                            logs[name] = wandb.Video(video)
                        else:
                            logs[name] = wandb.Video(video)
                self.step_characteristics[name] = []
            self.episode_of_agent_0_counter += 1
        if len(logs)>0:
            wandb.log(logs, step=self.global_counter.get_count())

    def evaluate_values_and_rewards(self, logs):
        ACTION_TO_NAME = {0: "right", 1: "down", 2: "left", 3: "up"}
        motivation_model, policy = self.model.motivation_model, self.model.policy
        prev_images, actions, new_images, positions, grid_size, obs_matrix = self.get_triplets_and_coordinates()

        reward_matrix_by_action = [np.zeros(grid_size) for i in range(4)]
        agent_occurence_by_action = [np.zeros(grid_size) for i in range(4)]
        with torch.no_grad():
            print("Images shape: {}".format(prev_images.shape))
            print("Actions shape: {}".format(actions.shape))
            rewards = motivation_model.intrinsic_reward(prev_images, actions, new_images)
            print("Rewards shape: {}".format(rewards.shape))
        for i in range(len(positions)):
            position = tuple(positions[i])
            reward_matrix_by_action[actions[i]][position] += rewards[i]
            agent_occurence_by_action[actions[i]][position] += 1
        reward_heatmaps = {}
        all_rewards = np.zeros(grid_size)
        all_occurences = np.zeros(grid_size)
        for action in range(4):
            reward_heatmaps["Reward heatmap after arriving to cell from {}".format(ACTION_TO_NAME[action])] = \
                reward_matrix_by_action[action]/agent_occurence_by_action[action]
            all_rewards += reward_matrix_by_action[action]
            all_occurences += agent_occurence_by_action[action]
        reward_heatmaps["Mean reward heatmap after arriving to cell".format(ACTION_TO_NAME[action])] = \
            all_rewards/all_occurences

        for key, value in reward_heatmaps.items():
            self.log_array_as_heatmap(value, key, logs)

        shape = obs_matrix.shape[:2]
        with torch.no_grad():
            values = policy.predict_values(torch.flatten(obs_matrix, start_dim=0, end_dim=1).to(self.device)).reshape(shape)
            values[torch.from_numpy(FOUR_ROOMS_OBSTACLES==1)] = float("nan")
            self.log_array_as_heatmap(values.cpu().numpy(), "Value of state", logs)



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

        previous_observation, current_observation, current_environment_state =\
            self.locals["obs_tensor"],\
            torch.as_tensor(self.locals["new_obs"]).to(self.device), \
            self.locals["obs_tensor"]
        actions = self.locals["clipped_actions"]
        self.update_estimated_probability_of_ground_truth(previous_observation, current_observation, actions)

        coordinates = [self.locals["infos"][agent_idx]["position"] for agent_idx in range(self.quantity_of_agents)]
        x_positions = [coordinate[1] for coordinate in coordinates]
        y_positions = [coordinate[0] for coordinate in coordinates]
        self.update_heatmap(x_positions, y_positions)

        done = self.locals["dones"][0]
        if done:
            self.sanity_checked = True
        agent_obs = None if self.sanity_checked else previous_observation
        self.log_video_if_ready(current_environment_state, done, agent_obs)

        if self.global_counter.get_count() % self.log_frequency == 0 and self.global_counter.get_count()>0:
            log = {}
            self.process_mean_characteristics(log)
            self.process_heatmap(log)
            self.log_timer(log)
            self.evaluate_values_and_rewards(log)
            wandb.log(log, step=self.global_counter.get_count())
