from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb

class LoggerCallback(BaseCallback):
    def __init__(self, verbose, wandb_project_name, config, global_counter, logged_agents = 3):
        super(LoggerCallback, self).__init__(verbose)
        self.wandb_project_name = wandb_project_name
        self.config = config
        self.logged_agents = logged_agents
        self.global_counter = global_counter

    def _on_training_start(self):
        wandb.init(project = self.wandb_project_name, config=self.config)
        wandb.define_metric("Aggregated env steps")
        self.environment_steps = 0
        self.optimizer_steps = 0
        self.aggregated_env_steps = 0
        self.video_array = [[] for i in range(self.logged_agents)]
        self.episode_reward = [0 for i in range(self.logged_agents)]

    def _on_step(self):
        self.global_counter.count()
        train_rewards = self.locals["rewards"][:self.logged_agents]
        train_observations = self.locals["obs_tensor"][:self.logged_agents]
        train_x_pos = [info["x_pos"] for info in self.locals["infos"][:self.logged_agents]]
        train_dones = self.locals["dones"][:self.logged_agents]
        for agent_idx in range(self.logged_agents):
            self.episode_reward[agent_idx] += train_rewards[agent_idx]
            self.video_array[agent_idx].append(train_observations[agent_idx][-1:])
            wandb.log({"Rewards/Reward per step of agent #{}".format(agent_idx): train_rewards[agent_idx],
                       "Rewards/Accumulated reward of agent #{}".format(agent_idx): self.episode_reward[agent_idx], 
                       "Position/Current x position of agent #{}".format(agent_idx): train_x_pos[agent_idx]}, 
                       step = self.global_counter.get_count())
            if train_dones[agent_idx]:
                video_of_agent = torch.stack(self.video_array[agent_idx]).detach().cpu().numpy()
                wandb.log({"Rewards/Reward per episode of agent #{}".format(agent_idx): self.episode_reward[agent_idx],
                           "Video/Video of train evaluation of agent #{}".format(agent_idx): wandb.Video(video_of_agent, fps=30)},
                           step = self.global_counter.get_count())
                self.video_array[agent_idx] = []
                self.episode_reward[agent_idx] = 0
        return True
