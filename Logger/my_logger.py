import wandb
from stable_baselines3.common.logger import Logger
import numpy as np

class A2CLogger(Logger):
    def __init__(self, folder, output_formats, num_workers):
        super().__init__(folder, output_formats)
        wandb.define_metric("Number_of_actor-critic_gradient_steps")
        self.current_value_loss = None
        self.current_entropy_loss = None
        self.current_policy_loss = None
        self.logged_for_new_step = False
        self.current_gradient_step = 1
        self.episodes = np.zeros(num_workers)
        self.ext_rewards = np.zeros(num_workers)

    def record(self, key, value, exclude = None):
        super().record(key, value, exclude)

        if key == "train/n_updates":
            wandb.log({"Number_of_actor-critic_gradient_steps": value})
            self.current_gradient_step = value
            self.current_value_loss = None
            self.current_entropy_loss = None
            self.current_policy_loss = None
            self.logged_for_new_step = False

        if key == "train/entropy_loss":
            wandb.log({"Losses/Current entropy loss": value,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})

        if key == "train/policy_loss":
            wandb.log({"Losses/Current policy loss": value,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})

        if key == "train/value_loss":
            wandb.log({"Losses/Current value loss": value,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})

        if key == "train/state_prediction_loss":
            wandb.log({"Losses/Current state prediction loss": value,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})

        if key == "train/action_prediction_loss":
            wandb.log({"Losses/Current action prediction loss": value,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})

        if key == "train/icm_loss":
            wandb.log({"Losses/Current icm loss": value,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})

        if key == "train/episode":
            self.episodes[value] += 1
            wandb.log({"Environment/Extrinsic reward of agent {}".format(value): self.ext_rewards[value],
                       "Episode": self.episodes[value]})
            self.ext_rewards[value] = 0

        if key == "train/x_coordinate":
            x_positions, agent_idx = value
            wandb.log({"Environment/Current x position of agent {}".format(agent_idx): x_positions,
                       "Episode": self.episodes[agent_idx]})

        if key == "train/extrinsic_reward":
            self.ext_rewards += value

        if (not self.logged_for_new_step) and (self.current_entropy_loss is not None) and\
            (self.current_policy_loss is not None) and (self.current_value_loss is not None):
            wandb.log({"Current total actor-critic model loss":
                self.current_entropy_loss + self.current_policy_loss + self.current_value_loss,
                "Number_of_actor-critic_gradient_steps": self.current_gradient_step})
            self.logged_for_new_step = True
