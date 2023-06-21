from stable_baselines3.common.logger import Logger
import numpy as np
import wandb

class AgentLogger(Logger):
    def __init__(self, log_frequency, folder, output_formats, global_counter):
        super().__init__(folder, output_formats)
        self.model_logs = {"train/entropy_loss": [], "train/policy_loss": [],
                           "train/policy_gradient_loss": [], "train/clip_fraction": [],
                           "train/clip_range": [],
                           "train/value_loss": [], "train/final/icm_loss": [],
                           "train/final/forward_loss": [], "train/final/inverse_loss": [],
                           "train/raw/forward_loss": [], "train/raw/inverse_loss": [],
                           "train/grads/ICM grad norm (Before clipping)": [], 
                           "train/grads/A2C grad norm (After clipping)": []}
        self.global_counter = global_counter
        self.log_frequency = log_frequency
        self.last_call = 0

    def save_data(self, key, value):
        if key in self.model_logs.keys():
            self.model_logs[key].append(value)
    
    def record(self, key, value, exclude=None):
        if self.global_counter.get_count() - self.last_call >= self.log_frequency or \
                (self.global_counter.get_count() == self.last_call and self.last_call>self.log_frequency):
            if key == "train/n_updates":
                self.model_logs["Performance/Agent_grad_steps"] = value
            if key in self.model_logs.keys():
                wandb_log_info = {}
                loss_name, loss_stats = key, self.model_logs[key]
                wandb_log_info["mean/" + key + " of {} steps".format(self.log_frequency)] = np.mean(loss_stats)
                wandb_log_info["std/" + key + " of {} steps".format(self.log_frequency)] = np.std(loss_stats)
                self.model_logs[loss_name] = []
                steps = self.global_counter.get_count()
                wandb.log(wandb_log_info, step=self.global_counter.get_count())
                self.last_call = self.global_counter.get_count()
        if key in self.model_logs.keys():
            self.save_data(key, value)
        super().record(key, value, exclude)


