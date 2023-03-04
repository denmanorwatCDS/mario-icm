from stable_baselines3.common.logger import Logger
import numpy as np
import wandb

class A2CLogger(Logger):
    def __init__(self, log_frequency, folder, output_formats, global_counter):
        super().__init__(folder, output_formats)
        self.model_logs = {"train/entropy_loss": [], "train/policy_loss": [],
                           "train/value_loss": [], "train/icm_loss": [],
                           "train/forward_loss": [], "train/inverse_loss": [],
                           "train/ICM grad norm": [], "train/A2C grad norm": []}
        self.global_counter = global_counter
        self.log_frequency = log_frequency
        self.calls_quantity = 0

    def save_data(self, key, value):
        if key in self.model_logs.keys():
            self.model_logs[key].append(value)
    
    def record(self, key, value, exclude = None):
        if key == "train/n_updates":
            self.calls_quantity = value
        if self.calls_quantity > self.log_frequency and self.calls_quantity%self.log_frequency==1 and key in self.model_logs.keys():
            wandb_log_info = {}
            loss_name, loss_stats = key, self.model_logs[key]
            wandb_log_info["mean/" + key + " of {} steps".format(self.log_frequency)] = np.mean(loss_stats)
            wandb_log_info["std/" + key + " of {} steps".format(self.log_frequency)] = np.std(loss_stats)
            self.model_logs[loss_name] = []
            wandb.log(wandb_log_info, step=self.global_counter.get_count())
        if key in self.model_logs.keys():
            self.save_data(key, value)
        super().record(key, value, exclude)


