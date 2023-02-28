from stable_baselines3.common.logger import Logger
import wandb

class A2CLogger(Logger):
    def __init__(self, folder, output_formats, global_counter):
        super().__init__(folder, output_formats)
        self.model_logs = {"train/n_updates": None, "train/entropy_loss": None, 
                           "train/policy_loss": None, "train/value_loss": None}
        self.current_gradient_step = 1
        self.global_counter = global_counter
    
    def record(self, key, value, exclude = None):
        super().record(key, value, exclude)
        self.current_gradient_step += 1
        if key in self.model_logs.keys():
            if self.model_logs[key] is None:
                self.model_logs[key] = value
            else:
                info_for_wandb = {}
                for key, value in self.model_logs.items():
                    if value is not None:
                        info_for_wandb[key] = value
                        self.model_logs[key] = None
                wandb.log(info_for_wandb, step = self.global_counter.get_count())

