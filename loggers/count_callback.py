from stable_baselines3.common.callbacks import BaseCallback

class CountCallback(BaseCallback):
    def __init__(self, global_counter):
        super().__init__()
        self.global_counter = global_counter

    def _on_step(self):
        super()._on_step()
        self.global_counter.count()
