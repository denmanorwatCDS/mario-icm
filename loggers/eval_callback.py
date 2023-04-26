import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from torchvision.transforms import Grayscale

class LoggerEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, global_counter, n_eval_episodes=1):
        super(LoggerEvalCallback, self).__init__()
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.global_counter = global_counter
    
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_found_vest, sampled_observations = evaluate_policy(self.model, self.eval_env, 
                                                                   n_eval_episodes=self.n_eval_episodes, deterministic=True)
            wandb.log({"Evaluation/Found vest": mean_found_vest,
                       "Evaluation/Video": wandb.Video(sampled_observations, fps=30)},
                       step = self.global_counter.get_count())
    
def evaluate_policy(
    model, env,
    n_eval_episodes: int = 10,
    deterministic = True,
):
    found_vest_arr = []
    sampled_observations = []
    for i in range(n_eval_episodes):
        done = np.array([False])
        observation = env.reset()
        current_reward = 0
        current_length = 0
        while not done[0]:
            action, states = model.predict(observation, deterministic=deterministic)
            observation, reward, done, info = env.step(action)
            gray_observation = observation[:, 9:10]*0.2989 + observation[:, 10:11]*0.5870 + observation[:, 11:]*0.1140
            current_reward += reward
            current_length += 1
            if (i+1) % n_eval_episodes == 0:
                sampled_observations.append(gray_observation)
        if done[0]:
            found_vest = 1 if info[0]["elapsed_step"] != 525 else 0 
            found_vest_arr.append(found_vest)
    
    found_vest_arr = np.array(found_vest)
    
    mean_found_vest = np.mean(found_vest_arr)

    sampled_observations = np.concatenate(sampled_observations, axis = 0)
    return mean_found_vest, sampled_observations
