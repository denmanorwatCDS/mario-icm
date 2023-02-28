import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class LoggerEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, global_counter, n_eval_episodes=1):
        super(LoggerEvalCallback, self).__init__()
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.global_counter = global_counter
    
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, _, sampled_observations = evaluate_policy(self.model, self.eval_env, 
                                                                   n_eval_episodes=self.n_eval_episodes, deterministic=True)
            wandb.log({"Evaluation/Mean reward": mean_reward,
                       "Evaluation/Video": wandb.Video(sampled_observations, fps=30)},
                       step = self.global_counter.get_count())
    
def evaluate_policy(
    model, env,
    n_eval_episodes: int = 10,
    deterministic = True,
):
    episode_rewards = []
    episode_lengths = []
    sampled_observations = []
    for i in range(n_eval_episodes):
        done = np.array([False])
        observation = env.reset()
        current_reward = 0
        current_length = 0
        while not done[0]:
            action, states = model.predict(observation, deterministic=deterministic)
            observation, reward, done, info = env.step(action)
            current_reward += reward
            current_length += 1
            if (i+1) % n_eval_episodes == 0:
                #print(np.transpose(observation, axes=(0, 3, 1, 2)).squeeze()[-1:].shape)
                sampled_observations.append(np.transpose(observation, axes=(0, 3, 1, 2)).squeeze()[-1:])
        if done[0]:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
    
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    sampled_observations = np.stack(sampled_observations, axis = 0)
    print(sampled_observations.shape)
    return mean_reward, std_reward, sampled_observations