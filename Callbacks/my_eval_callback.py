from stable_baselines3.common.callbacks import EvalCallback
import wandb
import numpy as np

class CustomEvalCallback(EvalCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, eval_env, eval_freq,  
                 action_space_size, parallel_envs, verbose=0, detailed = True, 
                 deterministic=True, render=False, 
                 best_model_save_path="./logs/", log_path="./logs/"):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, verbose=verbose, eval_freq=eval_freq, 
                                             deterministic=deterministic, render=render, 
                                             best_model_save_path=best_model_save_path,
                                             log_path=log_path, n_eval_episodes=5)
        run = wandb.init(project = "Cartpole")
        wandb.define_metric("Agent steps")
        self.episodic_rewards = [0]*parallel_envs
        self.agent_steps = 0
        self.action_space_size = action_space_size
        self.parallel_envs = parallel_envs
        self.detailed = detailed
        self.eval_steps = 0
        self.eval_env = eval_env
        wandb.define_metric("Agent iterations")
        print("Eval envs: {}".format(self.parallel_envs))
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # typnpe: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print("I've been called")
            #print("Local keys: {}".format(self.locals.keys()))
            #print("Global keys: {}".format(self.globals.keys()))
            #print("Rewards: {}".format(self.locals["rewards"]))
            #print(self.locals["callback"])
            mean_reward, _, sampled_observations = evaluate_policy(self.model, self.eval_env, 
                                                                   n_eval_episodes=5, deterministic=True)
            video_array = np.concatenate(sampled_observations, axis = 0)
            print(video_array.shape)
            wandb.log({"Evaluation mean reward": float(mean_reward),
                       "Agent steps": self.n_calls})
            wandb.log({"Agent evaluation": wandb.Video(video_array, fps=30, format="gif"),
                        "Agent steps": self.n_calls})
        #print(self.locals["eval_env"].observation_buffer[-1])
        return True

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
            if i % n_eval_episodes == 0:
                observation_frame = np.transpose(observation, (0, 3, 1, 2))[:, -1:, :, :]
                sampled_observations.append(observation_frame)
        if done[0]:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
    
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, sampled_observations
