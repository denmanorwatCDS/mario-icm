from stable_baselines3.common.callbacks import BaseCallback
import wandb
import torch
import numpy as np
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, action_space_size, parallel_envs, HYPERPARAMS, project_name, quantity_of_logged_agents=3,
                 verbose=0, detailed = True, framedrop=6):
        super(CustomCallback, self).__init__(verbose)
        run = wandb.init(project = project_name, config=HYPERPARAMS)
        wandb.define_metric("Agent steps")
        self.episodic_rewards = [0]*min(parallel_envs, quantity_of_logged_agents)
        self.agent_steps = 0
        self.action_space_size = action_space_size
        self.parallel_envs = parallel_envs
        self.quantity_of_logged_agents = quantity_of_logged_agents
        self.detailed = detailed
        self.videos_buffer = [[] for i in range(min(parallel_envs, quantity_of_logged_agents))]
        self.framedrop=framedrop
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

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        #print("Local keys: {}".format(self.locals.keys()))
        #print("Global keys: {}".format(self.globals.keys()))
        #print(self.locals["obs_tensor"].shape)
        #print("Rewards: {}".format(self.locals["rewards"]))

        agents_rewards = self.locals["rewards"][:self.quantity_of_logged_agents]
        agents_dones = self.locals["dones"][:self.quantity_of_logged_agents]
        agents_obs = self.locals["obs_tensor"][:self.quantity_of_logged_agents]
        self.log_episodic_reward(agents_dones)
        self.save_rewards(agents_rewards, agents_dones)
        self.log_current_observation(agents_obs, agents_dones)
        if self.detailed:
            agents_probs = self.__get_model_probs(self.locals["obs_tensor"])[:self.quantity_of_logged_agents]
            self.log_actions_and_rewards(agents_probs)
        self.agent_steps += 1
        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


    def __get_model_probs(self, obs):
        action_distribution = self.model.policy.get_distribution(obs)
        resulting_probabilities = []
        for i in range(self.action_space_size):
            actions = torch.zeros(self.parallel_envs)+i
            actions = actions.cuda()
            probs = torch.exp(action_distribution.log_prob(actions))
            resulting_probabilities.append(probs)
        return torch.stack(resulting_probabilities).transpose(0, 1)
    

    def log_actions_and_rewards(self, probs_by_agent):
        action_reward_dict = dict()
        for i, actions in enumerate(probs_by_agent):
            action_reward_dict["Rewards/Agent #{} reward".format(i)] = self.episodic_rewards[i]
            for j, action in enumerate(actions):
                action_reward_dict["Agent #{} action #{} probability".format(i, j)] = action
        action_reward_dict["Agent steps"] = self.n_calls
        wandb.log(action_reward_dict)
    

    def save_rewards(self, reward_by_agent, is_agent_done):
        for i, reward_done in enumerate(zip(reward_by_agent, is_agent_done)):
            reward, done = reward_done
            self.episodic_rewards[i] += reward
            if done:
                self.episodic_rewards[i] = 0
    

    def log_episodic_reward(self, dones):
        for i, episodic_reward in enumerate(self.episodic_rewards):
            if dones[i]:
                wandb.log({"Rewards/Episodic reward of agent #{}".format(i): episodic_reward,
                           "Agent steps": self.n_calls})


    def log_current_observation(self, obs, dones):
        observations = obs
        for i in range(obs.shape[0]):
            observation, done = observations[i], dones[i]
            self.videos_buffer[i].append(observation[-1:])
            if done:
                video_array = torch.stack(self.videos_buffer[i][::self.framedrop]).cpu()
                print(video_array.shape)
                video_array = np.array(video_array)
                self.videos_buffer[i] = []
                wandb.log({"Agent №{}".format(i): wandb.Video(video_array, fps=30/self.framedrop, format="gif"),
                           "Agent steps": self.n_calls})
