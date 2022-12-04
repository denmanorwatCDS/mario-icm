from moviepy.editor import ImageSequenceClip
import wandb
import numpy as np

class WandBLogger():
    def __init__(self, HYPERPARAMS, MODEL_NAME, ACTION_SPACE, ACTION_NAMES, period = 10):
        MODEL_CONFIG = dict(HYPERPARAMS)
        MODEL_CONFIG["MODEL_NAME"] = MODEL_NAME
        wandb.init(project = "Mario", config = MODEL_CONFIG)
        wandb.define_metric("Episode")
        wandb.define_metric("Train steps")
        wandb.define_metric("Test steps")
        
        self.ACTION_SPACE = ACTION_SPACE
        self.ACTION_NAMES = ACTION_NAMES
        self.test_episode = 0
        self.total_test_steps = 0
        self.train_episode = 0
        self.total_train_steps = 0
        self.period = period
        self.gradient_steps = 0
        self.accumulated_reward = 0
        self.accumulated_intrinsic_reward = 0

        self.video_sequence_array = []
        self.steps = []

    def clean(self):
        self.video_sequence_array = []
        self.steps = []
            

    def __LogVideo(self, step, state, prefix, done):
        if (step % self.period) == 0:
            self.video_sequence_array.append(state)
        if done:
            current_episode =\
                self.test_episode if prefix == "test" else self.train_episode
            video_sequence = np.concatenate(self.video_sequence_array, axis = 0)
            print("Video shape: {}".format(video_sequence.shape), " from: {}".format(prefix))
            return video_sequence


    def LogLosses(self, model_grad_norm, 
                  policy_loss, value_loss, entropy_loss, total_loss, 
                  icm_loss = None, icm_grad_norm = None):
        wandb.log({"Policy loss": policy_loss,
                   "Episode": self.train_episode,
                   "Train steps": self.total_train_steps})
        wandb.log({"Value loss": value_loss, 
                   "Episode": self.train_episode,
                   "Train steps": self.total_train_steps})
        wandb.log({"Entropy loss": entropy_loss, 
                   "Episode": self.train_episode,
                   "Train steps": self.total_train_steps})
        wandb.log({"Total loss": total_loss, 
                   "Episode": self.train_episode,
                   "Train steps": self.total_train_steps})
        wandb.log({"Actor-critic grad norm": model_grad_norm, 
                   "Episode": self.train_episode,
                   "Train steps": self.total_train_steps})
        if self.total_train_steps%100 == 0:
            print("Iteration is: {}".format(self.total_train_steps))
        
        if icm_loss is not None:
            wandb.log({"Icm loss": icm_loss, 
                       "Episode": self.train_episode,
                       "Train steps": self.total_train_steps})
            wandb.log({"ICM-model grad norm": icm_loss, 
                       "Episode": self.train_episode,
                       "Train steps": self.total_train_steps})

    def LogInfo(self, prefix, reward, x_coordinate, state, done, 
                 intrinsic_reward = None):
        log_dict = {}

        current_episode = self.test_episode if prefix == "test" else self.train_episode
        if prefix == "train":
            self.total_train_steps += 1
            total_steps = self.total_train_steps
            metric_name = "Train steps"
        if prefix == "test":
            self.total_test_steps += 1
            total_steps = self.total_test_steps
            metric_name = "Test steps"

        if done:
            if prefix == "train":
                self.train_episode += 1
            if prefix == "test":
                self.test_episode += 1
            self.accumulated_reward = 0
            self.accumulated_intrinsic_reward = 0
        
        log_dict[prefix + " Reward per step"] = reward
        self.accumulated_reward += reward
        log_dict[prefix + " Accumulated reward"] = self.accumulated_reward
        
        if intrinsic_reward is not None:
            log_dict[prefix + " Intrinsic reward per step"] = intrinsic_reward
            self.accumulated_intrinsic_reward += intrinsic_reward
            log_dict[prefix + " Accumulated intrinsic reward"] = self.accumulated_intrinsic_reward

        log_dict["Episode"] = current_episode
        log_dict[metric_name] = total_steps
        log_dict[prefix + " X coordinate"] = x_coordinate
        
        evaluation_video = self.__LogVideo(total_steps, state, prefix, done)
        if done:
            log_dict[prefix + " Evaluation video"] =\
                wandb.Video(evaluation_video, fps = 10, format = "gif")

        wandb.log(log_dict)
        if done:
            self.clean()

    def LogTest(self, step, action_probabilities, chosen_action, 
                   accumulated_reward, x_coordinate, state, done):
        if done:
            self.test_episode += 1
            print("Test episode is done!")

        self.__LogVideo(step, state, "test", done)