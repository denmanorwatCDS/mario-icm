import gym
import torch
from torch import nn, optim
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym_super_mario_bros
import numpy as np
import argparse

from book_implementation.DQN_buffer import ExperienceReplay
from book_implementation.DQN import Qnetwork
from book_implementation.ICM_blocks import EncoderModel, ForwardModel, InverseModel
from book_implementation.CONFIG import params
from book_implementation.obs_preprocessing import prepare_initial_state, reset_env, prepare_state, prepare_multi_state
from book_implementation.nn_utils import loss_fn, policy, getICM, getICMInitializer
from book_implementation.train import minibatch_train
from collections import deque
import wandb

from ICM.ICM import ICM
import random

SEED = 322
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

mode = "Extrinsic"
override_memory = None
use_extrinsic = None
if mode == "Extrinsic":
    override_memory = True
    use_extrinsic = True
elif mode == "Intrinsic":
    override_memory = False
    use_extrinsic = False

parser = argparse.ArgumentParser()
parser.add_argument("ICM_type", help="Specify which version of ICM do you want to use (Or both). Valid options: {book, mine, both}")
args = parser.parse_args()
ICM_type = args.ICM_type

# gym_super_mario_bros.make()
env = gym.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
replay = ExperienceReplay(N=1000, batch_size=params['batch_size'], override_memory=override_memory)
Qmodel = Qnetwork()


forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()


ICMs = {}
all_model_params = list(Qmodel.parameters())
if ICM_type == "mine" or ICM_type == "both":
    ICMs["mine"] = ICM(action_dim=12, temporal_channels=3, eta=1)
    ICMInitializer = getICMInitializer(ICMs["mine"].feature, ICMs["mine"].forward_net, ICMs["mine"].inverse_net, SEED)
    all_model_params += list(ICMs["mine"].forward_net.parameters()) + list(ICMs["mine"].inverse_net.parameters())
    all_model_params += list(ICMs["mine"].feature.parameters())
    ICMInitializer(ICMs["mine"].feature, ICMs["mine"].forward_net, ICMs["mine"].inverse_net)


if ICM_type == "book" or ICM_type == "both":
    encoder = EncoderModel()
    forward_model = ForwardModel()
    inverse_model = InverseModel()
    ICMInitializer = getICMInitializer(encoder, forward_model, inverse_model, SEED)
    ICMs["book"] = getICM(encoder, forward_model, inverse_model, inverse_loss, forward_loss)
    all_model_params += list(encoder.parameters()) + list(forward_model.parameters()) 
    all_model_params += list(inverse_model.parameters())
    ICMInitializer(encoder, forward_model, inverse_model)

opt = optim.Adam(lr=0.001, params=all_model_params)

wandb.init()

epochs = 10_000
env.reset()
state1 = prepare_initial_state(env.render('rgb_array'))
eps=0.15
losses = []
episode_length = 0
switch_to_eps_greedy = 1000
state_deque = deque(maxlen=params['frames_per_state'])
e_reward = 0.
last_x_pos = env.env.env._x_position
ep_lengths = []
use_explicit = False
current_video = []
for i in range(epochs):
    opt.zero_grad()
    episode_length += 1
    q_val_pred = Qmodel(state1) #B
    if i > switch_to_eps_greedy: #C
        action = int(policy(q_val_pred,eps))
    else:
        action = int(policy(q_val_pred))
    for j in range(params['action_repeats']): #D
        state2, e_reward_, done, info = env.step(action)
        last_x_pos = info['x_pos']
        if done:
            state1 = reset_env(env)
            break
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    state2 = torch.stack(list(state_deque),dim=1) #E
    current_video.append(state2[0, -1:, :, :])
    replay.add_memory(state1, action, e_reward, state2) #F
    e_reward = 0
    if episode_length > params['max_episode_len']: #G
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            done = True
        else:
            last_x_pos = info['x_pos']
    if done:
        wandb.log({"Maximal x pos": info['x_pos']}, step=i)
        ep_lengths.append(info['x_pos'])
        state1 = reset_env(env)
        last_x_pos = env.env.env._x_position
        episode_length = 0
        current_video = (np.stack(current_video, axis = 0)*255).astype(np.uint8)
        wandb.log({"Agent train": wandb.Video(current_video, fps=30, format="gif")}, step=i)
        current_video = []
    else:
        state1 = state2
    if len(replay.memory) < params['batch_size']:
        continue
    replay.set_random_ind()
    forward_pred_errors = {}
    inverse_pred_errors = {}
    q_losses = {}
    losses = {}
    intrinsic_rewards = {}
    for type_of_icm, icm in ICMs.items():
        forward_pred_err, inverse_pred_err, q_loss, i_reward = minibatch_train(replay, icm, type_of_icm, Qmodel, qloss, 
                                                                use_extrinsic=use_extrinsic) #H
        forward_pred_errors[type_of_icm] = forward_pred_err
        inverse_pred_errors[type_of_icm] = inverse_pred_err
        q_losses[type_of_icm] = q_loss
        intrinsic_rewards[type_of_icm] = i_reward
        if len(ICMs)>1:
            if type_of_icm=="mine":
                q_loss = 0
        losses[type_of_icm] = loss_fn(q_loss, forward_pred_err, inverse_pred_err) #I
    for type_of_loss, loss in losses.items():
        loss.backward()
        wandb.log({"DQN loss {}".format(type_of_loss): q_losses[type_of_loss].mean().item(),
                   "Forward model loss {}".format(type_of_loss): forward_pred_errors[type_of_loss].flatten().mean().item(),
                   "Inverse model loss {}".format(type_of_loss): inverse_pred_errors[type_of_loss].flatten().mean().item(),
                   "Mean intrinsic reward of {}".format(type_of_loss): intrinsic_rewards[type_of_icm].flatten().mean().item()}, step=i)
    opt.step()
