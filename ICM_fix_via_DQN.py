import gym
import torch
from torch import nn, optim
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym_super_mario_bros
import numpy as np

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

# gym_super_mario_bros.make()
env = gym.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
replay = ExperienceReplay(N=1000, batch_size=params['batch_size'], override_memory=False)
Qmodel = Qnetwork()
encoder = EncoderModel()
forward_model = ForwardModel()
inverse_model = InverseModel()

forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()
my_ICM = ICM(action_dim=12, temporal_channels=3, eta=1)

all_model_params = list(Qmodel.parameters()) + list(encoder.parameters())
all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
all_model_params += list(my_ICM.forward_net.parameters()) + list(my_ICM.inverse_net.parameters())
all_model_params += list(my_ICM.feature.parameters())
opt = optim.Adam(lr=0.001, params=all_model_params)

ICMInitializer = getICMInitializer(encoder, forward_model, inverse_model)
ICMInitializer(encoder, forward_model, inverse_model)
ICMInitializer(my_ICM.feature, my_ICM.forward_net, my_ICM.inverse_net)

book_ICM = getICM(encoder, forward_model, inverse_model, inverse_loss, forward_loss)
# my_ICM = getICM(my_ICM.feature, my_ICM.forward_net, my_ICM.inverse_net, inverse_loss, forward_loss)

wandb.init()

epochs = int(1e6)
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
    q_val_pred = Qmodel(state1)
    if i > switch_to_eps_greedy:
        action = int(policy(q_val_pred,eps))
    else:
        action = int(policy(q_val_pred))
    for j in range(params['action_repeats']):
        state2, e_reward_, done, info = env.step(action)
        last_x_pos = info['x_pos']
        if done:
            state1 = reset_env(env)
            break
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    state2 = torch.stack(list(state_deque),dim=1)
    current_video.append(state2[0, -1:, :, :])
    replay.add_memory(state1, action, e_reward, state2)
    e_reward = 0
    if episode_length > params['max_episode_len']:
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            done = True
        else:
            last_x_pos = info['x_pos']
    if done:
        ep_lengths.append(info['x_pos'])
        wandb.log({"Maximal x pos": info['x_pos']}, step=i)
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
    forward_pred_err_mine, inverse_pred_err_mine, q_locc = minibatch_train(replay = replay, ICM = my_ICM, ICM_output="predictions",
                                                                 Qmodel = Qmodel, qloss=qloss, use_only_extrinsic=True, 
                                                                 use_extrinsic=True)
    forward_pred_err_book, inverse_pred_err_book, q_loss = minibatch_train(replay = replay, ICM = book_ICM, ICM_output="losses",
                                                                 Qmodel = Qmodel, qloss=qloss, use_only_extrinsic=True, 
                                                                 use_extrinsic=True)
    loss_book = loss_fn(q_loss, forward_pred_err_book, inverse_pred_err_book) 
    loss_mine = loss_fn(0, forward_pred_err_mine, inverse_pred_err_mine)
    # loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
    # inverse_pred_err.flatten().mean())
    # losses.append(loss_list)
    loss_book.backward()
    loss_mine.backward()
    opt.step()
    wandb.log({"DQN loss": q_loss.mean().item(),
               "Forward model loss book": forward_pred_err_book.flatten().mean().item(),
               "Inverse model loss book": inverse_pred_err_book.flatten().mean().item(),
               "Forward model loss mine": forward_pred_err_mine.flatten().mean().item(),
               "Inverse model loss mine": inverse_pred_err_mine.flatten().mean().item()}, step=i)

done = True
state_deque = deque(maxlen=params['frames_per_state'])
log_of_agent = []
for step in range(12000):
    if done:
        env.reset()
        state1 = prepare_initial_state(env.render('rgb_array'))
    q_val_pred = Qmodel(state1)
    action = int(policy(q_val_pred,eps))
    state2, reward, done, info = env.step(action)
    state2 = prepare_multi_state(state1, state2)
    state1=state2
    log_of_agent.append(state1[0, -1:, :, :])

video_array = (np.stack(log_of_agent, axis = 0)*255).astype(np.uint8)

wandb.log({"Agent evaluation": wandb.Video(video_array, fps=30, format="gif")})
