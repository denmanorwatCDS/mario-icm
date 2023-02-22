from nes_py.wrappers import JoypadSpace #A
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT #B
import torch
import random
import numpy as np
from torch import nn
from torch import optim
import argparse
import pickle
import os
from icm_book.CONFIG import params

SEED = params["seed"]
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT) #C

from icm_book.dqn_buffer import ExperienceReplay
from icm_book.dqn import Qnetwork
from icm_book.icm_blocks import EncoderModel, ForwardModel, InverseModel
from icm_book.CONFIG import params
from icm_book.obs_preprocessing import prepare_initial_state, reset_env, prepare_state, prepare_multi_state
from icm_book.nn_utils import loss_fn, policy, getICM
from icm_book.train import minibatch_train
from collections import deque
import wandb

from icm_mine.icm import ICM

parser = argparse.ArgumentParser()
parser.add_argument("fixate_buffer", help="Specify, do we need to fixate buffer. Valid options: {yes, no}")
parser.add_argument("type_of_ICM", help="Specify, which implementation of ICM to use. Valid options: {mine, book}")
parser.add_argument("use_softmax", help="Specify, do we need to use softmax in ICM. Valid options: {yes, no}")
args = parser.parse_args()
assert args.fixate_buffer in ["yes", "no"], "Valid params for fixate buffer are yes or no"
assert args.type_of_ICM in ["mine", "book"], "Valid params for type of ICN are mine or book"
assert args.use_softmax in ["yes", "no"], "Valid params for use softmax are yes or no"

wandb.init(config=params)

replay = ExperienceReplay(N=params["buffer_size"], batch_size=params['batch_size'], seed=SEED)
if args.fixate_buffer == "yes":
    fixate_buffer = True
    if os.path.exists("/home/dvasilev/mario_icm/DQN_buffer/{}".format(SEED)):
        with open('/home/dvasilev/mario_icm/DQN_buffer/{}'.format(SEED), 'rb') as handle:
            replay.memory = pickle.load(handle)
            print("Buffer copied from file")
    else:
        print("Buffer created")

elif args.fixate_buffer == "no":
    fixate_buffer = False

use_softmax = None
if args.use_softmax == "yes":
    use_softmax = True
else:
    use_softmax = False
type_of_ICM = args.type_of_ICM

def createICMByType(type_of_ICM):
    ICM_model, opt = None, None
    if type_of_ICM == "mine":
        ICM_model = ICM(action_dim=12, temporal_channels=3, eta=1, inv_scale=params["inverse_scale"], 
                        forward_scale=params["forward_scale"], use_softmax=use_softmax)
        all_model_params = list(Qmodel.parameters()) + list(ICM_model.feature.parameters()) #A
        all_model_params += list(ICM_model.forward_net.parameters()) + list(ICM_model.inverse_net.parameters())
        opt = optim.Adam(lr=0.001, params=all_model_params)

    if type_of_ICM == "book":
        encoder = EncoderModel()
        forward_model = ForwardModel()
        inverse_model = InverseModel()
        ICM_model = getICM(encoder, forward_model, inverse_model, inverse_loss, forward_loss)
        all_model_params = list(encoder.parameters()) + list(forward_model.parameters()) 
        all_model_params += list(inverse_model.parameters()) + list(Qmodel.parameters())
        opt = optim.Adam(lr=0.001, params=all_model_params)
    return ICM_model, opt

Qmodel = Qnetwork()

forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()

ICM_model, opt = createICMByType(type_of_ICM)

epochs = 25_000
if fixate_buffer:
    epochs = replay.N - len(replay.memory)


env.reset()
state1 = prepare_initial_state(env.render('rgb_array'))
eps=0.15
losses = []
episode_length = 0
switch_to_eps_greedy = 1000
state_deque = deque(maxlen=params['frames_per_state'])
e_reward = 0.
last_x_pos = env.env.env._x_position #A
ep_lengths = []
use_extrinsic = False
current_video = []
for i in range(epochs):
    if i % 25 == 0:
        print("{}/{}".format(i, epochs))
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
    if not fixate_buffer:
        current_video.append(state2[0, -1:, :, :])
    replay.add_memory(state1, action, e_reward, state2) #F
    e_reward = 0
    if episode_length > params['max_episode_len']: #G
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            done = True
        else:
            last_x_pos = info['x_pos']
    if done:
        ep_lengths.append(info['x_pos'])
        state1 = reset_env(env)
        #last_x_pos = env.env.env._x_position
        episode_length = 0
        if not fixate_buffer:
            current_video = (np.stack(current_video, axis = 0)*255).astype(np.uint8)
            wandb.log({"Maximal x pos": info['x_pos']}, step=i)
            wandb.log({"Agent train": wandb.Video(current_video, fps=30, format="gif")}, step=i)
            current_video = []
    else:
        state1 = state2
    if len(replay.memory) < params['batch_size']:
        continue
    forward_pred_err, inverse_pred_err, q_loss = minibatch_train(replay, ICM_model, Qmodel, qloss, use_extrinsic=use_extrinsic) #H
    loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err) #I
    loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
    inverse_pred_err.flatten().mean())
    losses.append(loss_list)
    loss.backward()
    opt.step()
    if not fixate_buffer:
        wandb.log({"DQN loss": q_loss.mean().item(),
                   "Forward model loss": forward_pred_err.flatten().mean().item(),
                   "Inverse model loss": inverse_pred_err.flatten().mean().item()}, step=i)

if fixate_buffer:
    print("Started fixated buffer experiment!")
    if not os.path.exists("/home/dvasilev/mario_icm/DQN_buffer/{}".format(SEED)):
        with open("/home/dvasilev/mario_icm/DQN_buffer/{}".format(SEED), 'wb') as handle:
            pickle.dump(replay.memory, handle)
            print("Replay buffer dumped!")


    ICM_model, opt = createICMByType(type_of_ICM)
    inverse_iterations = params["inverse_iterations"]
    forward_iterations = params["forward_iterations"]

    for i in range(inverse_iterations):
        state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
        action_batch = action_batch.view(action_batch.shape[0], 1) #A

        if isinstance(ICM_model, ICM):
            forward_pred_err, inverse_pred_err = ICM_model.get_losses(state1_batch, action_batch, state2_batch)
            probabilities = ICM_model.get_probability_distribution(state1_batch, state2_batch)
            action_array = action_batch.flatten()
            mean_probability_of_right_action = probabilities[torch.arange(0, params["batch_size"]), action_array].mean().item()
            accuracy = (action_array==probabilities.argmax(dim=1)).sum()/action_array.shape[0]
            wandb.log({"Mean probability of right action": mean_probability_of_right_action,
                       "Accuracy": accuracy}, step = i)
        else:
            forward_pred_err, inverse_pred_err = ICM_model(state1_batch, action_batch, state2_batch)

        forward_pred_reward = forward_pred_err
        wandb.log({"Forward model loss": forward_pred_err.flatten().mean().item(),
                   "Inverse model loss": inverse_pred_err.flatten().mean().item(),
                   "Mean intrinsic reward": forward_pred_reward.flatten().mean().item()}, step=i)
        inverse_pred_err = loss_fn(0, inverse_pred_err, 0)/params["beta"]
        opt.zero_grad()
        inverse_pred_err.backward()
        opt.step()
    
    for i in range(forward_iterations):
        state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch()
        action_batch = action_batch.view(action_batch.shape[0], 1)
        if isinstance(ICM_model, ICM):
            forward_pred_err, inverse_pred_err = ICM_model.get_losses(state1_batch, action_batch, state2_batch)
        else:
            forward_pred_err, inverse_pred_err = ICM_model(state1_batch, action_batch, state2_batch)
            forward_pred_err = loss_fn(0, 0, forward_pred_err)/params["beta"]
        wandb.log({"Forward model loss": forward_pred_err.flatten().mean().item(),
                   "Inverse model loss": inverse_pred_err.flatten().mean().item()})
        opt.zero_grad()
        forward_pred_err.backward()
        opt.step()


if not fixate_buffer:
    done = True
    current_video = []
    state_deque = deque(maxlen=params['frames_per_state'])
    for step in range(5000):
        if done:
            env.reset()
            state1 = prepare_initial_state(env.render('rgb_array'))
            current_video = (np.stack(current_video, axis = 0)*255).astype(np.uint8)
            wandb.log({"Agent test": wandb.Video(current_video, fps=30, format="gif")}, step=i)
            current_video = []
        q_val_pred = Qmodel(state1)
        action = int(policy(q_val_pred,eps))
        state2, reward, done, info = env.step(action)
        state2 = prepare_multi_state(state1,state2)
        state1=state2
        current_video.append(state2[0, -1:, :, :])
