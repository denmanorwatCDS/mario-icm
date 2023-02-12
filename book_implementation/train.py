import torch
from torch.nn import functional as F
from book_implementation.CONFIG import params
from torch import nn

def minibatch_train(replay, ICM, ICM_type, Qmodel, qloss, use_extrinsic=True):
    assert ICM_type in ["book", "mine"]
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1) #A
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    
    if ICM_type == "book":
        forward_pred_err, inverse_pred_err = ICM(state1_batch, action_batch, state2_batch) #B
        forward_pred_reward = forward_pred_err
    elif ICM_type == "mine":
        forward_pred_err, inverse_pred_err = ICM.get_losses(state1_batch, action_batch, state2_batch, 
                                                            inv_scale=params["inverse_scale"], forward_scale=params["forward_scale"])
        forward_pred_reward = torch.from_numpy(ICM.intrinsic_reward(state1_batch, action_batch, state2_batch).reshape(-1, 1))
    i_reward = (1. / params['eta']) * forward_pred_reward #C
    reward = i_reward.detach() #D
    if use_extrinsic: #E
        reward = reward_batch 
    qvals = Qmodel(state2_batch) #F
    reward += params['gamma'] * torch.max(qvals)
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack( (torch.arange(action_batch.shape[0]), \
    action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss, i_reward