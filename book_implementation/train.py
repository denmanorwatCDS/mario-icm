import torch
from torch.nn import functional as F
from book_implementation.CONFIG import params
from torch import nn

def minibatch_train(replay, ICM, ICM_output, Qmodel, qloss, use_extrinsic=False, use_only_extrinsic=False):
    assert ((not use_only_extrinsic) or use_extrinsic), "If you use only extrinsic, you MUST use use_extrinsic"
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    if ICM_output == "losses":
        forward_pred_err, inverse_pred_err = ICM(state1_batch, action_batch, state2_batch)
        forward_pred_reward = forward_pred_err
    if ICM_output == "predictions":
        forward_pred_err, inverse_pred_err = ICM.get_losses(state1_batch, action_batch, state2_batch, 
                                                            inv_scale=params["inverse_scale"], forward_scale=params["forward_scale"])
        forward_pred_reward = torch.from_numpy(ICM.intrinsic_reward(state1_batch, action_batch, state2_batch).reshape(-1, 1))
        
        #icm_loss = (self.beta*state_prediction_loss + 
        #                (1-self.beta)*action_prediction_loss)
        """
        forward_loss = nn.MSELoss(reduction='none')
        inverse_loss = nn.CrossEntropyLoss(reduction='none')
        action_pred, state2_hat_pred, state2_hat = ICM(state1_batch, action_batch, state2_batch)
        forward_pred_err = params["forward_scale"] * forward_loss(state2_hat_pred, \
            state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        inverse_pred_err = params["inverse_scale"] * inverse_loss(action_pred, \
            action_batch.detach().flatten()).unsqueeze(dim=1)
        """
    i_reward = (1. / params['eta']) * forward_pred_reward
    reward = i_reward.detach()
    if use_extrinsic:
        reward += reward_batch
    if use_only_extrinsic:
        reward = reward_batch
    qvals = Qmodel(state2_batch)
    reward += params['gamma'] * torch.max(qvals)
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack( (torch.arange(action_batch.shape[0]), action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss