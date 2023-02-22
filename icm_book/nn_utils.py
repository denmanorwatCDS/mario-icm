from icm_book.CONFIG import params
import torch
from torch.nn import functional as F
from torch import nn
import math
import os
import pickle

def loss_fn(q_loss, inverse_loss, forward_loss):
    loss_ = (1 - params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss


def getICM(encoder, forward_model, inverse_model, inverse_loss, forward_loss):
    
    def ICM(state1, action, state2, forward_scale=params["forward_scale"], inverse_scale=params["inverse_scale"]):
        state1_hat = encoder(state1) #A
        state2_hat = encoder(state2)
        state2_hat_pred = forward_model(state1_hat.detach(), action.detach()) #B
        forward_pred_err = forward_scale * forward_loss(state2_hat_pred, \
                        state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        pred_action = inverse_model(state1_hat, state2_hat) #C
        inverse_pred_err = inverse_scale * inverse_loss(pred_action, \
                                        action.detach().flatten()).unsqueeze(dim=1)
        return forward_pred_err, inverse_pred_err
    return ICM

def policy(qvalues, eps=None): #A
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0,high=7,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues)), num_samples=1) #B