from book_implementation.CONFIG import params
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

def getQInitializer(Qmodel):
    q_param_list = []
    for layer in Qmodel.parameters():
        q_param_list.append(layer.data)
    def QInitializer(Qmodel):
        for i, layer in enumerate(Qmodel.parameters()):
            layer.data = q_param_list[i]
    return QInitializer

def getICMInitializer(encoder, forward_model, inverse_model, seed):
    if not os.path.isfile("/home/dvasilev/mario_icm/ICM_pickled/{}".format(seed)):
        encoder_list = []
        forward_list = []
        inverse_list = []
        seed = seed
        for layer in encoder.parameters():
            encoder_list.append(layer.data)
        for layer in forward_model.parameters():
            forward_list.append(layer.data)
        for layer in inverse_model.parameters():
            inverse_list.append(layer.data)
        params = {"encoder": encoder_list,
                "forward": forward_list,
                "inverse": inverse_list}
        print("Created weights for new seed: {}".format(seed))
        with open("/home/dvasilev/mario_icm/ICM_pickled/{}".format(seed), 'wb') as handle:
            pickle.dump(params, handle)
    def ICMInitializer(encoder, forward_model, inverse_model):
        with open("/home/dvasilev/mario_icm/ICM_pickled/{}".format(seed), 'rb') as handle:
            param_dict = pickle.load(handle)
            encoder_list = param_dict["encoder"]
            forward_list = param_dict["forward"]
            inverse_list = param_dict["inverse"]
        for i, layer in enumerate(encoder.parameters()):
            layer.data = encoder_list[i]
        for i, layer in enumerate(forward_model.parameters()):
            layer.data = forward_list[i]
        for i, layer in enumerate(inverse_model.parameters()):
            layer.data = inverse_list[i]
    return ICMInitializer

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