from threading import Lock
import torch
import numpy as np
from Config.ENV_CFG import DEVICE

class TensorConvertionManager:
    torch_lock = Lock()
    numpy_lock = Lock()


    def untorchify_observation(observation):
        with TensorConvertionManager.numpy_lock:
            observation = torch.squeeze(observation, dim=0)
            observation = observation.cpu().numpy()
            observation = (observation.transpose(1, 2, 0)*255).astype(int)
            return observation
