import envpool
from envpool.python.protocol import EnvPool
import numpy as np

from packaging import version
from stable_baselines3.common.vec_env import VecEnvWrapper
import gym
from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvObs,
  VecEnvStepReturn,
)

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")

class VecAdapter(VecEnvWrapper):
  """
  Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
  :param venv: The envpool object.
  """

  def __init__(self, venv: EnvPool):
    # Retrieve the number of environments from the config
    venv.num_envs = venv.spec.config.num_envs
    super().__init__(venv=venv)

  def step_async(self, actions: np.ndarray) -> None:
    self.actions = actions

  def reset(self) -> VecEnvObs:
    if is_legacy_gym:
      return self.venv.reset()
    else:
      return self.venv.reset()[0]

  def seed(self, seed = None) -> None:
    # You can only seed EnvPool env by calling envpool.make()
    pass

  def step_wait(self) -> VecEnvStepReturn:
    if is_legacy_gym:
      obs, rewards, dones, info_dict = self.venv.step(self.actions)
    else:
      obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
      dones = terms + truncs
    infos = []
    # Convert dict to list of dict
    # and add terminal observation
    for i in range(self.num_envs):
      infos.append(
        {
          key: info_dict[key][i]
          for key in info_dict.keys()
          if isinstance(info_dict[key], np.ndarray)
        }
      )
      if dones[i]:
        infos[i]["terminal_observation"] = obs[i]
        if is_legacy_gym:
          obs[i] = self.venv.reset(np.array([i]))
        else:
          obs[i] = self.venv.reset(np.array([i]))[0]
    return obs, rewards, dones, infos