import numpy as np

import torch
import torchvision.transforms as T

import gym
from gym import Env
from gym.spaces import Box


class SkipFrame(gym.Wrapper) :
    def __init__(self, env: Env, skip: int) -> None:
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)

    def step(self, action) :
        total_reward = 0.0
        done = False
        
        for i in range(self._skip) :
            obs, reward, done, info = self.env.step(action)
            
            if i == self._skip - 2 :
                self._obs_buffer[0] = obs
            if i == self._skip - 1 :
                self._obs_buffer[1] = obs
            
            total_reward += reward
            
            if done :
                break

        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper) :
    transforms = T.Grayscale()

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation) :
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)

        return observation

    def observation(self, observation) :
        observation = self.permute_orientation(observation)
        observation = self.transforms(observation)

        return observation

class ResizeObservation(gym.ObservationWrapper) :
    def __init__(self, env: Env, shape) -> None:
        super().__init__(env)
        if isinstance(shape, int) :
            self.shape = (shape, shape)
        else :
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation) :
        transforms = T.Compose([
            T.Resize(self.shape),
            T.Normalize(0, 255)
        ])

        observation = transforms(observation).squeeze(0)

        return observation

class CustomReward(gym.Wrapper) :
    def __init__(self, env: Env):
        super().__init__(env)
        self._score = 0

    def step(self, action) :
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._score) / 40.0
        self._score = info['score']

        if done :
            if info['flag_get'] :
                reward += 350.0
            else :
                reward -= 50.0

        return state, reward / 10.0, done, info