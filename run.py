import datetime
import os
from pathlib import Path

import torch

from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

from agent import Mario
from wrapper import SkipFrame, GrayScaleObservation, ResizeObservation
from logger import MetricLogger
from actions import MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

env.reset()

use_cuda = torch.cuda.is_available()
print('Use CUDA : {}'.format(use_cuda))
print()

save_dir = Path('./checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
if not os.path.exists('./checkpoints/{}'.format(datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S'))) :
    save_dir.mkdir(parents=True)
#save_dir = './checkpoints/{}'.format(datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S'))
#os.mkdir(save_dir)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes) :
    state = env.reset()

    while True :
        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        q, loss = mario.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or info['flag_get'] :
            break

        env.render()

    logger.log_episode()

    if e % 5 == 0 :
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
