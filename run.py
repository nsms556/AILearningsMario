import datetime
import os
import argparse

import torch

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Mario
from wrapper import SkipFrame, GrayScaleObservation, ResizeObservation, CustomReward
from actions import MOVEMENT
from logger import TensorboardLogger


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
env = CustomReward(env)

env.reset()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', required=False)
parser.add_argument('--render', '-r', type=bool, default=True)
parser.add_argument('-n', type=int, default=-1)
parser.add_argument('--tensorboard', '-t', type=bool, default=True)
args = parser.parse_args()

save_dir = './checkpoints/{}'.format(datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S'))
if not os.path.exists(save_dir) :
    os.mkdir(save_dir)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
if args.checkpoint :
    mario.load(args.checkpoint)

logger = TensorboardLogger(save_dir)

print('Tensor Deivce : {}\n'.format(mario.device))

episodes = args.n
e = 0
try :
    while True :
        e += 1
        state = env.reset()
        logger.init_episode()

        while True :
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)

            q, loss = mario.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done or info['flag_get'] :
                break
            
            if args.render :
                env.render()

        logger.log_episode()

        if e % 5 == 0 :
            logger.record(e, mario.exploration_rate, mario.curr_step)

        if episodes > 0 and e > episodes :
           break
        
finally :
    if input('Save Current Weights? (Y/N)').lower() == 'y' :
        mario.save()

    logger.close()
    env.close()