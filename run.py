import datetime
import os
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

from agent import Mario
from wrapper import SkipFrame, GrayScaleObservation, ResizeObservation
from actions import MOVEMENT
from logger import MetricLogger

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

env.reset()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', required=False)
parser.add_argument('--render', '-r', type=bool, default=False)
parser.add_argument('-n', type=int, default=50)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print('Use CUDA : {}'.format(use_cuda))
print()

save_dir = './checkpoints/{}'.format(datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S'))
if not os.path.exists(save_dir) :
    os.mkdir(save_dir)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
if args.checkpoint :
    mario.load(args.checkpoint)

logger = MetricLogger()
tensorboard_writer = SummaryWriter(save_dir + '/')

episodes = args.n
try :
    for e in range(episodes) :
        state = env.reset()

        while True :
            if mario.curr_step == mario.burnin :
                print('Logging Start')
                
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)

            q, loss = mario.learn()
            if q != None and loss != None :
                logger.log_step(reward, loss, q)
                tensorboard_writer.add_scalar('Training Loss', loss, (mario.curr_step-mario.burnin)/mario.learn_rate)
                tensorboard_writer.add_scalar('Q Value', q, (mario.curr_step-mario.burnin)/mario.learn_rate)

            state = next_state

            if done or info['flag_get'] :
                break
            
            if args.render :
                env.render()

        logger.log_episode()
        if e % 5 == 0 :
            logger.record(e, mario.exploration_rate, mario.curr_step)
        
finally :
    if input('Save Current Weights? (Y/N)').lower() == 'y' :
        mario.save()

    tensorboard_writer.close()