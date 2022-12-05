import random

from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models import MarioNet


class Mario :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, state_dim, action_dim, save_dir) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_rate = 5e5

        self.memory = deque(maxlen=20000)
        self.batch_size = 32

        self.gamma = 0.9
        self.learning_rate = 0.0001

        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.burnin = 5000
        self.learn_rate = 4
        self.sync_rate = 1e4

    def act(self, state) :
        if np.random.rand() < self.exploration_rate :
            action_idx = np.random.randint(self.action_dim)
        
        else :
            state = state.__array__()
            state = torch.tensor(state).to(self.device)
            state = state.unsqueeze(0)

            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        
        return action_idx

    def cache(self, state, next_state, action, reward, done) :
        state = state.__array__()
        next_state = next_state.__array__()

        state = torch.tensor(state).to(self.device)
        next_state = torch.tensor(next_state).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        self.memory.append((state, next_state, action, reward, done, ))

    def recall(self) :
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self) :
        if self.curr_step % self.sync_rate == 0 :
            self.sync_Q_target()

        if self.curr_step % self.save_rate == 0 :
            self.save()

        if self.curr_step < self.burnin :
            return None, None

        if self.curr_step % self.learn_rate != 0 :
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_e = self.td_estimate(state, action)
        td_t = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_e, td_t)

        return (td_e.mean().item(), loss)

    def td_estimate(self, state, action) :
        current_Q = self.net(state, 'online')[np.arange(0, self.batch_size), action]

        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done) :
        next_state_Q = self.net(next_state, 'online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, 'target')[np.arange(0, self.batch_size), best_action]

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_Q_target(self) :
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self) :
        save_path = self.save_dir + '/Mario_Net_{}.pt'.format(int(self.curr_step // self.save_rate))

        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)

        print('MarioNet saved to {} at step {}'.format(save_path, self.curr_step))

    def load(self, model_path) :
        checkpoint = torch.load(model_path)

        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']