import time
import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger :
    def __init__(self, save_dir) -> None:
        self.writer = SummaryWriter(save_dir + '/')

        self.init_episode()

        self.record_time = time.time()
    
    def log_step(self, reward, loss, q) :
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        
        if loss :
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self) :
        if self.curr_ep_loss_length == 0 :
            ep_avg_loss = 0
            ep_avg_q = 0
        else :
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)

        self.writer.add_scalar('Episode Total Reward', self.curr_ep_reward)
        self.writer.add_scalar('Episode Length', self.curr_ep_length)
        self.writer.add_scalar('Episode Average Loss', ep_avg_loss)
        self.writer.add_scalar('Episode Average Q', ep_avg_q)
        
    def init_episode(self) :
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step) :
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f'Episode {episode} - '
            f'Step {step} - '
            f'Epsilon {epsilon} - '
            f'Time Delta {time_since_last_record} - '
            f'Time {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}'
        )

    def close(self) :
        self.writer.close()

class MetricLogger :
    def __init__(self) -> None:
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()

        self.record_time = time.time()

    def log_step(self, reward, loss, q) :
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        
        if loss :
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self) :
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_length == 0 :
            ep_avg_loss = 0
            ep_avg_q = 0
        else :
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)

        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

    def init_episode(self) :
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step) :
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f'Episode {episode} - '
            f'Step {step} - '
            f'Epsilon {epsilon} - '
            f'Mean Reward {mean_ep_reward} - '
            f'Mean Length {mean_ep_length} - '
            f'Mean Loss {mean_ep_loss} - '
            f'Mean Q Value {mean_ep_q} - '
            f'Time Delta {time_since_last_record} - '
            f'Time {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}'
        )
