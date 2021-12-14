import copy

import torch.nn as nn


class MarioNet(nn.Module) :
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84 :
            raise ValueError('Expecting Height : 84, input {}'.format(h))
        if w != 84 :
            raise ValueError('Expecting Width : 84, input {}'.format(w))
        
        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters() :
            p.requires_grad = False

    def forward(self, x, model) :
        if model == 'online' :
            return self.online(x)
        elif model == 'target' :
            return self.target(x)