import numpy as np
import random
from collections import namedtuple, deque
from agent import Agent

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

LR = 5e-4               # learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, hidden_layer_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            hidden_layer_size: Dimension of the hidden layer
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        actions = self.fc1(state)
        actions = F.relu(actions)
        actions = self.fc2(actions)
        actions = F.relu(actions)
        actions = self.fc3(actions)

        return actions


class DqnAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, gamma=0.99, update_every=4, tau=1e-3, batch_size=64, mode=Agent.TRAIN, hidden_layer_size=64):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            gamma (float): discount factor
            update_every (int): how often to update the network
            tau (float): for soft update of target parameters
            batch_size (int): minibatch size
            mode (boolean): mode activated (0 for training, 1 for playing)
            hidden_layer_size (int): dimension of the hidden layers
        """
        super(DqnAgent, self).__init__(state_size, action_size, seed, gamma, update_every, tau, batch_size, mode)
        self.name = f'DqnAgent'

        # Q-Network
        self.hidden_layer_size = hidden_layer_size
        self.qnetwork_local = QNetwork(state_size, hidden_layer_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, hidden_layer_size, action_size, seed).to(device)
        self.qnetwork_target.eval()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


class DoubleDqnAgent(DqnAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, gamma=0.99, update_every=4, tau=1e-3, batch_size=64, mode=Agent.TRAIN, hidden_layer_size=64):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            gamma (float): discount factor
            update_every (int): how often to update the network
            tau (float): for soft update of target parameters
            batch_size (int): minibatch size
            mode (boolean): mode activated (0 for training, 1 for playing)
            hidden_layer_size (int): dimension of the hidden layers
        """
        super(DoubleDqnAgent, self).__init__(state_size, action_size, seed,
                                             gamma, update_every, tau, batch_size, mode, hidden_layer_size)

        self.name = f'DoubleDqnAgent'

        # activate double dqn
        self.use_double_dqn = True
