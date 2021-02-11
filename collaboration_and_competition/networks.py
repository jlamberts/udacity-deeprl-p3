import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor network for multi-agent DDPG.

    Converts state -> optimal action"""

    def __init__(
        self,
        state_size=24,
        action_size=2,
        hidden_layer_size=64,
        seed=42,
        batchnorm_inputs=True,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_size (int): Size of the hidden layer
            seed (int): Random seed
            batchnorm_inputs (bool): if True, apply batch normalization to the inputs
                Per Lillicrap et al (2016) this can help training and generalization with different physical dimensions for inputs
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.batchnorm_layer = nn.BatchNorm1d(state_size) if batchnorm_inputs else None

        self.inputs_actor = nn.Linear(state_size, hidden_layer_size)
        self.hidden_actor = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.actions = nn.Linear(hidden_layer_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> policy"""
        if self.batchnorm_layer:
            state = self.batchnorm_layer(state)

        # calculate policy using actor network
        policy = self.inputs_actor(state)
        policy = F.relu(policy)
        policy = self.hidden_actor(policy)
        policy = F.relu(policy)

        # tanh will give us an output in the range (-1, 1)
        actions = F.tanh(self.actions(policy))

        return actions


class CriticNetwork(nn.Module):
    """Critic network for multi-agent DDPG.

    Takes in all states + all actions and returns a Q-value for those state+action combinations."""

    def __init__(
        self,
        state_size=24,
        num_agents=2,
        action_size=2,
        hidden_layer_size=64,
        seed=42,
        batchnorm_inputs=True,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            num_agents (int): Number of agents
            hidden_layer_size (int): Size of the hidden layer
            seed (int): Random seed
            batchnorm_inputs (bool): if True, apply batch normalization to the inputs
                Per Lillicrap et al (2016) this can help training and generalization with different physical dimensions for inputs
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.num_agents = num_agents

        self.batchnorm_layer = (
            nn.BatchNorm1d((state_size + action_size) * num_agents)
            if batchnorm_inputs
            else None
        )

        self.inputs_critic = nn.Linear(
            (state_size + action_size) * num_agents, hidden_layer_size
        )
        self.hidden_critic = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.outputs_critic = nn.Linear(hidden_layer_size, 1)

    def forward(self, states, actions):
        """Build a network that maps states, action -> q-value.

        Expects states and actions from both agents."""

        combined_input = torch.cat((states, actions), dim=-1)
        if self.batchnorm_layer:
            state = self.batchnorm_layer(combined_input)
        # calculate policy using actor network|
        value = self.inputs_critic(combined_input)
        value = F.relu(value)
        value = self.hidden_critic(value)
        value = F.relu(value)
        value = self.outputs_critic(value)
        return value