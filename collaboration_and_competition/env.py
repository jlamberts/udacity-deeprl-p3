from unityagents import UnityEnvironment
import numpy as np
import torch


class UnityEnvWrapper:
    """Little helper class to initialize a unity env and make it a little easier to interact with."""

    def __init__(
        self, file_name="Reacher_Linux_NoVis/Reacher.x86_64", train_mode=True, **kwargs
    ):
        self.env = UnityEnvironment(file_name=file_name, **kwargs)
        self._env_info = None
        self.train_mode = train_mode
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.states.shape[1]

    @property
    def env_info(self):
        if not self._env_info:
            self._env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self._env_info

    @property
    def states(self):
        return self.env_info.vector_observations

    @property
    def step_tuple(self):
        """(next_state, reward, done)"""
        return (
            torch.tensor(self.env_info.vector_observations, dtype=torch.float),
            torch.tensor(self.env_info.rewards, dtype=torch.float).unsqueeze(-1),
            self.env_info.local_done,
        )

    def get_random_actions(self, n_agents=None, clip=True):
        """Get random actions for `n_agents` agents, sampled from the random normal distribution.

        If n_agents is not provided, one random action will be generated per agent in the environment.
        If `clip` is set to True, values will be clipped between [-1,1]
        """
        if not n_agents:
            n_agents = self.num_agents
        unclipped = np.random.randn(n_agents, self.action_size)
        return np.clip(unclipped, -1, 1) if clip else unclipped

    def close(self):
        self.env.close()

    def step(self, actions):
        """(next_state, reward, done)"""
        self._env_info = self.env.step(actions)[self.brain_name]
        return self.step_tuple

    def reset(self, train_mode=True):
        self._env_info = self.env.reset(train_mode=train_mode)[self.brain_name]