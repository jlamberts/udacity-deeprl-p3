from train_agent import ACTOR_HIDDEN_LAYER_SIZE
from collaboration_and_competition.networks import ActorNetwork
from collaboration_and_competition.env import UnityEnvWrapper
from collaboration_and_competition.utils import mirror_states_

import torch

# Set Unity settings
UNITY_ENV_PATH = "Tennis_Linux_NoVis/Tennis.x86_64"
TRAIN_MODE = True

# Set number of episodes to run
NUM_EPISODES = 10

# Set Pretrained Model Info Here
ACTOR_HIDDEN_LAYER_SIZE = 128
ACTOR_BATCHNORM = False
WEIGHTS_PATH = "model_weights/actor_weights.pt"


if __name__ == "__main__":
    env = UnityEnvWrapper(file_name=UNITY_ENV_PATH, train_mode=TRAIN_MODE)
    actor = ActorNetwork(
        hidden_layer_size=ACTOR_HIDDEN_LAYER_SIZE, batchnorm_inputs=ACTOR_BATCHNORM
    )
    actor.load_state_dict(torch.load(WEIGHTS_PATH))

    for episode in range(NUM_EPISODES):
        all_rewards = torch.zeros(2)
        env.reset()
        states = torch.tensor(env.states, dtype=torch.float).view(1, -1)
        mirror_states_(states)

        done = [False]

        while not any(done):

            actions = torch.cat((actor(states[:, :24]), actor(states[:, 24:])), dim=-1)
            states, reward, done = env.step(actions.detach().numpy())

            states = states.view(1, -1)
            mirror_states_(states)

            all_rewards = all_rewards + reward.t()

        print(f"Rewards for episode {episode}: {all_rewards}")