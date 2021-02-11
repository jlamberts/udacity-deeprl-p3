from collaboration_and_competition.networks import ActorNetwork, CriticNetwork
from collaboration_and_competition.env import UnityEnvWrapper
from collaboration_and_competition.utils import (
    ReplayBuffer,
    OUNoise,
    soft_update,
    update_adaptive_noise,
    mirror_states_,
)
import pandas as pd
import torch
import torch.nn.functional as F

# Set Unity settings
UNITY_ENV_PATH = "Tennis_Linux_NoVis/Tennis.x86_64"
TRAIN_MODE = True

# Training loop params
NUM_EPISODES = 5000
TAU = 1e-3
GAMMA = 0.99
UPDATE_EVERY = 1
RANDOM_SEED = 42
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256

# OU noise params
NOISE_DECAY_RATE = None
ADAPTIVE_NOISE_TARGET_SCORE = 0.25
NOISE_LIMIT = 1.0
SIGMA = 0.5

# Network params
ACTOR_LR = 1e-3
CRITIC_LR = 3e-3
ACTOR_HIDDEN_LAYER_SIZE = 128
CRITIC_HIDDEN_LAYER_SIZE = 128
BATCHNORM_INPUTS = False


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize environment
    env = UnityEnvWrapper(UNITY_ENV_PATH, train_mode=TRAIN_MODE)

    # initialize main and target networks
    actor = ActorNetwork(
        batchnorm_inputs=BATCHNORM_INPUTS, hidden_layer_size=ACTOR_HIDDEN_LAYER_SIZE
    )
    critic = CriticNetwork(
        batchnorm_inputs=BATCHNORM_INPUTS, hidden_layer_size=CRITIC_HIDDEN_LAYER_SIZE
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # soft updating with tau=1 is equivalent to "hard updating"; we want our networks to start out with identical weights
    actor_target = ActorNetwork(
        batchnorm_inputs=BATCHNORM_INPUTS, hidden_layer_size=ACTOR_HIDDEN_LAYER_SIZE
    )
    soft_update(actor, actor_target, 1.0)

    critic_target = CriticNetwork(
        batchnorm_inputs=BATCHNORM_INPUTS, hidden_layer_size=CRITIC_HIDDEN_LAYER_SIZE
    )
    soft_update(critic, critic_target, 1.0)

    # we will use a single replay buffer and OU process for both actors
    buffer = ReplayBuffer(
        action_size=4, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=RANDOM_SEED
    )
    ou_noise = OUNoise(
        action_dimension=4, scale=NOISE_LIMIT, decay_rate=NOISE_DECAY_RATE, sigma=SIGMA
    )

    all_scores = []
    for episode in range(NUM_EPISODES):
        episode_score = torch.zeros(2)
        env.reset()
        ou_noise.reset()
        states = torch.tensor(env.states, dtype=torch.float, device=device).view(1, -1)
        mirror_states_(states)
        done = False
        time_step = 0

        while not done:
            # get noisy actions from actor, then take them in the env
            with torch.no_grad():
                actions = torch.clamp(
                    torch.cat((actor(states[:, :24]), actor(states[:, 24:])), dim=-1)
                    + ou_noise.noise(),
                    -1,
                    1,
                )
            next_states, reward, done = env.step(actions.numpy())

            # clean up and reshape results of step
            next_states = next_states.view(1, -1)
            mirror_states_(next_states)
            episode_score = episode_score + reward.t()
            # since we are using reward with the critic, which knows the whole env, we can just sum both agents' rewards
            reward = reward.sum()
            done = any(done)

            # add experience to replay buffer
            buffer.add(states, actions, reward, next_states, done)
            states = next_states

            # train agent if eligible
            if time_step % UPDATE_EVERY == 0 and len(buffer.memory) >= BATCH_SIZE:
                (
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                ) = buffer.sample()

                # update critic
                Q = critic(batch_states, batch_actions)
                batch_next_actions = torch.cat(
                    (
                        actor_target(batch_next_states[:, :24]),
                        actor_target(batch_next_states[:, 24:]),
                    ),
                    dim=-1,
                ).detach()
                Q_t_1 = critic_target(batch_next_states, batch_next_actions)
                critic_loss = F.mse_loss(
                    Q, batch_rewards + GAMMA * Q_t_1 * (1 - batch_dones)
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
                critic_optimizer.step()

                # update actors
                actor_1_actions = actor(batch_states[:, :24])
                actor_2_actions = actor(batch_states[:, 24:])

                Q = critic(
                    batch_states, torch.cat((actor_1_actions, actor_2_actions), dim=-1)
                )
                actor_gradient = -Q.mean()
                actor_optimizer.zero_grad()
                actor_gradient.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
                actor_optimizer.step()

                # soft update target networks
                soft_update(actor, actor_target, TAU)
                soft_update(critic, critic_target, TAU)

            time_step += 1
        ou_noise.decay()
        all_scores.append(episode_score)
        if episode % 50 == 0 and episode >= 100:
            rolling_score = torch.cat(all_scores[-100:]).numpy().max(1).mean()
            print(f"episode {episode} rolling avg:", rolling_score)

            if ADAPTIVE_NOISE_TARGET_SCORE:
                new_noise_weight = update_adaptive_noise(
                    rolling_score, ADAPTIVE_NOISE_TARGET_SCORE, NOISE_LIMIT
                )
                print("updated noise weight to", new_noise_weight)
                ou_noise.scale = new_noise_weight

    # save results
    pd.DataFrame(
        torch.cat(all_scores).numpy(), columns=["left_player", "right_player"]
    ).to_csv("scores/training_scores.csv", index=False)

    torch.save(actor.state_dict(), "model_weights/actor_weights.pt")
    torch.save(critic.state_dict(), "model_weights/critic_weights.pt")