import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gymnasium as gym

import os



# Neural Network Model Definition
class DynamicModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DynamicModel, self).__init__()
        self.fc1 = nn.Linear(observation_space + action_space, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, observation_space)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to save a model checkpoint
def save_checkpoint(model, optimizer, epoch, filename="model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, env):
    model = create_residual_model(env)
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']

    return model

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, observation, action, next_observation):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (observation, action, next_observation)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
learning_rate = 0.001
batch_size = 64
capacity = 10000
num_episodes = 500

def create_residual_model(env):
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    model = DynamicModel(observation_space, action_space)
    
    return model

def learn_residual_model(env, policy):
    # Model, optimizer, and replay buffer instantiation
    model = create_residual_model(env)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity)

    # Training Loop
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        total_loss = 0
        num_steps = 0

        while not done:
            
            actions, states = policy.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=False,
            )

            new_observations, rewards, dones, infos = env.step(actions)

            next_observation = new_observations[0]
            observation = observations[0]
            action = actions[0]
            done = dones[0]
            info = infos[0]

            episode_starts = dones

            residual_obs = next_observation - info['ob_nominal']

            # Store in replay buffer
            replay_buffer.push(torch.tensor([observation], dtype=torch.float32), 
                               torch.tensor([action], dtype=torch.float32), 
                               torch.tensor([residual_obs], dtype=torch.float32)) # target is residual obs

            if len(replay_buffer) > batch_size:
                # Sample a batch and update the model
                transitions = replay_buffer.sample(batch_size)
                batch_obs, batch_act, batch_next_obs = zip(*transitions)

                batch_obs = torch.cat(batch_obs)
                batch_act = torch.cat(batch_act)
                batch_next_obs = torch.cat(batch_next_obs)

                # Compute loss
                predicted_next_obs = model(batch_obs, batch_act)
                loss = nn.MSELoss()(predicted_next_obs, batch_next_obs)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store loss
                total_loss += loss.item()
                num_steps += 1

            observations = new_observations  

        #  save checkpoint
        save_every = 2
        checkpoint_dir = 'residual_model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        if episode % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pth')
            save_checkpoint(model, optimizer, episode, filename=checkpoint_path)

            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_last.pth')
            save_checkpoint(model, optimizer, episode, filename=checkpoint_path)
            print(f"Checkpoint saved at episode {episode}")


        # Print average loss
        average_loss = total_loss / num_steps if num_steps > 0 else 0
        print(f"Episode {episode}, Average Loss: {average_loss:.4f}")


