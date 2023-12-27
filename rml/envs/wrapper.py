import gymnasium as gym
import numpy as np

# Defines a continuous, infinite horizon, task where terminated is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info

class ActionNoiseWrapper(gym.Wrapper):

    def __init__(self, env, noise_fun):
        super().__init__(env)
        self.noise_fun = noise_fun

    def step(self, action):
        noise = self.noise_fun(action)
        # print("noise: ", noise)
        action_noise = action + noise
        return self.env.step(action_noise)


class GaussianNoise:

    def __init__(self, mu, sig, shape=None):
        self.mu = mu
        self.sig = sig
        self.shape = shape

    def __call__(self, variable):
        if self.shape is None:
            shape = variable.shape
        else:
            shape = self.shape
        return np.random.normal(self.mu, self.sig, shape)
    
    property
    def shape(self):
        return self.shape