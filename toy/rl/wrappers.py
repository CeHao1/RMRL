import gymnasium as gym
import numpy as np

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

# an action bias wrapper that adds a constant bias to the action
class ActionBiasWrapper(gym.Wrapper):
    def __init__(self, env, bias=0) -> None:
        super().__init__(env)
        self.bias = np.array(bias)
        print('========= set action bias to {} ==========='.format(self.bias))  

    def step(self, action):
        biased_action = action.copy()
        biased_action[0:3] += self.bias  

        # nominal model
        old_state = self.get_state()
        ob_nominal, rew_nominal, terminated_nominal, truncated_nominal, info_nominal = super().step(action)
        self.set_state(old_state)

        # biased model (real)
        ob, rew, terminated, truncated, info = super().step(biased_action)

        info.update({'ob_nominal': ob_nominal})

        return ob, rew, terminated, truncated, info

class ResidualModelWrapper(gym.Wrapper):
    def __init__(self, env, model) -> None:
        super().__init__(env)
        self.model = model

    def step(self, action):
        
        # nominal model
        old_ob = self.get_obs()
        ob, rew, terminated, truncated, info = super().step(action)

        # residual model
        residual_state = self.model(old_ob, action, ob)

        state = self.get_state()
        compensated_state = state + residual_state
        self.set_state(compensated_state)

        # return
        obs = self.get_obs()
        info = self.get_info(obs=obs)
        reward = self.get_reward(obs=obs, action=action, info=info)
        terminated = self.get_done(obs=obs, info=info)
        return obs, reward, terminated, False, info 

