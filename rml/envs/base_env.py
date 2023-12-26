
# Import required packages
import os.path as osp

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode

from rml.envs.wrapper import ContinuousTaskWrapper, SuccessInfoWrapper


def base_env(args):
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    log_dir = args.log_dir
    rollout_steps = 4800

    if len(args.model_ids) > 0:
        print("args. model_ids: ", args.model_ids)

    obs_mode = "state"

    if args.control_mode == "ee":
        control_mode = "pd_ee_delta_pose"
    elif args.control_mode == "base":
        control_mode = "base_pd_joint_vel_arm_pd_ee_delta_pose"

    reward_mode = "normalized_dense"
    if args.seed is not None:
        set_random_seed(args.seed)

    def make_env(
        env_id: str,
        max_episode_steps: int = None,
        record_dir: str = None,
    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            if len(args.model_ids) > 0:     # if model_ids is not empty, then use it
                env = gym.make(
                    env_id,
                    obs_mode=obs_mode,
                    reward_mode=reward_mode,
                    control_mode=control_mode,
                    render_mode="cameras",
                    max_episode_steps=max_episode_steps,
                    model_ids = args.model_ids,
                )
            else:
                env = gym.make(
                    env_id,
                    obs_mode=obs_mode,
                    reward_mode=reward_mode,
                    control_mode=control_mode,
                    render_mode="cameras",
                    max_episode_steps=max_episode_steps,
                )   
            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(env, record_dir, info_on_video=True)
            return env

        return _init

    # create eval environment
    if args.eval:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")
    eval_env = SubprocVecEnv(
        [make_env(env_id, record_dir=record_dir) for _ in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(args.seed)
    eval_env.reset()

    if args.eval:
        env = eval_env
    else:
        # Create vectorized environments for training
        env = SubprocVecEnv(
            [
                make_env(env_id, max_episode_steps=max_episode_steps)
                for _ in range(num_envs)
            ]
        )
        env = VecMonitor(env)
        env.seed(args.seed)
        env.reset()

    return env, eval_env