# Import required packages

import os.path as osp

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode

from toy.rl.parse import parse_args
from toy.rl.wrappers import *
from toy.rl.evaluation import evaluate_policy
from toy.rl.residual_model import learn_residual_model, load_checkpoint

# Defines a continuous, infinite horizon, task where terminated is always False
# unless a timelimit is reached.

TRAIN = "train"
EVAL = "eval"
RESIDUAL_TRAIN = "residual_train"

def main(args):
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    log_dir = args.log_dir
    rollout_steps = 4800
    algo = args.algo

    mode = args.mode

    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "normalized_dense"
    if args.seed is not None:
        set_random_seed(args.seed)

    def make_env(
        env_id: str,
        max_episode_steps: int = None,
        record_dir: str = None,
        residual_model = None,

    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

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
            if args.action_bias != 0:
                env = ActionBiasWrapper(env, bias=args.action_bias)  
            if args.residual:     
                # =============== create residual model ==================
                residual_model = load_checkpoint('residual_model_checkpoints/checkpoint_episode_68.pth', env)
                env = ResidualModelWrapper(env, residual_model)                                                                              
            return env

        return _init

    #  =============== create video recording directory ==================
    if mode == TRAIN:
        record_dir = osp.join(log_dir, "videos/train")
    elif mode == RESIDUAL_TRAIN:
        record_dir = osp.join(log_dir, "videos/residual_train")
    elif mode == EVAL:
        record_dir = osp.join(log_dir, "videos/eval")

    # =============== create envs ==================
    # nominal eval env
    eval_env = SubprocVecEnv(
        [make_env(env_id, record_dir=record_dir) for _ in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(args.seed)
    eval_env.reset()


    if mode == TRAIN:
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

    elif mode == EVAL:
        env = eval_env
    elif mode == RESIDUAL_TRAIN:
        env = eval_env


    # =============== create policy model ==================
    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(net_arch=[256, 256])
    if algo == "sac":
        model_algo = SAC
    elif algo == "ppo":
        model_algo = PPO

    model = model_algo(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=rollout_steps // num_envs,
        batch_size=400,
        gamma=0.8,
        n_epochs=15,
        tensorboard_log=log_dir,
        target_kl=0.05,
    )

    if mode == EVAL or mode == RESIDUAL_TRAIN:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path)


    # =============== train or eval ==================
    if mode == TRAIN:
        # define callbacks to periodically save our model and evaluate it to help monitor training
        # the below freq values will save every 10 rollouts
        train(args, model, eval_env, log_dir, rollout_steps, num_envs)
        eval(args, model, eval_env)
    elif mode == EVAL:
        eval(args, model, eval_env)
    elif mode == RESIDUAL_TRAIN:
        
        learn_residual_model(env, model)
    

    # close all envs
    eval_env.close()
    if mode == TRAIN:
        env.close()

def train(args, model, eval_env, log_dir, rollout_steps, num_envs):
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10 * rollout_steps // num_envs,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10 * rollout_steps // num_envs,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    # Train an agent with PPO for args.total_timesteps interactions
    model.learn(
        args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    # Save the final model
    model.save(osp.join(log_dir, "latest_model"))

def eval(args, model, eval_env):
    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=10,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 100
    success_rate = success.mean()
    print("Success Rate:", success_rate)




if __name__ == "__main__":
    main(parse_args())
