
import os.path as osp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rml.rl.agents.sensitive import Sensitive
from rml.rl.agents.sac import SAC

def build_rl(args, env, eval_env, log_dir, rollout_steps, num_envs):
    # Define the policy configuration and algorithm configuration
    if args.algo == "sac":  
        policy_kwargs = dict(net_arch=[256, 256])
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=400,
            gamma=0.8,
            tensorboard_log=log_dir,
        )
    elif args.algo == "ppo":
        policy_kwargs = dict(net_arch=[256, 256])
        model = PPO(
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
    elif args.algo == "sen":
        policy_kwargs = dict(net_arch=[256, 256])
        model = Sensitive(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=400,
            gamma=0.99,
            tensorboard_log=log_dir,
            learning_rate = 1e-3,
        )

    if args.eval or args.model_path is not None:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path, env=env)
    # else:
    # define callbacks to periodically save our model and evaluate it to help monitor training
    # the below freq values will save every 10 rollouts
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

    return model, eval_callback, checkpoint_callback