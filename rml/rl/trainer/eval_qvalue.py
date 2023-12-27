
from rml.envs.base_env import base_env
from rml.rl.build_rl import build_rl
from rml.args.arg_parse import parse_args

from rml.rl.trainer.evaluation import evaluate_policy_for_q
import numpy as np

def eval_q(args):
    args.eval = True
    env, eval_env = base_env(args)

    assert args.model_path is not None, "Must specify model path for evaluation"

    model, eval_callback, checkpoint_callback = build_rl(
        args, env, eval_env, args.log_dir, args.rollout_steps, args.n_envs)
    
    returns, ep_lens = evaluate_policy_for_q(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=10,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)

if __name__ == "__main__":
    args =  parse_args()
    eval_q(args)
