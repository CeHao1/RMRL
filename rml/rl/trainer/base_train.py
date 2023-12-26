from stable_baselines3.common.utils import set_random_seed

from rml.envs.base_env import base_env
from rml.rl.build_rl import build_rl
from rml.args.arg_parse import parse_args

def train(args):    
    
    # build env
    env, eval_env = base_env(args)

    # build agent
    model, eval_callback, checkpoint_callback = build_rl(
        args, env, eval_env, args.log_dir, args.rollout_steps, args.n_envs
    )

    # train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    # save model
    model.save(args.log_dir + "/latest_model")

    # close env
    env.close()


if __name__ == "__main__":
    args =  parse_args()
    train(args)