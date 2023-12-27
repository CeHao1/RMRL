

# input a base policy agent
# input a training agent, only has critic
from stable_baselines3.common.utils import set_random_seed

from rml.envs.base_env import base_env
from rml.rl.build_rl import build_rl
from rml.args.arg_parse import parse_args

from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def train(args):    
    
    # build env
    env, eval_env = base_env(args)

    # load base policy model
    policy_model, eval_callback, checkpoint_callback = build_rl(
        args, env, eval_env, args.log_dir, args.rollout_steps, args.n_envs
    )
    print('========== load base policy ==========')

    # build q value model
    args.algo = "sen"
    args.eval = False
    args.model_path = None
    model, eval_callback, checkpoint_callback = build_rl(
        args, env, eval_env, args.log_dir, args.rollout_steps, args.n_envs)
    print('========== load q value model ==========')



    policy_model.actor.name = 'base policy'
    model.actor.name = 'q value model'

    model.setup_actor(policy_model.actor)
    print('========== setup actor ==========')


    # '''
    if not args.eval:
        # train
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
        )

        # save model
        model.save(args.log_dir + "/latest_model")
    # '''

    '''
    # evaluate
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
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)
    '''


    # close env
    env.close()


if __name__ == "__main__":
    args =  parse_args()
    train(args)


