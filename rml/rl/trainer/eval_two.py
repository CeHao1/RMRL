

from rml.envs.base_env import base_env
from rml.rl.build_rl import build_rl
from rml.args.arg_parse import parse_args

from rml.rl.trainer.evaluation import evaluate_policy_for_q
import numpy as np

import torch as th

def eval_two(args):
    args.eval = True


    env, eval_env = base_env(args)
    # load model 1
    args.model_path =  "./log/q_test/nonoise_sparse/latest_model"
    model1, eval_callback, checkpoint_callback = build_rl(
        args, env, eval_env, args.log_dir, args.rollout_steps, args.n_envs)

    # load model 2
    args.model_path =  "./log/q_test/noised_sparse/latest_model"
    model2, eval_callback, checkpoint_callback = build_rl(
        args, env, eval_env, args.log_dir, args.rollout_steps, args.n_envs)
    
    # load replay
    for idx in range(10):
        replay = np.load('./log/qvalue_plot/replay_{}.npz'.format(idx))

        obs = replay['obs']
        act = replay['act']

        episode_qvalue1 = {'mean':[], 'std':[]}
        episode_qvalue2 = {'mean':[], 'std':[]}

        for o, a in zip(obs, act):
            # process q
            obs_torch = th.tensor(o, device=model1.device)
            actions_torch = th.tensor(a, device=model1.device)
            # shape tuple (2, x)

            # model 1
            qvalue_dist, qvalue, qvalue_mean, qvalue_std = model1.critic.get_q_dist(obs_torch, actions_torch)
            episode_qvalue1['mean'].append(tensor_to_numpy(qvalue_mean[0].squeeze()))
            episode_qvalue1['std'].append(tensor_to_numpy(qvalue_std[0].squeeze()))

            # model 2
            qvalue_dist, qvalue, qvalue_mean, qvalue_std = model2.critic.get_q_dist(obs_torch, actions_torch)
            episode_qvalue2['mean'].append(tensor_to_numpy(qvalue_mean[0].squeeze()))
            episode_qvalue2['std'].append(tensor_to_numpy(qvalue_std[0].squeeze()))

        np.savez('./log/qvalue_plot/qvalue1_{}.npz'.format(idx), **episode_qvalue1)
        np.savez('./log/qvalue_plot/qvalue2_{}.npz'.format(idx), **episode_qvalue2)


def tensor_to_numpy(arr):
    if isinstance(arr, th.Tensor):
        return arr.detach().cpu().numpy()
    return arr

if __name__ == "__main__":
    args =  parse_args()
    eval_two(args)