import argparse

import ast

def parse_list(value):
    try:
        # Safely evaluate the string as a Python literal (list, in this case)
        return ast.literal_eval(value)
    except:
        # Handle the case where the string is not a valid Python literal
        raise argparse.ArgumentTypeError("Invalid list format")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=50,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4800,
        help="Number of steps per rollout",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices = ["ee", "base"],
        default="ee",
        help="control mode to use",
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )

    parser.add_argument(
        "--model-ids",
        type=parse_list,
        default="[]",
        help="list of model ids to evaluate",
    )

    args = parser.parse_args()
    return args
