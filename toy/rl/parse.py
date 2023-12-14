import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=24,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(    
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm to use",
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
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval", "residual_train"]
    )
    
    parser.add_argument(
        "--action-bias",
        type=float,
        default=0,
        help="Constant bias to add to the action",
    )

    parser.add_argument(
        "--residual", action="store_true", help="whether include residual to model"
    )

    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args