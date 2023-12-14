# RMRL
residual model reinforcement learning


# Training
python toy/rl/pick_baseline.py --log-dir log_ppo1

# Evaluation
python toy/rl/pick_baseline.py --eval --model-path=logs_ppo1/best_model