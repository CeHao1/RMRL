# RMRL
residual model reinforcement learning



python main.py --log-dir log_ppo_base2 --mode train 

python main.py --log-dir log_ppo_base3 --mode eval --action-bias 0.5

python main.py --log-dir log_ppo_base3 --mode residual_train --action-bias 0.5 

```

--action-bias
--residual

```