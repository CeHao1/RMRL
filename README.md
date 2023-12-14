# RMRL
residual model reinforcement learning



python main.py --log-dir log_ppo_base2 --mode train 

python main.py --log-dir log_ppo_base3 --mode eval --action-bias 0.4

python main.py --log-dir log_ppo_base3 --mode residual_train --action-bias 0.5 

```

--action-bias
--residual

```

1. train base policy 
2. eval in bias
3. learn residual in bias
4. train policy in residual

python main.py --log-dir log_ppo_base2 --mode train 

5. eval in bias

