# RMRL
residual model reinforcement learning



python main.py --log-dir log_ppo_base10 --mode train 

python main.py --log-dir log_ppo_base3 --mode eval --action-bias 0.4

python main.py --log-dir log_ppo_base3 --mode residual_train --action-bias 0.5 

```

--action-bias
--residual

```

1. train base policy 
python main.py --log-dir log_ppo_base12 --mode train --total-timesteps 6_000_000


2. eval in bias
python main.py --log-dir log_ppo_base12 --mode eval --action-bias 0.5


3. learn residual in bias
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.1 

4. train policy in residual
python main.py --log-dir log_ppo_res_12 --mode train --residual

5. eval in bias
python main.py --log-dir log_ppo_res_10 --mode residual_train --action-bias 0.5 
