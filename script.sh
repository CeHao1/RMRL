
# 0.1
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.1 
python main.py --log-dir log_ppo_res_21 --mode train --residual --total-timesteps 6_000_000

python main.py --log-dir log_ppo_base12 --mode eval --action-bias 0.1 
python main.py --log-dir log_ppo_res_21 --mode eval --action-bias 0.1 

# 0.2
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.2
python main.py --log-dir log_ppo_res_22 --mode train --residual --total-timesteps 6_000_000
python main.py --log-dir log_ppo_res_22 --mode eval --action-bias 0.2

# 0.3
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.3
python main.py --log-dir log_ppo_res_23 --mode train --residual --total-timesteps 6_000_000
python main.py --log-dir log_ppo_res_23 --mode eval --action-bias 0.3

# 0.4
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.4
python main.py --log-dir log_ppo_res_24 --mode train --residual --total-timesteps 6_000_000
python main.py --log-dir log_ppo_res_24 --mode eval --action-bias 0.4

# 0.5
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.5
python main.py --log-dir log_ppo_res_25 --mode train --residual --total-timesteps 6_000_000
python main.py --log-dir log_ppo_res_25 --mode eval --action-bias 0.5

# 0.6
python main.py --log-dir log_ppo_base12 --mode residual_train --action-bias 0.6
python main.py --log-dir log_ppo_res_26 --mode train --residual --total-timesteps 6_000_000
python main.py --log-dir log_ppo_res_26 --mode eval --action-bias 0.6

