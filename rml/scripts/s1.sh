
python rml/rl/trainer/base_train.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickCube-v02"  



python rml/rl/trainer/base_train.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickCube-v02-back" \
--model-path "./log/sac_baseline/PickCube-v02-back/latest_model" 



# train q value model
python rml/rl/trainer/train_qvalue.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickCube-sen"  \
--model-path "./log/sac_baseline/PickCube-v02-back/latest_model" 


# load q value model and visualize
python rml/rl/trainer/eval_qvalue.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 1 --algo "sen" --log-dir "./log/sac_baseline/PickCube-sen"  \
--model-path "./log/sac_baseline/PickCube-sen/latest_model" 