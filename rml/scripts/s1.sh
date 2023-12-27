
python rml/rl/trainer/base_train.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickCube-v02"  \
--eval



python rml/rl/trainer/base_train.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickCube-v02-back" \
--model-path "./log/sac_baseline/PickCube-v02-back/latest_model" 



# train q value model
python rml/rl/trainer/train_qvalue.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickCube-sen2"  \
--model-path "./log/sac_baseline/PickCube-v02-back/latest_model" 


# load q value model and visualize
python rml/rl/trainer/eval_qvalue.py -e "PickCube-v0" --total-timesteps 10_000_000 -n 1 --algo "sen" --log-dir "./log/sac_baseline/PickCube-sen"  \
--model-path "./log/sac_baseline/PickCube-sen/latest_model" 


## door
# train 
python rml/rl/trainer/base_train.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/sac_baseline/OpenCabinetDoor-v1" \
--control-mode "base" --model-ids "['1000']" --eval

# eval
python rml/rl/trainer/base_train.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/sac_baseline/OpenCabinetDoor-v1"  --control-mode "base" --model-ids "['1000']"

python rml/rl/trainer/train_qvalue.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/sac_baseline/OpenCabinetDoor-v1-sen1"  --control-mode "base" --model-ids "['1000']"  \
--model-path "./log/sac_baseline/OpenCabinetDoor-v1/latest_model" 

python rml/rl/trainer/eval_qvalue.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 1 --algo "sen" \
--log-dir "./log/sac_baseline/OpenCabinetDoor-v1-sen1" --control-mode "base" --model-ids "['1000']" \
--model-path "./log/sac_baseline/OpenCabinetDoor-v1-sen1/latest_model" 
