


# ================================== train door
python rml/rl/trainer/base_train.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/OpenCabinetDoor-v1" \
--control-mode "base" --model-ids "['1000']" --eval

# 1. train q with noise
python rml/rl/trainer/train_qvalue.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/OpenCabinetDoor-v1-sen4"  --control-mode "base" --model-ids "['1000']"  \
--model-path "./log/dist_baseline/OpenCabinetDoor-v1/latest_model" --noise_std 0.3 --reward-mode "sparse"


# 2. eval q without noise
python rml/rl/trainer/eval_qvalue.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 1 --algo "sen" \
--log-dir "./log/dist_baseline/OpenCabinetDoor-v1-sen4" --control-mode "base" --model-ids "['1000']" \
--model-path "./log/dist_baseline/OpenCabinetDoor-v1-sen4/latest_model"  --reward-mode "sparse" --eval



# ================================== train drawer
python rml/rl/trainer/base_train.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/OpenCabinetDrawer-v1" \
--control-mode "base" --model-ids "['1000']" --eval --noise_std 0.5

# 1. train q with noise
python rml/rl/trainer/train_qvalue.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/OpenCabinetDrawer-v1-sen4"  --control-mode "base" --model-ids "['1000']"  \
--model-path "./log/dist_baseline/OpenCabinetDrawer-v1/latest_model" --noise_std 0.5 --reward-mode "sparse"


# 2. eval q without noise
python rml/rl/trainer/eval_qvalue.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 1 --algo "sen" \
--log-dir "./log/dist_baseline/OpenCabinetDrawer-v1-sen4" --control-mode "base" --model-ids "['1000']" \
--model-path "./log/dist_baseline/OpenCabinetDrawer-v1-sen4/latest_model"  --reward-mode "sparse"



# ========================== train insert
python rml/rl/trainer/base_train.py -e "PegInsertionSide-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/PegInsertionSide-v0" 


# train charger

python rml/rl/trainer/base_train.py -e "PlugCharger-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/PlugCharger-v0" 