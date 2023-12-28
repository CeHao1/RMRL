


# train door
python rml/rl/trainer/base_train.py -e "OpenCabinetDoor-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/OpenCabinetDoor-v1" \
--control-mode "base" --model-ids "['1000']"

# train drawer
python rml/rl/trainer/base_train.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/OpenCabinetDrawer-v1" \
--control-mode "base" --model-ids "['1000']"


# train insert
python rml/rl/trainer/base_train.py -e "PegInsertionSide-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/PegInsertionSide-v0" 


# train charger

python rml/rl/trainer/base_train.py -e "PlugCharger-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/dist_baseline/PlugCharger-v0" 