#!/bin/bash

# "PandaAvoidObstacles-v0" # not manipulation

env_id=(
    # "LiftCube-v0"
    # "PickCube-v0"
    # "StackCube-v0"
    # "PickSingleYCB-v0"
    # "PickSingleEGAD-v0"
    # "PickClutterYCB-v0"
    # "AssemblingKits-v0"
    # "PegInsertionSide-v0"
    # "PlugCharger-v0"
    # "TurnFaucet-v0"
    # "OpenCabinetDoor-v1" # different control mode
    # "OpenCabinetDrawer-v1"
    # "Hang-v0"
    # "Pour-v0"
)

# count=0
# for env in ${env_id[@]}
# do
#     python src/run_baseline.py -e "$env" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/$env" --control-mode "base" &

#     # Increment the counter
#     ((count++))

#     # Every three jobs, wait for them to finish before continuing
#     if (( count % 1 == 0 )); then
#         wait
#     fi
# done

# Wait for the last set of jobs if they don't make a complete group of three
# wait



# python src/run_baseline.py -e "OpenCabinetDoor-v1" --total-timesteps 5_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/OpenCabinetDoor-v1" \
# --control-mode "base" --model-ids "['1000']"

python src/run_baseline.py -e "OpenCabinetDoor-v1" --total-timesteps 8_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/OpenCabinetDoor-multi" \
--control-mode "base" 

python src/run_baseline.py -e "OpenCabinetDrawer-v1" --total-timesteps 8_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/OpenCabinetDrawer-multi" \
--control-mode "base" 

# python src/run_baseline.py -e "OpenCabinetDoor-v1"  -n 1 --algo "sac" --log-dir "./log/sac_baseline/111" --control-mode "base" --model-ids "['1000']"

# python src/run_baseline.py -e "LiftCube-v0"  -n 1 --algo "sac" --log-dir "./log/sac_baseline/111" 

# python -m mani_skill2.utils.download_asset "OpenCabinetDoor-v1"