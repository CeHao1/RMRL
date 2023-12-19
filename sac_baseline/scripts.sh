#!/bin/bash

# "PandaAvoidObstacles-v0" # not manipulation

env_id=(
    # "LiftCube-v0"
    # "PickCube-v0"
    "StackCube-v0"
    # "PickSingleYCB-v0"
    # "PickSingleEGAD-v0"
    # "PickClutterYCB-v0"
    # "AssemblingKits-v0"
    "PegInsertionSide-v0"
    # "PlugCharger-v0"
    # "TurnFaucet-v0"
    # "OpenCabinetDoor-v1" # different control mode
    # "OpenCabinetDrawer-v1"
    # "Hang-v0"
    # "Pour-v0"
)

count=0
for env in ${env_id[@]}
do
    python src/run_baseline.py -e "$env" --total-timesteps 5_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/$env" &

    # Increment the counter
    ((count++))

    # Every three jobs, wait for them to finish before continuing
    if (( count % 2 == 0 )); then
        wait
    fi
done

# Wait for the last set of jobs if they don't make a complete group of three
wait

# python src/run_baseline.py -e "PegInsertionSide-v0"  -n 1 --algo "sac" --log-dir "./log/sac_baseline/PegInsertionSide-v0" --eval