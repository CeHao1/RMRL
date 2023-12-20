

# pick single ycb
python src/run_baseline.py -e "PickSingleYCB-v0" --total-timesteps 5_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickSingleYCB-single" \
--model-ids "['024_bowl']"

# pick multi ycb
python src/run_baseline.py -e "PickSingleYCB-v0" --total-timesteps 5_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PickSingleYCB-multi" 


# drawer single
python src/run_baseline.py -e "OpenCabinetDoor-v1" --total-timesteps 5_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/OpenCabinetDoor-single" \
--control-mode "base" --model-ids "['1000']"

# insert 
python src/run_baseline.py -e "PegInsertionSide-v0" --total-timesteps 10_000_000 -n 30 --algo "sac" --log-dir "./log/sac_baseline/PegInsertionSide-longer" 