

# train dense/sparse q without noise


python rml/rl/trainer/train_qvalue.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/q_test/nonoise_sparse"  --control-mode "base" --model-ids "['1000']"  \
--model-path "./log/dist_baseline/OpenCabinetDrawer-v1/latest_model" --noise_std 0.0 --reward-mode "sparse"

# train dense/sparse q with noise

python rml/rl/trainer/train_qvalue.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 30 --algo "sac" \
--log-dir "./log/q_test/noised_sparse"  --control-mode "base" --model-ids "['1000']"  \
--model-path "./log/dist_baseline/OpenCabinetDrawer-v1/latest_model" --noise_std 0.5 --reward-mode "sparse"

# eval the q value

python rml/rl/trainer/eval_qvalue.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 1 --algo "sen" \
--log-dir "./log/q_test/noised_sparse" --control-mode "base" --model-ids "['1000']" \
--model-path "./log/q_test/noised_sparse/latest_model"  --reward-mode "sparse"

python rml/rl/trainer/eval_two.py -e "OpenCabinetDrawer-v1" --total-timesteps 10_000_000 -n 1 --algo "sen" \
--log-dir "./log/q_test/noised_sparse" --control-mode "base" --model-ids "['1000']" \
--model-path "./log/q_test/noised_sparse/latest_model"  --reward-mode "sparse"