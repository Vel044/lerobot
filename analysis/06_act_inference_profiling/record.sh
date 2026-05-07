LEROBOT_PROFILING=1 LEROBOT_PROFILING_CSV=/home/vel/lerobot/analysis/06_act_inference_profiling/inference_timing.csv \
    python -m lerobot.record  \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=R12254705 \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=R07254705 \
    --robot.disable_torque_on_disconnect=true \
    --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 360, 'fps': 30}}" \
    --dataset.single_task="Put the bottle into the black basket." \
    --policy.path=/home/vel/so101-bottle/last/pretrained_model \
    --dataset.repo_id=${HF_USER}/eval_so101_bottle --dataset.push_to_hub=false \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=1 \
    --resume=true 

