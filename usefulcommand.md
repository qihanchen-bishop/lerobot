机械臂标定流程
1 夹爪必须打开，主从臂一起配对，wrist的初始位置应该一致，也要旋转
2 gripper映射出错，断电重连
sudo chmod 666 /dev/ttyACM*
lerobot-calibrate     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM3     --teleop.id=my_awesome_leader_arm_r
lerobot-calibrate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=my_awesome_follower_arm
conda run --no-capture-output -n lerobot python -u mycode/gui_record_so101_bimanual.py
conda run --no-capture-output -n lerobot python -u mycode/gui_view_lerobot_dataset.py
conda run --no-capture-output -n lerobot python -u mycode/gui_eval_lerobot_policy.py
conda run --no-capture-output -n lerobot python -u mycode/gui_so101_visual_calibration.py

Mask ACT 实验训练
1A: RGB -> U-Net -> Mask -> ACT，action loss 不反传到 U-Net
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 1A --root simdata/cube1 --repo-id cube1 --output-dir outputs/train/mask_act_1a_object --mask-target-keys observation.images.object --batch-size 2 --steps 100000 --device cuda

1B: RGB -> U-Net -> Mask -> ACT，seg loss + action loss 联合优化 U-Net mask 链路
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 1B --root simdata/cube1 --repo-id cube1 --output-dir outputs/train/mask_act_1b_object --mask-target-keys observation.images.object --batch-size 2 --steps 100000 --device cuda

2A: ACT 看预测 Mask + U-Net encoder latent，action loss 不反传到 U-Net
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 2A --root simdata/cube1 --repo-id cube1 --output-dir outputs/train/mask_act_2a_all --mask-target-keys observation.images.occluder observation.images.object observation.images.region observation.images.left_arm observation.images.right_arm --batch-size 2 --steps 100000 --device cuda

2B: ACT 看预测 Mask + U-Net encoder latent，action loss 优化 latent encoder，但不从 mask decoder 链路反传
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 2B --root simdata/cube1 --repo-id cube1 --output-dir outputs/train/mask_act_2b_all --mask-target-keys observation.images.occluder observation.images.object observation.images.region observation.images.left_arm observation.images.right_arm --batch-size 2 --steps 100000 --device cuda

3: ACT 只看 U-Net encoder latent，不显式输入 Mask，decoder 子任务训练仍保留
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 3 --root simdata/cube1 --repo-id cube1 --output-dir outputs/train/mask_act_3_all --mask-target-keys observation.images.occluder observation.images.object observation.images.region observation.images.left_arm observation.images.right_arm --batch-size 2 --steps 100000 --device cuda

conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 4A --root simdata/cube1 --repo-id cube1 --output-dir outputs/train/mask_act_4a_object --batch-size 8 --steps 100000 --device cuda