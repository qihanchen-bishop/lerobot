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

conda run -n sam2 python tools/yolo_sam2/convert_sam2_masks_to_lerobot.py \
  --source-root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277 \
  --seg-task-dir /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277/seg/task1 \
  --output-root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 \
  --overwrite

soarmcube277_mask_task1 普通 LeRobot ACT 训练，只使用真实 RGB left_front
conda run --no-capture-output -n lerobot python -u mycode/train_lerobot_policy.py --policy-type act --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --image-keys observation.images.left_front --state-keys observation.state --output-dir outputs/train/act_soarmcube277_mask_task1_left_front --job-name act_soarmcube277_mask_task1_left_front --batch-size 8 --steps 100000 --device cuda --rebuild-view

soarmcube277_mask_task1 Mask ACT canonical 五 mask 定义
mask 顺序: observation.images.occluder observation.images.object observation.images.region observation.images.left_arm observation.images.right_arm

1A: RGB -> U-Net -> 五 mask -> ACT，action loss 不反传到 U-Net
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 1A --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_1a_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

1B: RGB -> U-Net -> 五 mask -> ACT，seg loss + action loss 联合优化 U-Net mask 链路
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 1B --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_1b_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

2A: ACT 看预测五 mask + U-Net encoder latent，action loss 不反传到 U-Net
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 2A --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_2a_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

2B: ACT 看预测五 mask + U-Net encoder latent，action loss 优化 latent encoder，但不从 mask decoder 链路反传
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 2B --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_2b_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

3: ACT 只看 U-Net encoder latent，不显式输入 mask，decoder 子任务训练仍保留
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 3 --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_3_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

4A: 五 mask + object/region 指标作为 ACT env state
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 4A --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_4a_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

4B: 五 mask + ACT encoder metric token 监督
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 4B --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_4b_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

4C: 五 mask + ACT decoder chunk-autoregressive metric 反馈
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 4C --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_4c_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

5: RGB-only inference，训练时用 canonical 五 mask 语义 latent teacher 蒸馏
conda run --no-capture-output -n lerobot python -u mycode/train_mask_act_policy.py --experiment 5 --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 --repo-id soarmcube277_mask_task1 --rgb-key observation.images.left_front --output-dir outputs/train/mask_act_5_soarmcube277_task1 --batch-size 2 --steps 100000 --device cuda --rebuild-view

conda run --no-capture-output -n lerobot python -u mycode/train_lerobot_policy.py \
  --policy-type act \
  --root /home/romilab/Projects/IsaacLab/source/lerobot/data/soarmcube277_mask_task1 \
  --repo-id soarmcube277_mask_task1 \
  --image-keys observation.images.left_front \
  --state-keys observation.state \
  --output-dir outputs/train/act_soarmcube277_mask_task1_left_front \
  --job-name act_soarmcube277_mask_task1_left_front \
  --batch-size 8 \
  --steps 100000 \
  --device cuda \
  --rebuild-view

  python mycode/train_lerobot_policy.py \
    --policy-type act \
    --root /home/qihan/data/lerobot/data/newdata_3object \
    --repo-id newdata_3object \
    --image-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --chunk-size 60 \
    --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --batch-size 8 \
    --steps 100000 \
    --device cuda \
    --video-backend pyav \
    --rebuild-view
 python mycode/train_lerobot_policy.py --policy-type act --root /home/qihan/data/lerobot/data/newdata_3object --repo-id newdata_3object --image-keys observation.images.front observation.images.side --state-keys observation.state --output-dir outputs/train/act_newdata_3object_front_side --job-name act_newdata_3object_front_side  --chunk-size 60 --n-action-steps 60 --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 --batch-size 8 --steps 100000 --device cuda --video-backend pyav --rebuild-view

python mycode/train_lerobot_policy.py --policy-type diffusion --root /data/qihan/lerobot/data/newdata_3object --repo-id newdata_3object --image-keys observation.images.front observation.images.side --state-keys observation.state --output-dir outputs/train/diffusion_newdata_3object_front_side_obs4_h40_a16 --job-name diffusion_newdata_3object_front_side_obs4_h40_a16 --diffusion-n-obs-steps 4 --diffusion-horizon 40 --diffusion-n-action-steps 16 --batch-size 8 --steps 100000 --device cuda --video-backend pyav --rebuild-view

python mycode/train_mask_act_policy.py \
    --experiment 2B \
    --root /data/qihan/lerobot/data/newdata_3object \
    --repo-id newdata_3object \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys \
      observation.images.front_occluder \
      observation.images.front_object \
      observation.images.front_region \
      observation.images.front_tool \
      observation.images.side_occluder \
      observation.images.side_object \
      observation.images.side_region \
      observation.images.side_tool \
    --output-dir outputs/train/mask_act_2B_newdata_3object_front_side_act_aligned \
    --steps 100000 \
    --batch-size 8 \
    --chunk-size 60 \
    --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --device cuda \
    --num-workers 8 \
    --video-backend pyav \
    --rebuild-view