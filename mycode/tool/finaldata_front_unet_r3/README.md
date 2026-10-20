# finaldata_front_unet_r3

`finaldata` 数据集的 front 视角 U-Net，来自第三轮累计伪标签迭代。
这里的 `r3` 表示迭代 round3，不含最初的种子训练；不是时间戳。

## 来源

- 原模型目录：`/data/qihan/sam2/segdata/temp/finaldata/iterunet/iter_r1/round3/model_ef323abe`
- 模型架构：ResNet18 U-Net。
- 输入：RGB，640 x 360；归一化与颜色表见 `model_config.json`。
- 类别 ID：0 background、1 occluder、2 object、3 region、4 tool、5 leftarm、6 rightarm。
- 仅用于 front；没有据此确认 side 视角的适用性。
- 原训练、评估和数据来源记录随包保留；记录中的绝对路径是溯源信息。

## 文件

- `best.pt`：best epoch 29 的 PyTorch checkpoint，供兼容的训练代码加载。
- `latest.pt`：最后训练 checkpoint，不应与 best 权重混用。
- `model.ts`：从最佳模型导出的 TorchScript，推理建议使用此文件。
- `model_config.json`：输入尺寸、归一化、类别顺序与调色板。
- `infer_light_semseg.py`：独立图片/视频推理入口，依赖 PyTorch、OpenCV、NumPy。

模型二进制文件通过 Git LFS 同步。首次使用此仓库时先安装 `git-lfs`，
然后在仓库根目录执行以下命令，下载此模型包的完整权重：

```bash
git lfs install --local
git lfs pull origin --include="mycode/tool/finaldata_front_unet_r3/*"
```

```bash
/home/qihan/miniconda3/envs/sam2/bin/python \
  /home/qihan/data/lerobot/mycode/tool/finaldata_front_unet_r3/infer_light_semseg.py \
  --model-dir /home/qihan/data/lerobot/mycode/tool/finaldata_front_unet_r3 \
  --input /absolute/path/to/front.mp4 \
  --output-dir /absolute/path/to/new_preview \
  --device cpu
```

## 校验值

```text
best.pt  e2348dad988835ed82b3c1c3a8c24d6f272eca13767e4d2c38d9bed3d331061c
model.ts 074d87d18b43d25269aa00f9541599fa14b37f7e9ce9c3fc9ee4b6510bd4e9ea
```

## 全量结果命名

建议分割结果目录命名为 `front_unet_r3_full_v1`，数据集导出目录命名为
`finaldata_front_unet_r3_v1`。`r3` 固定标识模型迭代轮次；重新过滤或重新
导出时递增 `v1`，不把结果版本冒充新的模型代数。

此目录仅为模型导出包，不表示全量数据集已分割或已完成审核。
