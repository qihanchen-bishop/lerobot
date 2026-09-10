# 从自主语义标注到 QToken-GRU 残差控制：当前方法完整说明

核对日期：2026-09-09。LeRobot 核对提交：`bd466b8`；SAM2 以本机当前源码和落盘记录为准。

本文描述目前实际使用的模块、训练监督和执行时序，不把历史讨论中的候选方案写成已实现功能。SAM2 部分依据 samcodex 所在 `/data/qihan/sam2` 项目的流程文档、脚本及数据集产物核对，没有将其他对话的未落盘设想当作代码事实。

## 1. 一句话概括

先用 AI 生成少量可审核的目标提示，通过 SAM2 视频传播和不确定性诊断反复修复标签，再训练轻量 U-Net，在线为 ACT 提供语义图；进一步用三个几何监督的可学习 token 引导 ACT 编码任务相关关系，最后冻结基础策略，用最新观察和历史特征训练 GRU，对基础动作块进行时间对齐的局部残差修正。

这是一条分阶段构建的流水线，不是 SAM2、YOLO、U-Net、ACT、GRU 同时端到端训练：

```text
离线标注：类别说明 + 视频
          → AI 初始框/点提示 → 渲染审核 → SAM2 双向传播
          → 质量诊断 → 自动重试/少量人工复核
          → 可靠 mask 转框 → 分视角 YOLO → 扩展提示和标注 → 再次诊断

离线模型训练：审核后的语义标签 → 分视角轻量 U-Net
             RGB + 冻结 U-Net 预测语义图 → ACT 动作学习
             front 标签几何 + 标签质量 → 可选 QToken 辅助监督
             冻结 ACT/QToken + 对齐示范动作 → GRU 残差学习

在线使用：新 RGB、follower 关节 → U-Net/ACT 慢计划
          新 RGB 特征 + 当前计划 + 历史缓存 → GRU 修正
          原始基础动作 + 对应目标时刻的有界残差
```

在线不运行 SAM2 标注迭代，不读取未来视频帧，也不需要数据集标签。

## 2. 数据、类别与表示约定

当前策略实验使用 `data/bettersetup_v5`：90 条 episode、53,723 帧、30Hz，front/side 两个 RGB 视角。RGB、state 和原动作记录沿用原 bettersetup；v4/v5 主要完善分割。

| ID | 类别 | 含义 | 展示色 RGB | front | side |
| --- | --- | --- | --- | --- | --- |
| 0 | background | 未单独标注的可见表面 | 0,0,0 | 有 | 有 |
| 1 | occluder | 可形变遮挡布 | 64,160,255 | 有 | 有 |
| 2 | object | 被操作的方块、纸团、螺丝等 | 255,105,97 | 有 | 有 |
| 3 | region | 目标区域 | 119,221,119 | 有 | 有 |
| 4 | tool | 连接夹爪的长条工具，不含夹爪 | 255,209,102 | 有 | 有 |
| 5 | leftarm | 左机械臂可见结构及相关线材 | 234,146,199 | 有 | 无 |
| 6 | rightarm | 右机械臂可见结构及相关线材 | 173,214,101 | 有 | 无 |

因此 front 是六个前景类别、含背景七通道；side 是四个前景类别、含背景五通道。共有类别 ID 和颜色一致，side 未标注机械臂的像素归入其背景，不能把两个视角的“背景”理解为完全相同的物理集合。

原始标签是互斥的可见表面分割，不是遮挡后的完整物体形状。tool 遮住 object 的像素应为 tool；接触不意味着合并身份。多类概率可以在同一像素非零，但其和为1，这表示分类不确定性，不表示多个不透明物体同时占有同一可见像素。

原始监督保存为每视角、每前景类别的独立二值 mask 视频；训练时按类别表重建 class-ID 图。彩色视频用于审核，不用压缩后的展示颜色反推监督类别。

## 3. 自主 Prompt 与不确定性驱动的 SAM2 标注

### 3.1 首批提示如何获得

1. 阅读 `labels.txt`，明确每类包括/排除什么，以及各视角类别集合。自然物体颜色与语义展示颜色不是一回事。
2. 选少量有代表性的视频和关键帧，覆盖刚暴露、接触、遮挡、重新出现、落入目标区等状态。不能只选分割最容易的静止画面。
3. AI/Codex 根据图像和任务定义生成离散 JSON 提示，记录视频、视角、帧号、对象身份及框；必要时增加正点、相邻类别负点和可见区间 `visible_ranges`。实际字段遵循项目 prompt schema，不依靠自然语言临时解释。
4. 将 JSON 渲染为框/点叠加图，结合前后帧检查，再由人工确认首批语义身份与范围。自主生成不等于无需审核。
5. 以审核通过的提示运行 SAM2，获得少量视频的完整时序标签。

可见区间只表达“此时应能看到某部分目标”，不能为了提高完整性分数而事后随意删掉漏标区间。真实完全遮挡应通过上下文和审核记录解释。

### 3.2 SAM2 传播与局部修复

当前自动迭代规范要求保存独立 forward 和 reverse 传播，随后形成互斥语义 mask，并保留两个方向作为一致性诊断证据。不同方向可共享提示，因此它们不是统计上独立的真值，也不能排除一致地分错类别。

已通过审核的区域应保留，问题片段加上下文后局部重跑。每轮保存提示、mask、完成标记、评分、重试队列和人工 override，避免用新一轮覆盖历史证据。

目录结构：`origin/<dataset>` 只读，`temp/<dataset>/...` 保存轮次，验证后才发布到 `final/<dataset>`。

### 3.3 不确定性如何计算

以某一视角、类别的二值 mask 为 `M_t`，面积为 `A_t`，归一化质心为 `c_t=(x/W,y/H)`。所有诊断逐帧保存。

| 指标 | 当前代码计算方式 | 主要用途 |
| --- | --- | --- |
| 时间一致性 `T_t` | 与前、后邻帧 IoU 的较小值 | 定位轮廓突变、身份漂移 |
| 面积变化 `D^A_t` | 相邻对 `abs(log((A_t+1)/(A_(t-1)+1)))`，前后取最大 | 扩大、缩小、突然消失 |
| 质心跳变 `D^C_t` | 相邻归一化质心欧氏距离，前后取最大 | 不合理位置跳转 |
| 连通数 `N_t` | 8邻域连通域数，忽略面积小于图像面积0.0001的碎片 | 过度碎裂 |
| 双向一致性 `B_t` | 同帧同类 forward/reverse mask IoU | 提示和传播方向敏感性 |
| 预期可见但缺失 | `visible_expected AND A_t==0` | 防止漏标被当成合理空白 |

序列端点没有的一侧 IoU 设为1；空对空 IoU也为1。质心缺失时，相邻质心距离不计算。因此“始终漏掉物体”的空mask可以在若干项上得高分，必须同时检查预期可见性。

设各阈值为 `tau_T,tau_A,tau_C,N_max,tau_B`，综合分数为五项平均：

```text
s_T = clip(T_t / tau_T, 0, 1)
s_A = clip(1 - D^A_t / tau_A, 0, 1)
s_C = clip(1 - D^C_t / tau_C, 0, 1)
s_N = clip(N_max / max(N_t, 1), 0, 1)
s_B = clip(B_t / tau_B, 0, 1)     # 仅在双向结果可用时
quality_score = (s_T + s_A + s_C + s_N + s_B) / 5
```

双向不可用时，当前实现给 `s_B=1`，同时记录 `bidirectional_available=false`，不是把未知一致性测成1。跨版本比较时必须同时报告双向覆盖率；只看总分会偏乐观。

基础诊断阈值：时间IoU 0.6、面积log变化0.45、质心变化0.08、最大连通数3、双向IoU 0.6、总分0.6。按类覆盖：object最多2个连通域；occluder时间IoU 0.5、最多8个；tool时间IoU 0.4、质心0.12、最多6个、双向IoU 0.5。

`uncertain` 是低总分或任一关键异常的逻辑或，包括预期可见时漏标。即使总分较高，严重异常仍单独保留，不被其他高分平均掉。

**这些是人工设定的质量诊断代理，不是概率校准后的不确定性，也不是“该标签有95%概率正确”。** 不应称为 SAM2 网络原生置信度。运动、接触和真正遮挡都可能触发低分。

### 3.4 评分如何驱动重试和扩充

`build_uncertainty_gate.py` 根据各项证据而非总分单独分流：

- `accept`：不存在严重结构异常，分数和可用的双向证据满足要求；也接受另一条“无uncertain且无结构异常、分数>=0.55”的稳定分支。
- `retry`：自动重生成提示和局部mask，不立即要求人工逐帧修复。
- `review`：同一问题耗尽默认两次自动重试预算，交给人工。人工可明确给出 accept/retry/reject/review，必须写原因。

门控默认接受分数0.75、低分阈值0.55、接受双向IoU 0.8。这些是门控参数，不同于上节生成异常标志的基础阈值。

转场保护：如果只是时间IoU/面积变化，且没有漏标、低双向、碎裂、质心异常，同时双向可用且IoU>=0.9、分数>=0.55，可以自动接受。目的是不要把“刚暴露”的重要训练帧全删掉。

自动重试把相邻异常合并为片段，增加首/中/尾或暴露边界锚点，生成更紧框及正负点，重新传播并重新评分。必要时降低的是**候选提示检测门槛**，不是降低最终质量门槛。

### 3.5 YOLO 的角色与迭代闭环

YOLO在这里是提示框生成器，不是最终逐像素标注器，也不是策略网络：

```text
审核/门控通过的 SAM2 mask → 包围框 → 分视角YOLO训练
新视频 → YOLO提示框 → SAM2时序标签 → 评分和修复
新通过的标签 → 扩展YOLO训练集 → 下一轮
```

如果某帧中真实存在或预期可见的任一类别未通过，不能只留下其他类别框训练YOLO，否则漏掉的正类被当成背景。当前门控支持整帧过滤，并保留按类别原因。

流程文档建议按6→12→30→全量episode逐步扩充，兼顾困难、新外观和稳定样本，使用episode隔离验证。此数字是推荐扩展日程，不代表bettersetup_v5严格执行了完全相同的每一轮。

实际v5产物中可见接触场景、螺丝、episode065等多轮专门修复目录；front继承v4并修复过右夹爪身份，side继续迭代。标准流程的重试/发布条件与某次历史数据的真实达成状态必须分别报告。

### 3.6 发布条件与当前证据边界

发布应核对：视频尺寸/FPS/帧数、episode元数据、零像素重叠、质量数组与全局帧对齐、LeRobot标准加载、问题片段处置记录；不能仅凭高均分发布。

当前v5质量摘要仍列出21条完整性异常记录。它们需要结合对应版本的可见区间和人工例外解释，不能写成“全部漏标已自动消除”。继承的 `BUILD_INFO.md` 标题仍是front-only v4，不能单独作为v5全视角验收证明。

示例诊断统计（当前 `segmentation_quality/summary.json`）：

| 流 | 平均quality_score | uncertain比例 |
| --- | ---: | ---: |
| front object | 0.9786 | 5.45% |
| front tool | 0.9452 | 18.25% |
| side object | 0.9779 | 6.60% |
| side tool | 0.9473 | 16.38% |

高均分与较多异常帧可以同时存在。这里的uncertain比例既不是人工待审比例，也不是真实错误率。

## 4. 从离线标签训练轻量 U-Net

SAM2用于较昂贵的离线生成；策略运行时用单帧轻量分割器替代，避免在线视频标注流程的成本。

当前模型包：

| 项目 | front | side |
| --- | --- | --- |
| 路径 | `models/unet_front_v4_r1` | `models/unet_side` |
| 网络 | depthwise-separable TinyUNet，width48 | 同类结构，width32 |
| 输入 | RGB，480×270，全视野resize | 同左 |
| 归一化 | uint8/255，再ImageNet均值/标准差 | 同左 |
| 输出 | 7通道logits | 5通道logits |
| 标签来源 | 审核后的v4 front | v5 side修订标签 |

离线训练器支持按像素类别频率反平方根归一化的加权CE，以及可配置前景soft Dice。CE学习像素身份；Dice补充前景区域重叠监督。它们的权重应以模型checkpoint的训练参数为准，不能把旧SEM实验固定的tool/object权重直接套到这个独立训练器。

当前检查到的训练循环并没有直接读取逐帧SAM2质量分数乘CE的通路。其可靠性主要来自训练前的标签迭代、审核与样本构建。**质量门控、U-Net分割损失、QToken质量加权是三个不同环节，不能声称它们自动全部启用。**

每视角manifest记录17,937个采样帧，15,118训练、2,819验证，按episode隔离。模型包报告front mIoU=0.96488，side约0.9693。这是对审核SAM2标签的一致性，不是对独立全人工真值的准确率；小螺丝、接触、近完全遮挡仍是困难情况。

## 5. UNET-SEM：将语义作为额外视觉输入

### 5.1 概率图如何进入 ACT

对每个视角 `v`，冻结分割网络产生 `P_v=softmax(logits_v/T)`，当前温度为1。以类别展示色归一化得到 `C_k∈[0,1]^3`，生成soft语义RGB图：

`S_v(u)=sum_k P_v(k,u) C_k`。

高置信区域接近类别纯色；边界不确定时出现颜色混合。不是先argmax再染色，也不是把7/5通道概率直接塞给ImageNet ResNet。

颜色映射是固定压缩：多类别概率向量被压成3通道，不能保证可逆，存在信息丢失。这是为了复用RGB视觉编码器的工程选择，不意味着概率RGB必然优于独立semantic adapter。

ACT对原RGB与soft语义RGB分别调用**同一个可训练ResNet18和投影层**。共享的是权重，不是提前把图像相加；各路特征保留为独立视觉token并拼入encoder。单视角是两路图像，双视角是四路图像。实际FS顺序为front_semantic、side_semantic、front、side。

### 5.2 训练和推理

- 训练：预测语义图和原RGB、follower关节状态输入ACT，监督原数据集action，学习60步动作块；ACT训练含动作L1和VAE的KL项。
- 分割网络：冻结，不由动作loss更新；不需要先在策略训练内把分割学好再切换标签输入。
- 推理：使用相同冻结分割器从新RGB得到语义；无需离线标签，VAE不读取专家未来动作，使用其推理latent约定。
- 已训练UNET-SEM-V5-FS与QTOKEN配置均关闭camera/modality embedding，保留ACT本来的位置编码。共享视觉编码器也不等于显式几何跨视角匹配。

这里不是SEM-1-v2的概率adapter残差融合，不是ViewFus的单应恢复，不是ActionSEM的动作监督微调分割。旧配置文件中可能保留这些分支的默认参数，只有当前experiment实际启用的代码路径才构成方法。

### 5.3 能提供什么、不能提供什么

语义图显式突出目标、工具、区域和遮挡物，可能减少纹理干扰；原RGB保留分割漏掉的机械臂、边界和外观信息。是否改善策略必须通过相同采集、训练和实机条件验证。

双视角在ACT内联合编码，但当前没有标定、三维位置编码或显式“用side补全front遮挡”的模块。看不见的目标仍可能在预测中消失，语义输入不能自动变成完整场景状态。

## 6. QToken：用几何监督组织任务相关特征

### 6.1 三个token是什么

在UNET-SEM基础上，增加三个可学习的512维初始向量 `e_1,e_2,e_3`，跨样本共享。它们与latent、关节state、视觉token一起进入ACT Transformer encoder：

```text
[latent, state, e1, e2, e3, visual tokens] + 对应位置编码
                         ↓ encoder自注意力
                       [h1,h2,h3]
                         ├→ 三个几何预测头，输出维度1、1、2
                         └→ 作为encoder上下文参与动作decoder预测
```

初始参数表共3×512=1536个数，但encoder后的 `h_i` 随当前图像与状态变化。它们是encoder中的固定语义角色token，不是DETR那种独立decoder中的可变对象查询，不进行集合匹配。

### 6.2 三个监督目标只从front标签计算

1. **object可见面积占比**：`g1=sum(M_object)/(H*W)`。直接面积比例，不是旧4A中的平方根面积。
2. **object到region距离**：两个mask的像素质心欧氏距离，除以图像对角线长度。它不是边界距离，也不能单独证明object完全落入region。
3. **tool左端点指向object的二维向量**：在宽高分别归一化的坐标中，对tool mask做加权协方差/PCA，得到长轴；以mask像素的投影极值确定两端点，选图像x较小者为左端点；输出归一化object质心减该端点，得到有正负的dx、dy。

图像坐标中x向右、y向下。第三项不是tool质心距离，也不是三维高度/接触力。图像左端不一定始终是物理有效接触端；工具近竖直、短小或严重遮挡时，PCA方向和端点可能不稳定。

监督来自训练时的front数据集mask，不是让模型只拟合自己当前U-Net输出计算出的指标。单双视角版本都如此；side视觉可以参与预测，但不生成另一组side几何监督。

### 6.3 不可见和低质量样本如何处理

- object不可见时，面积目标为0，仍监督面积头。
- object或region缺失时，距离目标不参与几何loss。
- object或tool缺失时，二维向量不参与几何loss。
- 无效关系的零占位不是“距离为零”；通过validity mask排除。

当前QToken配置使用软质量权重：`w(q)=min(q/0.95,1)^1`，q截断到[0,1]。面积权重使用object质量；关系权重使用两个类别权重的较小值，再乘validity。

每组使用SmoothL1，beta=0.01；先按组对有效质量权重归一化，再对三个组取平均，使二维向量不会仅因维度为2就占双倍组权重：

`L_geo=(1/3) sum_g [sum(w_g * SmoothL1(pred_g,target_g))/max(sum(w_g),1)]`。

总损失为ACT动作L1、其配置的KL项，再加 `lambda_geo*L_geo`；当前 `lambda_geo=1`。权重归一化意味着不是简单把全批loss乘一个固定质量比例，均匀缩小所有权重时可能被分母抵消。

这里的软加权与离线YOLO样本门控不矛盾：门控防止坏伪标签扩散，软权重控制策略几何监督可信度。普通UNET-SEM-V5-FS的配置没有mask_quality_dir，而对应QToken有；不能称两者都使用了同一质量监督。

### 6.4 为什么可能起作用

几何loss促使三个encoder隐藏特征保留与目标暴露、接近目标区、工具相对关系有关的信息；动作decoder可读取这些特征，动作loss也会优化可训练ACT及query参数。

但这不是显式监督注意力热图：没有要求某个注意力头一定落在object像素。几何回归表现好，也不能证明动作解码实际依赖该token。需要置零/替换token的动作消融和实机对照建立证据。

推理时三个数值输出不是外部控制信号，不直接切换阶段或判断完成。几何关系缺失时，encoder仍产生隐藏特征，但不能把未监督情况下的标量输出当作可靠测量。当前没有额外的几何有效性预测头。

## 7. QToken-GRU：在冻结计划上做历史条件残差

### 7.1 A/B/C各自是什么

| 阶段 | 冻结慢策略 | 快分支新增特征 |
| --- | --- | --- |
| A | 普通双RGB ACT | 最新双RGB特征 |
| B | 双RGB+双U-Net语义ACT | 最新双RGB + 最新双视角分割描述 |
| C，即本文QToken-GRU | 双视角QToken ACT | 最新双RGB + 当前已生效慢计划的三个query隐藏特征 |

三者共用残差训练器。A另外训练MLP对照；B/C当前没有单独MLP。单视角QToken存在，但已完成的这组残差A/B/C都是双视角。

C不是在QToken ACT内部插入一个端到端GRU：先训练基础QToken策略，再冻结它，只训练额外特征投影、GRU及残差输出头。没有动作梯度回传U-Net或重新训练query。

### 7.2 最新视觉、语义和历史如何编码

每次快更新只读取每个视角最新一帧。复用冻结基础ResNet权重，在完整视野resize到180×320后提取特征，各视角空间池化为2×3×512，两路合计6144维。不是每次重新编码8组旧图。

B每次额外运行两路冻结U-Net，概率图2×3池化，并提取每前景类面积、概率加权质心、置信度、可见性，以及object-region/object-tool的质心dx、dy、距离和validity。前视增加80维、侧视58维，共138维，合计6282维。

B的默认可见规则包含argmax区域占比>=0.0001、区域平均概率>=0.6及概率数值/归一化有效。几何无效时附validity，不将零距离当测量；合法soft map仍可保留。这里的tool关系是**质心**关系，区别于QToken的PCA左端点。

C在原metric head输入处通过forward pre-hook取三个512维隐藏状态，加三个数值有效标志，合计1539维；与最新RGB拼接总维数7683。标志只说明token存在且有限，不表示物体可见。

Query在慢ACT产生计划时生成，绑定plan_id；新计划尚未生效时不提前使用新query。快分支每次更新RGB，但query是慢语义上下文，不是5Hz新分割结果。

### 7.3 每个历史记录的控制信息

除视觉特征外，记录当前follower关节、因果估计速度、上一条指令、当前计划目标区间的动作及相对state差、有效性、计划年龄、时间间隔、计划切换标志和目标延迟。

v2的plan_steps=9，10关节时context为223维：30维state/速度/上一动作，90维基础动作，90维相对差，9维有效性，4维时间/切换信息。context以训练子集action标准差缩放，不对残差套用绝对动作均值。

外部缓存最近8条“特征+context”，每次从零隐藏状态运行单层hidden128 GRU，取最新有效时刻输出。历史图像特征不重算；历史计划条件保留其当时值。

这不是跨调用永不清空的循环记忆，也没有Transformer KV cache。v2为5Hz，因此8条记录首尾相距1.4秒。正常0.2秒更新保留历史，间隔超过0.3秒清空；新episode/reset也清空。

### 7.4 残差的定义与训练

设目标槽位的原计划为 `a_base`，示范为 `a_demo`，残差目标隐含为 `a_demo-a_base`，输出：

`delta_a=b*tanh(Linear(h_t))`，`a_corrected=a_base+delta_a`。

输出头零初始化。逐关节上限 `b_j=clip(P95_train(abs(a_demo-a_base)),0.05,5)`，单位是数据集校准后的原生命令单位，不是度数。上限来自训练集，不是经机器人认证的安全值。

损失是以训练action标准差缩放的修正动作SmoothL1(beta0.1)，加 `0.001*(delta_a/std)^2`，只统计有效目标。AdamW lr1e-4、weight decay1e-4、batch128、梯度裁剪1，最多50epochs、patience8，以验证归一化MAE保存best。

这里的监督仍是原数据集leader action，输入state是follower。不是此前Follower-Delta、Anchor-Delta，也不是用state未来差监督。

训练中上一条指令取示范action，shadow中取实际返回指令；专家画面也不会因模型修正而改变。因此存在闭环分布偏移，方法属于监督残差模仿学习，不是离线强化学习。

### 7.5 v2的真实时间配置

```text
fps=30
ACT chunk=60，slow_stride=30，slow_delay=3
fast_stride=6，fast_delay=3，correction_steps=9
history=8，history_gap=0.3秒
返回结果最大观察年龄=0.15秒
已接受残差最大使用年龄=0.4秒
```

第6帧观察预测第9–17帧残差；第12帧预测第15–23帧；第18帧预测第21–29帧；第24帧预测第27–35帧。邻轮重叠3帧。fast_delay是目标时间偏移，不是机械臂停下来等100ms。

新结果覆盖同一计划、同一未来槽位的旧结果，不叠加多个残差；每个槽位保留自己的观察时间，不能用新结果刷新旧残差寿命。过期前缀丢弃，剩余槽位不挪位。计划切换清空旧残差。

慢策略每30帧观察一次，代码中的计划ready时间包含模拟/实测推理延迟，因此生效切换不必恰好发生在旧计划第30槽。晚到前可继续有效旧计划。当前实现也没有“每块第24步后硬停止更新”的额外规则，而是全局按6帧周期调度，再按plan_id和有效性管理结果。

当前执行器只返回动作数组，没有真实机器人I/O。shadow为顺序专家视频回放，不是并发线程真实控制，也不是会生成反事实画面的物理仿真。

## 8. 离线缓存、验收与已经取得的结果

冻结特征在训练前预计算，每episode独立shard；保留模型SHA、配置、数据指纹和split，参数变化使用新缓存。归一化处理器和类别顺序必须与基础checkpoint配套，不能只复制残差best.pt就任意搭配策略。

90个episode按任务分层划分72训练/18验证，seed1000。v2残差训练记录数7,169/1,686，验证有效预测槽位15,064；相邻预测有重叠，所以这些槽位不是15,064个统计独立样本。基础ACT和分割器可能见过这些episode，必须称“残差头未见episode”，而不是全系统泛化验证。

截至核对日期，SF-v2三组正式训练完成、队列退出0，每组audit记录59项测试通过；checkpoint重载、在线/离线窗口一致性和多时刻缓存校验通过。

| v2实验 | 基础归一化MAE | 修正后MAE | 正式epochs | 最佳epoch，从1计 |
| --- | ---: | ---: | ---: | ---: |
| A：ACT+GRU | 0.060210 | 0.049784 | 17 | 9 |
| B：UNET+GRU | 0.069496 | 0.054454 | 23 | 15 |
| C：QToken+GRU | 0.070484 | 0.054696 | 31 | 23 |

各关节及各plan-age组误差均下降，三组残差饱和率均0。A的MLP为0.049816，与GRU几乎持平，尚未显示历史窗口的额外收益。

| v2短shadow | 快分支p95 | 慢分支p95 | 有效更新 | 过期前缀 |
| --- | ---: | ---: | ---: | ---: |
| A | 42.1ms | 36.6ms | 19/19 | 0 |
| B | 62.1ms | 82.5ms | 19/19 | 0 |
| C | 42.8ms | 90.5ms | 19/19 | 0 |

每组约4秒回放，117条返回动作中93条含非零残差。它说明这次时间配置在该回放中有效，不保证持续实时性，也不能解释为实机成功率。

**延迟归因需要修正：** B额外执行双U-Net和语义描述，但不能把整条62ms全部归给U-Net。模型包历史benchmark记录front平均1.223ms、side平均0.876ms，均为独立模型特定条件；与共享GPU、预处理/ResNet/几何/同步/GRU的整链路不是同一测量。尚需相同负载下逐模块profile，不能宣称小GRU链路一定比完整ACT快。

v1是每2帧更新、delay1、输出2步、历史约0.47秒；v2的目标、采样和历史均改变。两版MAE不能当作严格相同任务直接排名。

## 9. 三种“质量/不确定性”不能混用

| 层次 | 来源 | 使用阶段 | 当前作用 |
| --- | --- | --- | --- |
| 离线标签质量 | SAM2时序/双向/几何诊断 | 标注、YOLO样本构建、QToken训练 | 重试/复核及监督权重 |
| 在线语义置信度 | U-Net概率及mask有效性 | B残差推理 | 几何validity和输入描述 |
| Query数值有效性 | token是否存在、是否NaN/Inf | C残差推理 | 缺失/坏值掩码 |

第一类用了前后帧和反向传播，在线不能直接获得；第二类未校准，不等于标签质量；第三类不表达语义可信度。当前没有统一学习式不确定性估计器，也没有置信度驱动的控制收敛保证。

## 10. 成立条件、局限与下一步证据

成立条件：类别定义清楚、关键物体可辨、相机配置和输入处理一致；示范覆盖接触/暴露/恢复关键状态；时间戳和关节顺序正确；标签错误能通过独立审核发现；运行延迟落在预测时间预算内。

主要局限：

1. 自动标注会系统性地分错同色物体，时间一致和双向一致都可能很高；不能仅靠分数解决身份错误。
2. 固定图像平面的几何不是三维状态，PCA端点不等于真实接触点；完全遮挡时关系不确定。
3. 共享ResNet不保证语义有帮助，也没有显式跨视角配准或遮挡补全。
4. QToken几何头不是阶段控制器，面积/距离不是任务完成判据；未实现CLF、MPC或严格闭环保证。
5. C的query更新慢，GRU虽有最新RGB，不能把它说成最新语义状态追踪器。
6. 重叠残差覆盖有边界跳变风险，幅度限幅不等于速度/加速度安全；部署需额外约束并实测。
7. 离线动作误差下降不能替代实机成功率、恢复能力和完成时间；当前A的MLP/GRU结果也不支持“历史已被证明关键”。

建议分别验证：自动标注的人工用时与独立人工抽样精度；RGB与RGB+语义的匹配实机对照；QToken辅助监督和token干预；GRU/MLP与残差关闭对照；最新语义与慢query更新频率；端到端延迟p95/p99及实际按时施加比例。以上是后续验证方向，不计作已取得结果。

## 11. 源码与产物索引

以下均为本次核对的本机来源，不把流程规范当作每个历史版本都已达成的实验报告。

- SAM2总流程：[自动迭代标注流程](/data/qihan/sam2/segdata/SAM2_不确定性驱动自动迭代标注流程.md)。
- 指标精确定义：[evaluate_segmentation_quality.py](/data/qihan/sam2/tools/yolo_sam2/evaluate_segmentation_quality.py)。
- 门控与重试：[build_uncertainty_gate.py](/home/qihan/.codex/skills/sam2-uncertainty-auto-iterator/scripts/build_uncertainty_gate.py)。
- 整帧过滤：[filter_yolo_by_gate.py](/home/qihan/.codex/skills/sam2-uncertainty-auto-iterator/scripts/filter_yolo_by_gate.py)。
- U-Net训练：[train_light_semseg.py](/data/qihan/sam2/tools/train_light_semseg.py)。
- 模型包：[front](/home/qihan/data/lerobot/data/bettersetup_v5/models/unet_front_v4_r1/README.md)、[side](/home/qihan/data/lerobot/data/bettersetup_v5/models/unet_side/README.md)。
- 语义输入与几何目标：[train_mask_act_policy.py](/home/qihan/data/lerobot/mycode/train_mask_act_policy.py)，重点看 `semantic_rgb_maps`、`front_query_metric_targets`、质量加权和metric loss路径。
- Query插入与读出：[modeling_act.py](/home/qihan/data/lerobot/src/lerobot/policies/act/modeling_act.py)，查 `encoder_metric_token_embed` 和 `metric_heads`。
- 实际配置：[UNET-SEM](/home/qihan/data/lerobot/outputs/train/semantic/UNET-SEM-v5-front-side-bettersetup-v5/mask_act_run_config.json)、[QToken](/home/qihan/data/lerobot/outputs/train/semantic/UNET-SEM-v5-front-side-QToken-bettersetup-v5/mask_act_run_config.json)。
- 残差核心：[slow_fast_residual.py](/home/qihan/data/lerobot/mycode/slow_fast_residual.py)。
- 语义/query适配：[slow_fast_semantic.py](/home/qihan/data/lerobot/mycode/slow_fast_semantic.py)。
- 缓存、训练和shadow：[train_slow_fast_residual.py](/home/qihan/data/lerobot/mycode/train_slow_fast_residual.py)。
- v2配置与命令：[v2执行说明](../draftv2/slow_fast_residual_v2_s6_d3_h9.md)、[串行脚本](/home/qihan/data/lerobot/run_slow_fast_residual_v2_queue.sh)。
- 最新结果位于 `outputs/train/residual/SF-v2-S6-D3-H9-{A,B,C}-FS/`，核对 `audit.json`、`gru8/result.json`、`gru8/shadow_result.json`、`gru8/best.pt`。

## 12. 可用于草稿的简短方法表述

本方法采用质量驱动的分阶段语义策略学习框架。首先，根据类别定义自主生成稀疏视觉提示，经少量人工审核后驱动SAM2视频传播，并联合时间一致性、面积与质心变化、连通结构、双向传播一致性及预期可见性定位不可靠标注。通过自动局部重试、质量过滤和分视角YOLO提示模型迭代扩展标签，再训练轻量分视角U-Net，提供在线soft语义图。

策略学习阶段，将语义概率映射为固定调色板的soft RGB，与原始RGB共用视觉编码器并作为独立视觉token输入ACT。进一步引入三个几何监督的encoder query token，分别表示目标可见面积、目标到区域的质心距离、工具线左端点到目标的二维位移。几何监督由front标签生成，采用可见性掩码和连续质量权重，动作decoder通过共享encoder上下文利用这些任务相关特征。

在残差控制阶段，冻结训练好的基础策略，缓存最新视觉与控制特征历史，用GRU预测时间对齐的有界动作残差。QToken-GRU还读取当前生效慢计划的query隐藏状态作为语义上下文。当前版本每6个控制帧更新一次，从观察后第3帧开始预测9帧残差，并通过计划身份、目标时间和过期检查管理重叠修正。该方法已获得离线误差改善及shadow验证，但语义和历史模块的因果贡献、真实闭环收益及部署实时性仍需独立实机实验验证。
