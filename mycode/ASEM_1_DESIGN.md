# ASEM-1 Design Record

## Objective

ASEM-1 keeps the architecture, inputs, and supervised segmentation loss of SEM-1. It additionally
allows the ACT action objective to update the semantic segmentation network. Segmentation remains
the primary objective: action supervision may make a semantic map more useful for control, but it
must not erase small classes or turn the soft map into an unconstrained action-feature side channel.

## Losses

The supervised semantic loss is

```text
L_seg = weighted_multiclass_CE + dice_loss_weight * foreground_Dice_loss
```

The ACT loss remains unchanged:

```text
L_ACT = action_L1 + kl_weight * VAE_KL
```

Only `action_L1` has a gradient path through ACT's visual input into the semantic map. The VAE KL
term does not depend on the visual input, so it does not supervise the segmentation network.

## Guarded Task Gradient

Let `theta_seg` be the segmentation-network parameters:

```text
g_seg = grad(theta_seg, seg_loss_weight * L_seg)
g_act = grad(theta_seg, action_loss_weight * L_ACT)
```

If the gradients conflict (`dot(g_act, g_seg) < 0`), ASEM-1 removes the component of the action
gradient that opposes supervised segmentation:

```text
g_act_projected = g_act - dot(g_act, g_seg) / ||g_seg||^2 * g_seg
```

The projected action gradient is then capped, but never amplified, relative to the supervised
segmentation gradient:

```text
scale = min(1, scheduled_ratio * ||g_seg|| / ||g_act_projected||)
g_final = g_seg + scale * g_act_projected
```

ACT parameters outside the segmentation network receive the ordinary full ACT gradient.

## Schedule And Defaults

```text
action_to_seg_grad_ratio   = 0.10
action_to_seg_warmup_steps = 20000
action_to_seg_ramp_steps   = 20000
conflict_projection        = enabled
```

The scheduled ratio is zero during warmup, increases linearly from 0 to 0.10 during the ramp, and
stays at 0.10 afterwards. The 0.10 cap and 20k/20k schedule are conservative project choices, not
hyperparameters copied from a paper. They are intended to keep pixel supervision dominant while
testing whether action supervision adds useful task sensitivity. In the completed SEM-1 baseline,
object/tool Dice was about 0.19/0.42 at step 10k and 0.66/0.71 at step 20k. This is why action
gradients remain disabled until 20k rather than being introduced while the small classes are weak.

## Relation To Prior Work

- The conflict test and projection are based on PCGrad's gradient-surgery principle. ASEM-1 uses a
  one-way variant: action gradients are projected to protect the primary segmentation objective;
  segmentation gradients are never projected to protect the auxiliary action objective.
- Controlling one task by its gradient norm relative to another is inspired by GradNorm. ASEM-1 is
  not the full GradNorm algorithm: it uses a fixed upper bound and does not learn dynamic loss
  weights from relative task-training rates.
- The action reconstruction objective remains ACT's L1 objective. This preserves comparability with
  SEM-1 and avoids changing both the action model and gradient routing in one experiment.
- Foreground Dice complements weighted cross entropy by directly constraining overlap for foreground
  classes, including object and tool.

## Why This Differs From 1B And 2B

The legacy 1B/2B path directly summed action and binary-mask losses and allowed an unconstrained
action gradient into the mask network. In the completed runs, 1B ended with segmentation BCE around
0.0139 versus about 0.0057 for 1A, and visual previews showed foreground leakage. ASEM-1 addresses
that failure mode with four safeguards:

1. mutually exclusive multiclass softmax maps;
2. weighted CE plus foreground Dice anchoring;
3. delayed and norm-capped action gradients;
4. removal of action-gradient components that oppose supervised segmentation.

## Logged Diagnostics

ASEM-1 records:

```text
action_to_seg_target_grad_ratio
seg_supervised_grad_norm
action_to_seg_raw_grad_norm
action_to_seg_raw_grad_ratio
action_to_seg_grad_cosine
action_to_seg_projected_grad_cosine
action_to_seg_grad_conflict
action_to_seg_applied_scale
action_to_seg_applied_grad_norm
action_to_seg_applied_grad_ratio
```

The key acceptance criterion is that segmentation Dice remains comparable to SEM-1 while action
performance improves. Training loss alone is insufficient; checkpoint previews and real-policy
rollouts must also be compared.

## References

1. Yu et al., "Gradient Surgery for Multi-Task Learning," NeurIPS 2020.
   https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
2. Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask
   Networks," ICML 2018. https://proceedings.mlr.press/v80/chen18a.html
3. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," RSS 2023.
   https://arxiv.org/abs/2304.13705
4. Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
   Segmentation," 3DV 2016. https://arxiv.org/abs/1606.04797
