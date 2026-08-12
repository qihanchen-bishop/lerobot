# SSACT-2 Experiment Design

## 1. Objective and boundary

SSACT-2 extends a converged SSACT-1 checkpoint with two trainable outputs:

1. an adaptive execution-horizon head that predicts how many actions may be
   executed before observing and replanning again;
2. a bounded residual-action head that corrects the nominal ACT action chunk.

An optional CLF-QP projects the first corrected action toward a phase-specific
semantic objective. The QP is not a learned head. It must run in `shadow` mode
until semantic-model residuals, observation error, action limits, and QP
feasibility have been calibrated on trajectories excluded from training.

SSACT-2 therefore has three deployment modes:

- `residual`: learned horizon and bounded learned residual only;
- `qp-shadow`: log the QP correction but execute the learned residual action;
- `qp-active`: execute the QP projection only after calibration gates pass.

The experiment must not describe `qp-active` as certified when it uses the
current all-300-episode SSACT-1 checkpoint, because that checkpoint has no
independent calibration/test split.

## 2. Nominal policy and trainable heads

Let the frozen SSACT-1 checkpoint produce a normalized nominal action chunk

```text
u_nom[1:H] = pi_ssact1(rgb, soft_semantics, robot_state, phase_history).
```

Freezing SSACT-1 during the first SSACT-2 stage prevents the new losses from
damaging the segmentation, phase, dynamics, and nominal ACT behavior already
learned by SSACT-1.

The residual controller consumes the current 28-dimensional semantic state,
the normalized robot state, predicted phase probabilities, and each nominal
action. A GRU produces a bounded residual for every action step:

```text
delta_u[k] = residual_scale * tanh(residual_head(h[k]))
u_corr[k] = u_nom[k] + delta_u[k].
```

The residual scale is defined in normalized action units and must also be
checked after denormalization against per-joint position and rate limits. The
initial reference value is `0.25` normalized units. The output layer starts at
zero, so SSACT-2 initially reproduces SSACT-1 exactly.

The same GRU emits one hazard logit for each of the first `K_max` steps. The
cumulative event probability determines the learned execution length:

```text
p_stop_by[k] = 1 - product_{j<=k}(1 - sigmoid(hazard[j])).
K_pred = first k where p_stop_by[k] >= execution_quantile.
```

Clamp `K_pred` to `[K_min, K_max]`. The initial setting is `K_min=4`,
`K_max=24`, and `execution_quantile=0.5` at 30 Hz. Thus the policy observes at
least every 0.8 seconds, even though ACT still predicts a 60-step chunk.

## 3. Execution-horizon supervision

The head must learn a maximum open-loop interval, not an arbitrary imitation of
the fixed `n_action_steps` setting. For each training frame, define the first
future replan event as the earliest of:

- the next automatically generated phase boundary;
- the end of the current episode;
- the first future frame whose required semantic labels fail the offline
  quality threshold;
- `K_max`, which is treated as right-censored when no earlier event exists.

This gives an exact event step and an event-observed flag. Train the hazard head
with discrete censored survival NLL. Low-confidence phase trajectories receive
lower loss weight. A phase boundary is a useful re-observation target because
the visual objective and appropriate correction law change there.

At deployment, the learned length remains an upper bound chosen from familiar
data. Online event triggers can always shorten it when segmentation confidence
drops, the phase changes, semantic innovation exceeds its calibrated interval,
or CLF progress is violated.

## 4. Residual-action supervision

The demonstrations do not contain a separately labeled corrective action. A
residual target is therefore defined relative to the frozen nominal checkpoint:

```text
delta_u_target = u_expert - stop_gradient(u_nom).
```

The nominal action must be generated through the inference path, without the
ACT VAE encoder seeing the expert action. Using ACT's training-mode output would
leak the target action into the nominal prediction and make the residual target
invalid.

The residual losses are:

```text
L_res_imitation = masked SmoothL1(u_corr, u_expert)
L_res_size      = mean(||delta_u||^2)
L_res_smooth    = mean(||delta_u[k] - delta_u[k-1]||^2)
L_res_dynamics  = quality-masked semantic prediction loss for u_corr.
```

Recommended initial weights are `1.0`, `0.01`, `0.02`, and `0.05`. Gradients
from `L_res_dynamics` pass through the frozen semantic dynamics model into the
residual head, but do not update the dynamics model. The residual bound and
imitation anchor limit model exploitation.

Success-only demonstrations mostly teach small in-distribution corrections.
They do not identify reliable recovery actions after a grasp miss, object loss,
or severe entanglement. Recovery/correction demonstrations are required before
claiming that the residual head handles those states.

## 5. Phase-specific visual objective and CLF-QP

For each phase, use the interpretable image-space objectives already defined in
`docs/draft/main.tex`: cloth/object exposure, object-cloth contact,
object-goal distance/contact, and restoration. Keep view-local coordinates and
combine objective values with calibrated view/class reliability; do not average
front and side centroids as if they shared a physical coordinate frame.

The one-step semantic dynamics prediction is differentiated with respect to the
first corrected action to obtain a local action gradient of the phase objective.
The QP finds the smallest bounded modification satisfying the linearized robust
decrease constraint:

```text
min 0.5 * ||u - u_corr||_W^2 + p_slack * slack^2
s.t. DeltaV_linearized(u) + beta_model + beta_obs
     <= -rho_phase * V_current + slack
     u_lower <= u <= u_upper.
```

The current `BoxConstrainedCLFProjector` can solve this one-constraint box QP.
It cannot enforce coupled collision constraints. `beta_model` and `beta_obs`
must come from an independent calibration split, not training NLL or softmax
confidence. QP actions must also be checked in denormalized robot joint units.

Enable `qp-active` only when all of the following hold on calibration and shadow
runs:

- requested semantic prediction interval has empirical coverage at least 95%;
- QP feasibility without excessive slack is at least 99% in the declared
  operating domain;
- no projected command violates position, velocity, or per-cycle target-change
  limits after denormalization;
- shadow corrections reduce the observed phase objective more often than the
  nominal/residual action and do not reduce task success;
- camera latency and action timing remain inside the calibrated bounds.

Otherwise the controller executes `u_corr`, logs the proposed QP action, and
forces early replanning when the QP gate fails.

## 6. Training stages

Use trajectory-level splits before any SSACT-2 result is reported:

- train: episodes 0-239;
- calibration: episodes 240-269;
- test: episodes 270-299.

If task/object combinations are uneven, replace these ranges with a stratified
trajectory split and store the explicit episode lists. Do not reuse the current
all-data SSACT-1 checkpoint for final paper numbers.

Training proceeds as follows:

1. retrain or fine-tune SSACT-1 on the train split only;
2. generate nominal inference chunks and adaptive-horizon labels on the train
   split, preferably as an offline cache;
3. initialize SSACT-2 from SSACT-1, freeze SSACT-1, and train the horizon and
   residual heads;
4. optionally unfreeze only the semantic dynamics and the final ACT decoder
   layers for a short low-learning-rate fine-tune after the new heads are
   stable;
5. calibrate semantic residual intervals, stop threshold, QP margins, and
   action bounds on the calibration split;
6. lock every threshold, then evaluate once on the test split and on robot-side
   shadow runs before active QP deployment.

## 7. Required ablations and metrics

Compare the following under the same train/calibration/test split:

- `SSACT-1`: fixed execution length, no residual, no QP;
- `SSACT-2-H`: learned execution length only;
- `SSACT-2-HR`: learned length plus bounded residual;
- `SSACT-2-HR-QP-shadow`: log QP corrections without execution;
- `SSACT-2-HR-QP-active`: only after all activation gates pass.

Report action L1 and task success together with execution-length MAE, event
negative log likelihood, early/late replan rate, average executed steps,
residual norm, action-limit violations, semantic objective decrease rate,
dynamics interval coverage, QP feasibility/slack, and end-to-end latency.
Segmentation Dice and phase accuracy must also be checked to verify that adding
the controller did not regress SSACT-1 perception.

## 8. Acceptance criteria for the first implementation

The first SSACT-2 checkpoint is usable for controlled robot evaluation only if:

- loading the SSACT-1 checkpoint leaves missing keys only in the two new heads;
- zero-initialized residual inference exactly reproduces SSACT-1 actions;
- execution labels never cross episode boundaries and censored samples are
  handled correctly;
- corrected actions improve held-out action L1 without action-limit violations;
- learned replanning outperforms fixed 24-step execution on held-out event
  timing or task success;
- QP shadow logs contain the nominal, residual, projected action, objective,
  margins, feasibility, slack, and the reason whenever projection is rejected.

