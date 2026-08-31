# Simple Stage V5

`STAGE-SIMPLE-V5-*` adds one deterministic four-class stage token to ACT. It has no learned phase
model, temporal history, transition model, rollback, teacher forcing, or auxiliary stage loss.

For every training or inference frame, the frozen front U-Net produces a mutually exclusive class map.
The current stage is computed directly:

| Index | Stage | Current-frame rule |
| ---: | --- | --- |
| 0 | expose | object area is below the visibility threshold |
| 1 | separate | visible object pixels are one-pixel adjacent to occluder pixels |
| 2 | transport | object is visible, separated, and not inside the region |
| 3 | restore | at least 95% of the one-pixel ring around the object is region |

The inside-region rule is evaluated before object-occluder adjacency. This keeps the frame in restore
when the cloth approaches an object that is already inside the target region. The four-way one-hot token
is passed to ACT as `observation.environment_state` and is detached from the frozen U-Net.

The four visual ablations remain:

| Experiment | ACT visual inputs | Stage source |
| --- | --- | --- |
| `STAGE-SIMPLE-V5-F-RGB` | front RGB | current front U-Net frame |
| `STAGE-SIMPLE-V5-F-UNETSEM` | front RGB + front semantic RGB | current front U-Net frame |
| `STAGE-SIMPLE-V5-FS-RGB` | front + side RGB | current front U-Net frame |
| `STAGE-SIMPLE-V5-FS-UNETSEM` | two RGB + two semantic RGB | current front U-Net frame |
