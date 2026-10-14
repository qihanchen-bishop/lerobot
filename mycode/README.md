# mycode Scripts

## SO101 Bimanual Data Recorder

Script: `mycode/gui_record_so101_bimanual.py`

This script provides a Tkinter GUI for collecting LeRobot-format bimanual SO101 teleoperation datasets. It connects two follower arms, two leader arms, and optional RGB/RealSense cameras, then records synchronized robot observations, actions, videos, and episode metadata.

### Main Use

Run it when collecting dual-arm SO101 demonstrations:

```bash
conda run -n lerobot python mycode/gui_record_so101_bimanual.py
```

The GUI is used to configure arm ports, calibration path, camera inputs, dataset location, task name, episode length, FPS, preview options, and save/export controls.

### Initial Object Position

Before each recording, select the object's initial workspace grid cell in `采集位置`.
The default grid has 3 rows and 4 columns, matching the evaluation convention:
`(0,0)` is the top-left cell, x increases to the right, and y increases downward.
Rows and columns can be changed; changing the grid clears the selection.
These labels describe the manually chosen workspace cell, not measured pixel or 3-D coordinates.

The selection is captured when `Record` is clicked. Changes during recording apply
to the next episode. Successfully saved episodes receive one record in
`<dataset_root>/meta/episode_positions.jsonl`, containing `episode_index`, task,
grid shape, `grid_cell`, x/y indices, axis conventions, and frame count.
Discarded recordings do not receive a record. The shared single-arm recorder can
continue without a position label. Existing unlabeled episodes display `未标注`.
The episode browser includes a position column, and `位置统计` shows counts grouped
by task, grid shape, and position, with an `导出 CSV` button. Recorder cleanup and
deletion reindex the labels with the episodes and retain the original backup.

### Camera Resolution

`Low res cameras` defaults to enabled (640×360). Disable it for 1280×720.
Both modes request 30 FPS; this is a target, not a measured capture throughput.
The live preview reports observed update rates. Higher resolution increases image
processing and storage work, so check actual preview rates on the collection setup.

Both camera previews fit the available window area while preserving aspect ratio,
including enlarging low-resolution frames to use the available display space.
The default layout is `左右` (side by side); `预览布局` also offers `上下` (stacked).
Each view has its own Zoom controls. `Fit` restores that view's complete frame;
`+` zooms in, with scrollbars for inspecting the enlarged image. Zoom also works
when the source image already fits at its native size. Preview scaling does not
change the recorded resolution, which still defaults to 640×360 at a target 30 FPS.
Use `收起设置` to hide the device/camera configuration and leave more space for
the previews; recording buttons and the initial-position selector remain visible.
Connecting automatically collapses these settings; `展开设置` opens them again.
Connecting also hides the episode list to give both previews more room. Toggle
`示范列表` to show it; stopping the recorder restores the list for dataset browsing.

### Arm Device Matching

The current version uses stable Linux serial paths by default instead of volatile `/dev/ttyACM*` names:

```text
/dev/serial/by-id/usb-1a86_USB_Single_Serial_<board_serial>-if00
```

Current default arm-to-board mapping in the script:

| Arm | Default port | Board serial |
| --- | --- | --- |
| Left follower | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E122511-if00` | `5B3E122511` |
| Right follower | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E119029-if00` | `5B3E119029` |
| Left leader | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E118729-if00` | `5B3E118729` |
| Right leader | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E121504-if00` | `5B3E121504` |

The recorder extracts the board serial number from the selected port and uses it as the calibration id when `Calibration match` is set to `Board serial`. This makes the arm mapping independent of the order in which USB devices appear as `/dev/ttyACM0`, `/dev/ttyACM1`, etc.

If a board-serial calibration file does not exist but the legacy arm-id calibration file exists, the script copies the legacy calibration to the board-serial filename automatically. If no board serial can be resolved, it falls back to the legacy arm id.

### Devices

The recorder builds:

- left SO101 follower arm
- right SO101 follower arm
- left SO101 leader teleoperator
- right SO101 leader teleoperator
- optional front/side OpenCV cameras
- optional RealSense RGB/depth camera

It combines the two followers into a local bimanual follower robot and combines the two leaders into a local bimanual teleoperator.

### Dataset Output

The script writes LeRobot dataset data under the selected dataset root. Recorded data includes:

- bimanual motor observations
- bimanual actions
- configured camera video streams
- episode/task metadata
- dataset statistics and video metadata used by LeRobot tooling

### Related Notes

Use `Board serial` calibration matching for normal dual-arm collection. Use `Arm id` only when you intentionally want the older calibration naming behavior.
