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
