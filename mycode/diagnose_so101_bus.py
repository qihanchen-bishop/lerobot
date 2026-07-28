#!/usr/bin/env python

"""Diagnose SO101 Feetech serial ports and motor bus replies.

This tool is intentionally read-only for the servos.  It opens each selected
serial port, pings motor IDs 1..6, and optionally scans common Feetech
baudrates.  It is useful when LeRobot reports:

    Full found motor list (id: model_number): {}

Run from the repository root, for example:

    conda run -n lerobot python mycode/diagnose_so101_bus.py --port /dev/ttyACM2 --scan-baud
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CURRENT_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(CURRENT_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_SRC))

from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig


DEFAULT_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_BAUDRATES = [
    4_800,
    9_600,
    14_400,
    19_200,
    38_400,
    57_600,
    115_200,
    128_000,
    250_000,
    500_000,
    1_000_000,
]


@dataclass
class PortReport:
    port: str
    serial: str = ""
    occupied_by: str = ""
    open_error: str = ""
    pings: dict[int, int | None] | None = None
    baud_scan: dict[int, dict[int, int]] | None = None


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return ""
    return (result.stdout + result.stderr).strip()


def _udev_property(port: str, key: str) -> str:
    output = _run(["udevadm", "info", "-q", "property", "-n", port])
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return ""


def _occupancy(port: str) -> str:
    output = _run(["fuser", "-v", port])
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines)


def _make_robot(port: str) -> SO101Follower:
    config = SOFollowerRobotConfig(id=f"diagnose_{Path(port).name}", port=port, cameras={})
    return SO101Follower(config)


def _ping_ids(robot: SO101Follower, ids: list[int], retries: int) -> dict[int, int | None]:
    return {motor_id: robot.bus.ping(motor_id, num_retry=retries, raise_on_error=False) for motor_id in ids}


def diagnose_port(port: str, ids: list[int], retries: int, scan_baud: bool) -> PortReport:
    report = PortReport(
        port=port,
        serial=_udev_property(port, "ID_SERIAL_SHORT"),
        occupied_by=_occupancy(port),
    )
    if report.occupied_by:
        return report

    robot = _make_robot(port)
    try:
        robot.bus._connect(handshake=False)
        robot.bus.set_timeout()
        report.pings = _ping_ids(robot, ids, retries)
        if scan_baud:
            report.baud_scan = {}
            for baudrate in DEFAULT_BAUDRATES:
                robot.bus.set_baudrate(baudrate)
                found = {
                    motor_id: model
                    for motor_id, model in _ping_ids(robot, ids, retries).items()
                    if model is not None
                }
                report.baud_scan[baudrate] = found
    except Exception as exc:
        report.open_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if robot.bus.is_connected:
                robot.bus.port_handler.closePort()
        except Exception:
            pass
    return report


def print_report(report: PortReport, expected_model: int) -> None:
    serial = f" serial={report.serial}" if report.serial else ""
    print(f"\n== {report.port}{serial} ==")

    if report.occupied_by:
        print("BUSY: another process is using this port:")
        print(report.occupied_by)
        return

    if report.open_error:
        print(f"OPEN FAILED: {report.open_error}")
        return

    pings = report.pings or {}
    found = {motor_id: model for motor_id, model in pings.items() if model is not None}
    print(f"default baud ping: {pings}")

    if found and all(model == expected_model for model in found.values()):
        if len(found) == len(pings):
            print("result: OK, all expected motors replied.")
        else:
            missing = [motor_id for motor_id, model in pings.items() if model is None]
            print(f"result: PARTIAL, missing motor IDs {missing}.")
    elif found:
        print(f"result: WRONG MODEL OR MIXED BUS, found {found}.")
    else:
        print("result: NO MOTOR REPLIES on the default baudrate.")

    if report.baud_scan is not None:
        any_found = False
        print("baud scan:")
        for baudrate, baud_found in report.baud_scan.items():
            if baud_found:
                any_found = True
            print(f"  {baudrate}: {baud_found}")
        if not any_found:
            print("baud scan result: no motor replied on any scanned baudrate.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        action="append",
        help="Serial port to diagnose. Can be passed multiple times. Defaults to all /dev/ttyACM* ports.",
    )
    parser.add_argument("--ids", default="1,2,3,4,5,6", help="Comma-separated motor IDs to ping.")
    parser.add_argument("--retries", type=int, default=1, help="Extra ping retries per motor.")
    parser.add_argument("--scan-baud", action="store_true", help="Also scan common Feetech baudrates.")
    parser.add_argument("--expected-model", type=int, default=777, help="Expected STS3215 model number.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = [int(part) for part in args.ids.split(",") if part.strip()]
    ports = args.port or sorted(glob.glob("/dev/ttyACM*"))
    if not ports:
        print("No /dev/ttyACM* ports found.")
        raise SystemExit(1)

    print(f"uid={os.getuid()} user={os.environ.get('USER', '')}")
    modem_state = _run(["systemctl", "is-active", "ModemManager"])
    if modem_state:
        print(f"ModemManager: {modem_state}")

    for port in ports:
        print_report(diagnose_port(port, ids, args.retries, args.scan_baud), args.expected_model)


if __name__ == "__main__":
    main()
