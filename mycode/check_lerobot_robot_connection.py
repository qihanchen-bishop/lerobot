#!/usr/bin/env python
"""Connect to a LeRobot robot once, read one observation, then disconnect."""

from dataclasses import dataclass
from pprint import pformat

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    make_robot_from_config,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class RobotConnectionCheckConfig:
    robot: RobotConfig


@parser.wrap()
def check_connection(cfg: RobotConnectionCheckConfig) -> None:
    init_logging()
    print("Robot config:")
    print(pformat(cfg.robot))
    robot = make_robot_from_config(cfg.robot)
    try:
        print("Connecting robot...")
        robot.connect()
        print("Connected. Reading one observation...")
        observation = robot.get_observation()
        print(f"Observation keys: {sorted(observation)}")
        print("Connection check OK.")
    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    check_connection()


if __name__ == "__main__":
    main()
