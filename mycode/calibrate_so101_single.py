#!/usr/bin/env python

"""Run LeRobot calibration with the STS3215 single-turn feedback fix.

Some STS3215 firmware can retain multi-turn position feedback mode.  Older
LeRobot checkouts then read positions outside 0..4095 and fail while writing
the signed 11-bit homing offset.  Upstream LeRobot fixes this by clearing bit
4 of the Phase register during motor configuration.  This local entry point
applies the same compatibility fix before invoking the standard calibrator.
"""

from __future__ import annotations

from functools import wraps

from lerobot.motors.feetech import FeetechMotorsBus


def install_sts3215_single_turn_fix() -> None:
    original = FeetechMotorsBus.configure_motors
    if getattr(original, "_sts3215_single_turn_fix", False) is not True:

        @wraps(original)
        def configure_motors(self: FeetechMotorsBus, *args, **kwargs) -> None:
            original(self, *args, **kwargs)
            sts3215_motors: list[str] = []
            for motor, config in self.motors.items():
                if config.model != "sts3215":
                    continue
                sts3215_motors.append(motor)
                phase = self.read("Phase", motor, normalize=False)
                if phase & 0x10:
                    self.write("Phase", motor, phase & ~0x10)

            # Changing the persistent Phase setting does not clear an already
            # accumulated multi-turn count on every firmware revision. Detect
            # that state before calibration computes an invalid homing offset.
            positions = self.sync_read("Present_Position", sts3215_motors, normalize=False)
            out_of_range = {
                motor: int(position)
                for motor, position in positions.items()
                # With an existing Homing_Offset, a valid single-turn
                # Present_Position may be negative near the wrap boundary.
                # Only values whose magnitude reaches a full encoder turn
                # indicate a stale multi-turn accumulator.
                if abs(int(position)) >= self.model_resolution_table[self.motors[motor].model]
            }
            if out_of_range:
                details = ", ".join(f"{motor}={position}" for motor, position in out_of_range.items())
                raise RuntimeError(
                    "STS3215 still has stale multi-turn position feedback "
                    f"({details}). Switch off the arm's motor power, wait 10 seconds, power it on, "
                    "and start calibration again. Close programs that are using the USB port while "
                    "power-cycling."
                )

        configure_motors._sts3215_single_turn_fix = True  # type: ignore[attr-defined]
        FeetechMotorsBus.configure_motors = configure_motors

    # Broadcast discovery is unreliable with some controller/firmware
    # combinations even though direct ID pings work. The normal handshake
    # knows the expected IDs, so safely fall back to pinging those IDs only.
    original_broadcast_ping = FeetechMotorsBus.broadcast_ping
    if getattr(original_broadcast_ping, "_direct_ping_fallback", False) is not True:

        @wraps(original_broadcast_ping)
        def broadcast_ping(self: FeetechMotorsBus, num_retry: int = 0, raise_on_error: bool = False):
            found = original_broadcast_ping(self, num_retry=num_retry, raise_on_error=raise_on_error)
            if found or not self.ids:
                return found
            direct_found: dict[int, int] = {}
            for motor_id in self.ids:
                model_number = self.ping(motor_id, num_retry=max(1, num_retry), raise_on_error=False)
                if model_number is not None:
                    direct_found[motor_id] = model_number
            return direct_found or found

        broadcast_ping._direct_ping_fallback = True  # type: ignore[attr-defined]
        FeetechMotorsBus.broadcast_ping = broadcast_ping


def main() -> None:
    install_sts3215_single_turn_fix()
    from lerobot.scripts.lerobot_calibrate import main as calibrate_main

    calibrate_main()


if __name__ == "__main__":
    main()
