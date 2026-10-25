"""Single-step VIOLA adapter; legacy dependencies stay in an isolated worker.

The GUI owns hardware and recording. This worker only consumes observations.
"""
from __future__ import annotations

import json
import os
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".cache/viola/venv/bin/python"
RGB_KEYS = ["observation.images.front", "observation.images.side"]


def checkpoint_file(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    candidates = sorted(path.glob("checkpoint-*.pt"))
    if len(candidates) != 1:
        raise ValueError(f"VIOLA needs exactly one checkpoint-*.pt in {path}, or an explicit .pt path")
    return candidates[0]


def is_viola_checkpoint(path):
    path = Path(path).expanduser()
    folder = path.parent if path.is_file() else path
    try:
        run = json.loads((folder / "run.json").read_text())
        return run.get("config", {}).get("name") == "CenterNetSpatialTemporalPolicy"
    except (OSError, ValueError):
        return False


def metadata(path):
    checkpoint = checkpoint_file(path)
    run = json.loads((checkpoint.parent / "run.json").read_text())
    if run.get("config", {}).get("name") != "CenterNetSpatialTemporalPolicy":
        raise ValueError("Not an official VIOLA spatial-temporal policy checkpoint")
    spec = run["dataset_spec"]
    for key in ("state_names", "action_names"):
        if len(spec[key]) != 10 or len(set(spec[key])) != 10:
            raise ValueError(f"VIOLA requires ten distinct {key}")
    return checkpoint, spec


def config_for_checkpoint(path):
    metadata(path)
    # CPU describes transport tensors in the GUI; the worker uses CUDA.
    return SimpleNamespace(
        type="viola", device="cpu", use_amp=False, n_action_steps=1,
        action_target="dataset_action",
        input_features={**{k: SimpleNamespace(shape=(3, 128, 128)) for k in RGB_KEYS},
                        "observation.state": SimpleNamespace(shape=(10,))},
        output_features={"action": SimpleNamespace(shape=(10,))},
    )


def validate_runtime(configuration):
    if (configuration["execution_mode"] != "synchronous"
            or configuration["replan_interval_steps"] != 1
            or configuration.get("prediction_steps") != 1
            or configuration["fusion_steps"] != 0
            or configuration.get("auto_replan") or configuration.get("use_amp")
            or not configuration.get("lock_grippers")):
        raise ValueError("VIOLA requires synchronous single-step prediction/replanning, no fusion/auto-replan/AMP, and locked grippers")
    if not PYTHON.is_file():
        raise ValueError("VIOLA environment is missing. Run: BASE_PYTHON=$(which python) bash mycode/setup_viola_baseline.sh")


class IdentityProcessor:
    def __call__(self, value):
        return value

    def reset(self):
        pass


class VIOLAEvalPolicy:
    def __init__(self, checkpoint, log_path, *, python=PYTHON, device="cuda"):
        self.checkpoint, self.spec = metadata(checkpoint)
        self.config = config_for_checkpoint(checkpoint)
        self.process = None
        self.connection = None
        self.log = Path(log_path).open("a", buffering=1)
        parent, child = socket.socketpair()
        try:
            self.connection = Connection(parent.detach())
            env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
            self.process = subprocess.Popen(
                [str(python), str(Path(__file__).resolve()), "--worker-fd", str(child.fileno()),
                 "--checkpoint", str(self.checkpoint), "--device", device],
                pass_fds=(child.fileno(),), stdout=self.log, stderr=self.log, cwd=ROOT, env=env,
            )
            child.close()
            ready = self._receive(180)
            if ready["spec"] != self.spec:
                raise ValueError("VIOLA run.json does not match checkpoint dataset_spec")
            self.inference_model_statistics = {
                key: value + ready["detector_statistics"][key]
                for key, value in ready["model_statistics"].items()
            }
            self.metadata = {"checkpoint_file": str(self.checkpoint), "worker_device": device,
                             "upstream_revision": ready["revision"], "history_length": 10,
                             "action_mode": "official_sampled_gmm", "policy_statistics": ready["model_statistics"],
                             "detector_statistics": ready["detector_statistics"],
                             "state_names": self.spec["state_names"], "action_names": self.spec["action_names"],
                             "worker_log": str(log_path)}
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"VIOLA worker did not respond within {timeout}s; see {self.log.name}")
        try:
            response = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"VIOLA worker exited; see {self.log.name}") from exc
        if "error" in response:
            raise RuntimeError(f"VIOLA inference failed: {response['error']}; see {self.log.name}")
        return response

    def reset(self):
        self.connection.send({"command": "reset"})
        self._receive(10)

    @staticmethod
    def observation_payload(batch):
        import numpy as np
        images = []
        for key in RGB_KEYS:
            image = batch[key].detach().cpu().numpy()
            if image.ndim != 4 or image.shape[:2] != (1, 3) or not np.isfinite(image).all():
                raise ValueError(f"Expected finite single-batch RGB image for {key}")
            if image.min() < 0 or image.max() > 1:
                raise ValueError(f"Expected unnormalized RGB in [0,1] for {key}")
            # LeRobot supplies RGB BCHW; the training adapter accepts OpenCV BGR HWC.
            rgb = np.rint(image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            images.append(np.ascontiguousarray(rgb[..., ::-1]))
        state = batch["observation.state"].detach().cpu().numpy()
        if state.shape != (1, 10) or not np.isfinite(state).all():
            raise ValueError("Expected finite VIOLA state with shape (1,10)")
        return {"command": "predict", "front": images[0], "side": images[1], "state": state[0]}

    def select_action(self, batch):
        import numpy as np
        import torch
        self.connection.send(self.observation_payload(batch))
        action = np.asarray(self._receive(10)["action"], dtype=np.float32)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("VIOLA worker returned invalid joint targets")
        return torch.from_numpy(action.copy()).unsqueeze(0)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            # Close only our inference worker, which has no robot/camera connection.
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.connection is not None:
            self.connection.close()
        self.log.close()


def worker(fd, checkpoint, device):
    import traceback
    import torch
    from viola_baseline import VIOLAInference, seed_all
    connection = Connection(fd)
    try:
        torch.set_num_threads(2)
        seed_all(1000)
        policy = VIOLAInference(checkpoint, device)
        policy.detector.model.requires_grad_(False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        def stats(model):
            parameters = list(model.parameters())
            return {"parameter_count": sum(p.numel() for p in parameters),
                    "trainable_parameter_count": sum(p.numel() for p in parameters if p.requires_grad),
                    "parameter_bytes": sum(p.numel()*p.element_size() for p in parameters),
                    "buffer_bytes": sum(p.numel()*p.element_size() for p in model.buffers())}
        connection.send({"spec": payload["dataset_spec"], "revision": payload["revision"],
                         "model_statistics": stats(policy.model), "detector_statistics": stats(policy.detector.model)})
        del payload
        while True:
            request = connection.recv()
            if request["command"] == "reset":
                policy.reset()
                connection.send({"ok": True})
            elif request["command"] == "predict":
                action = policy.predict(request["front"], request["side"], request["state"])
                connection.send({"action": action})
            else:
                raise ValueError("Unknown VIOLA worker request")
    except EOFError:
        pass
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        connection.send({"error": error})
    finally:
        connection.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-fd", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    worker(args.worker_fd, args.checkpoint, args.device)
