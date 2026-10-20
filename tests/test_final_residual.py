"""Final ViewQtoken residual ablations; optional real-checkpoint GPU smoke test."""

import os
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mycode.residual_eval import ACTResidualRuntime
from mycode.slow_fast_residual import (
    ActionPlan, QUERY_CONTEXT_VARIANTS, SEMANTIC_BASE_EXPERIMENTS,
    OnlineWindow, ResidualConfig, ResidualCorrector, context_record, residual_loss,
)
from mycode.slow_fast_semantic import QueryCapture
from mycode.train_slow_fast_residual import BasePolicyAdapter, episode_table, load_metadata, video_frames
from mycode.view_qtoken import ViewHeads


def config(variant):
    return ResidualConfig(variant=variant, fast_stride=6, fast_delay=3,
                          correction_steps=9, plan_steps=9, history_gap=0.3, residual_max_age=0.4)


def test_launcher_four_runs_and_two_caches():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "run_final_viewqtoken_residuals.sh", "all"], cwd=root,
                            env={**os.environ, "DRY_RUN": "1"}, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 12
    assert sum(" train " in line for line in lines) == 4
    assert sum(" shadow " in line for line in lines) == 4
    for line in lines:
        assert "--fast-stride 6 --fast-delay 3 --correction-steps 9" in line
        assert "--root " + str(root / "data/finaldata_seg") in line
    assert len({line.split("--cache ")[1].split()[0] for line in lines}) == 2
    assert len({line.split("--output ")[1].split()[0] for line in lines}) == 4


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.experiment = "FINAL-VIEWQTOKEN"
        self.config = SimpleNamespace(action_target="dataset_action", chunk_size=60)
        self.rgb_keys = ["observation.images.front", "observation.images.side"]
        model = SimpleNamespace(config=SimpleNamespace(metric_mode="encoder_tokens"),
                                metric_heads=nn.ModuleList([ViewHeads(512, n) for n in (1, 2, 1)]),
                                backbone=lambda x: {"feature_map": x})
        self.act_policy = SimpleNamespace(model=model)
        self.view_mask_suffixes = [["object", "tool"], ["object", "tool"]]
        self.semantic_calls = 0

    def _get_rgb_inputs(self, batch, denormalize):
        assert denormalize
        return [batch[k] for k in self.rgb_keys]

    def predict_view_logits_and_latent_from_rgbs(self, raw):
        self.semantic_calls += 1
        return raw, None

    def semantic_probabilities(self, logits):
        return [x.softmax(1) for x in logits]


def test_fast_feature_ablation_leaves_rgb_unchanged():
    adapter = object.__new__(BasePolicyAdapter)
    adapter.policy = FakePolicy()
    adapter.act, adapter.keys = adapter.policy.act_policy, adapter.policy.rgb_keys
    adapter.pre = lambda x: x
    state = torch.zeros(1, 10)
    images = [torch.rand(1, 3, 12, 16) for _ in range(2)]
    adapter.cfg = replace(config("viewqtoken_rgb"), image_height=12, image_width=16)
    rgb = adapter.features(images, state)
    assert adapter.policy.semantic_calls == 0
    adapter.cfg = replace(adapter.cfg, variant="viewqtoken_semquery")
    sem = adapter.features(images, state)
    torch.testing.assert_close(rgb, sem[:, :rgb.shape[-1]], rtol=0, atol=0)
    assert sem.shape[-1] > rgb.shape[-1]
    assert adapter.policy.semantic_calls == 1


@pytest.mark.parametrize("variant", ["viewqtoken_rgb", "viewqtoken_semquery"])
@pytest.mark.parametrize("mode", ["mlp", "gru"])
def test_runtime_loads_final_variants(tmp_path, variant, mode):
    cfg = config(variant)
    assert SEMANTIC_BASE_EXPERIMENTS[variant] == "FINAL-VIEWQTOKEN"
    query_dim = 1539 if variant in QUERY_CONTEXT_VARIANTS else 0
    model = ResidualCorrector(4 + query_dim, 223, 10, torch.ones(10), cfg, mode)
    path = tmp_path / "best.pt"
    names = [str(i) for i in range(10)]
    torch.save({"spec": model.spec, "model": model.state_dict(), "cache_id": "test",
                "contract": {"joints": names, "unit": "dataset_native", "scale": [1.] * 10}}, path)
    policy = FakePolicy()
    runtime = ACTResidualRuntime(path, mode, policy, None, None, names, names,
        dict(fps=30, replan_interval_steps=30, prediction_steps=60, fusion_steps=0,
             auto_replan=False, use_amp=False, execution_mode="asynchronous"))
    try:
        assert (runtime.adapter.query_capture is not None) == bool(query_dim)
        runtime.adapter.features = lambda rgbs, q: torch.zeros(1, 4)
        if query_dim:
            with pytest.raises(RuntimeError, match="no Qtoken"):
                runtime._infer_fast([], torch.zeros(10), torch.zeros(223), 0.)
            tokens = torch.randn(1, 3, 512)
            for i, head in enumerate(policy.act_policy.model.metric_heads):
                head(tokens[:, i])
            runtime.plan_features = runtime.adapter.query_capture.descriptor(torch.zeros(1, 10))
            torch.testing.assert_close(runtime.plan_features[:, :1536], tokens.flatten(1))
        residual = runtime._infer_fast([], torch.zeros(10), torch.zeros(223), 0.)
        assert residual.shape == (9, 10)
        assert not residual.any()
    finally:
        runtime.close()


@pytest.mark.skipif(os.environ.get("RUN_FINAL_RESIDUAL_SMOKE") != "1", reason="Opt-in local checkpoint test")
def test_real_final_checkpoint_and_four_heads(tmp_path):
    """Real RGB, original normalization, hard semantics, native targets, backward/reload."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data/finaldata_seg"
    checkpoint = root / "outputs/train/semantic/final-ViewQtoken/checkpoint_step_100000"
    info, records = load_metadata(dataset)
    states, actions, _ = episode_table(dataset, info, records[0])
    cfg = config("viewqtoken_rgb")
    adapter = BasePolicyAdapter(checkpoint, cfg, "cuda")
    images = [video_frames(dataset, info, records[0], k, [0, 6]) for k in adapter.keys]
    rgbs = [x[:1].cuda().float() / 255 for x in images]
    state = states[:1].cuda()
    batch = adapter.normalized(rgbs, state)
    with torch.no_grad():
        original = adapter.post(adapter.policy.predict_action_chunk(batch)).cuda()
    nominal = adapter.predict(rgbs, state)
    torch.testing.assert_close(nominal, original, rtol=0, atol=0)
    assert adapter.latest_plan_features is None
    assert not any(p.requires_grad for p in adapter.policy.parameters())
    fast_rgbs = [x[1:].cuda().float() / 255 for x in images]
    fast_state = states[6:7].cuda()
    rgb_features = adapter.features(fast_rgbs, fast_state)
    adapter.cfg = config("viewqtoken_semquery")
    adapter.query_capture = QueryCapture(adapter.act.model)
    try:
        sem_nominal = adapter.predict(rgbs, state)
        torch.testing.assert_close(sem_nominal, nominal, rtol=0, atol=0)
        plan_features = adapter.latest_plan_features.cuda()
        assert plan_features.shape == (1, 1539)
        assert plan_features[:, -3:].all()
        current_features = adapter.features(fast_rgbs, fast_state)
        torch.testing.assert_close(current_features[:, :rgb_features.shape[-1]], rgb_features, rtol=0, atol=0)
        assert rgb_features.shape[-1] == 6144
        assert current_features.shape[-1] == 6304
        enriched = torch.cat((current_features, plan_features), -1)
        plan = ActionPlan(0, 0., .1, nominal[0], 30)
        scale = actions.std(0).clamp_min(1e-3).cuda()  # Smoke-only scale, not a formal training cache.
        ctx = context_record(fast_state[0], state[0], actions[5].cuda(), plan, .2, .2, scale, cfg, True)
        base, valid = plan.at(.3, 9)
        target = actions[9:18].cuda()[None]
        for variant, feat in (("viewqtoken_rgb", rgb_features), ("viewqtoken_semquery", enriched)):
            for mode in ("mlp", "gru"):
                torch.manual_seed(1000)
                model = ResidualCorrector(feat.shape[-1], len(ctx), 10, torch.ones(10), config(variant), mode).cuda()
                batch = {"base": base[None], "target": target, "valid": valid[None]}
                f, c, lengths = feat[:, None].detach(), ctx[None, None], torch.ones(1, dtype=torch.long, device="cuda")
                assert not model(f, c, lengths).any()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                first = None
                for _ in range(12):
                    optimizer.zero_grad(set_to_none=True)
                    loss = residual_loss(model(f, c, lengths), batch, scale)
                    first = loss.item() if first is None else first
                    loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    optimizer.step()
                assert loss.item() < first
                assert not any(p.grad is not None for p in adapter.policy.parameters())
                path = tmp_path / f"{variant}-{mode}.pt"
                torch.save(model.state_dict(), path)
                other = ResidualCorrector(feat.shape[-1], len(ctx), 10, torch.ones(10), config(variant), mode).cuda()
                other.load_state_dict(torch.load(path, weights_only=True))
                with torch.no_grad():
                    torch.testing.assert_close(other(f, c, lengths), model(f, c, lengths), rtol=0, atol=0)
                    online = OnlineWindow(other).update(feat[0], ctx, .2)
                    torch.testing.assert_close(online, other(f, c, lengths)[0])
                print(f"{variant}/{mode}: dims={feat.shape[-1]}, smoke_loss={first:.6f}->{loss.item():.6f}, reload=OK")
    finally:
        adapter.query_capture.close()


@pytest.mark.skipif(os.environ.get("RUN_FINAL_RESIDUAL_SMOKE") != "1", reason="Opt-in local checkpoint test")
def test_real_cache_train_roundtrip(tmp_path, monkeypatch):
    """Two-episode scratch caches exercise the actual CLI training path, not formal runs."""
    import mycode.train_slow_fast_residual as runner

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data/finaldata_seg"
    info, records = load_metadata(data_root)
    records = records[:2]
    info = {**info, "total_episodes": 2, "total_frames": sum(r["length"] for r in records)}
    monkeypatch.setattr(runner, "load_metadata", lambda path: (info, records))
    original = None
    for variant in ("viewqtoken_rgb", "viewqtoken_semquery"):
        cfg = config(variant)
        args = runner.parser().parse_args([
            "cache", "--variant", variant, "--root", str(data_root),
            "--checkpoint", str(root / "outputs/train/semantic/final-ViewQtoken/checkpoint_step_100000"),
            "--cache", str(tmp_path / variant), "--output", str(tmp_path / "unused"),
            "--epochs", "2", "--batch-size", "64",
        ])
        runner.prepare_cache(args, cfg)
        data, scale, bounds, manifest, contract = runner.load_cache(Path(args.cache), cfg)
        split = json.loads((Path(args.cache) / "split.json").read_text())
        if original is None:
            original = (data, scale, bounds, split)
        else:
            torch.testing.assert_close(scale, original[1], rtol=0, atol=0)
            torch.testing.assert_close(bounds, original[2], rtol=0, atol=0)
            assert split == original[3]
            for group in ("train", "val"):
                for previous, current in zip(original[0][group], data[group], strict=True):
                    for key in ("base", "target", "context", "valid", "time"):
                        torch.testing.assert_close(previous[key], current[key], rtol=0, atol=0)
                    torch.testing.assert_close(previous["features"], current["features"][:, :6144], rtol=0, atol=0)
        assert len(manifest["segmentation_bundle"]) == 6
        for mode in ("mlp", "gru"):
            torch.manual_seed(1000)
            args.mode = mode
            args.output = str(tmp_path / f"{variant}-{mode}")
            runner.train(args, cfg)
            result = json.loads((Path(args.output) / "result.json").read_text())
            assert result["status"] == "verified" and result["reload_ok"]
            assert result["online_offline_max_diff"] <= 1e-4
            runner.shadow(args, cfg)
            replay = json.loads((Path(args.output) / "shadow_result.json").read_text())
            assert not replay["robot_io"]
            assert replay["attempted_residual_updates"] > 0
            assert replay["accepted_residual_updates"] > 0
