import pytest
import torch

from mycode.slow_fast_residual import (
    ActionPlan,
    MultiRateExecutor,
    OnlineWindow,
    ResidualConfig,
    ResidualCorrector,
    ResidualSequenceDataset,
    context_record,
    observation_valid,
    residual_loss,
    validate_joints,
)


def plan(i=0, time=0):
    return ActionPlan(i, time, time, torch.arange(120).reshape(60, 2).float(), 30)


@pytest.mark.parametrize("variant", ["act", "unet", "qtoken"])
def test_config(variant):
    assert ResidualConfig(variant=variant).history == 8
    with pytest.raises(ValueError):
        ResidualConfig(fast_delay=-1)
    with pytest.raises(ValueError):
        ResidualConfig(unit="degrees")
    with pytest.raises(ValueError):
        ResidualConfig(correction_steps=1)


@pytest.mark.parametrize("variant", ["act", "unet", "qtoken"])
def test_stride6_delay3_horizon9(variant):
    cfg = ResidualConfig(
        variant=variant,
        fast_stride=6,
        fast_delay=3,
        correction_steps=9,
        plan_steps=9,
        history_gap=0.3,
        residual_max_age=0.4,
    )
    model = ResidualCorrector(4, 3, 2, torch.ones(2), cfg)
    window = OnlineWindow(model)
    for i in range(8):
        result = window.update(torch.zeros(4), torch.zeros(3), i * 0.2)
    assert len(window.records) == 8
    assert result.shape == (9, 2)
    window.update(torch.zeros(4), torch.zeros(3), 2.0)
    assert len(window.records) == 1
    e = MultiRateExecutor(cfg, torch.ones(2))
    p = plan()
    assert e.accept_plan(p, 0.0)
    assert e.accept_residual(0, torch.ones(9, 2), 9 / 30, [0.2, 0.2], 0.28)
    # Newer update overwrites only the overlap, not the current old slot.
    assert e.accept_residual(0, torch.full((9, 2), 0.5), 15 / 30, [0.4, 0.4], 0.48)
    torch.testing.assert_close(e.command(14 / 30), p.actions[14] + 1)
    torch.testing.assert_close(e.command(15 / 30), p.actions[15] + 0.5)
    torch.testing.assert_close(e.command(23 / 30), p.actions[23] + 0.5)


def test_time_alignment_and_mask():
    p = plan()
    a, mask = p.at(59 / 30, 2)
    assert mask.tolist() == [True, False]
    assert a.tolist() == [[118, 119], [0, 0]]
    assert not p.at(-1, 2)[1].any()
    cfg = ResidualConfig()
    q = torch.tensor([1.0, 2.0])
    c = context_record(q, q, q, p, 0.0, 1 / 15, torch.ones(2), cfg)
    assert c[6:10].tolist() == [2.0, 3.0, 4.0, 5.0]  # fast delay = one target slot


def test_joint_order_and_units():
    validate_joints(["a", "b"], ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError):
        validate_joints(["a", "b"], ["b", "a"], ["a", "b"])
    cfg = ResidualConfig()
    model = ResidualCorrector(4, 3, 2, torch.tensor([2.0, 5.0]), cfg)
    with torch.no_grad():
        model.head.bias.fill_(100.0)
    r = model(torch.zeros(1, 8, 4), torch.zeros(1, 8, 3), torch.tensor([8]))
    torch.testing.assert_close(r[0, 0], torch.tensor([2.0, 5.0]))


def test_zero_and_disabled_equivalence():
    cfg = ResidualConfig()
    p = plan()
    executor = MultiRateExecutor(cfg, torch.ones(2))
    assert executor.accept_plan(p, 0.0)
    assert executor.accept_residual(0, torch.zeros(2, 2), 0.0, [0.0, 0.0], 0.0)
    torch.testing.assert_close(executor.command(0.0), p.actions[0], atol=0, rtol=0)
    assert executor.command(0.0) is None
    assert executor.accept_residual(0, torch.ones(2, 2), 1 / 30, [0.0, 0.0], 0.0)
    torch.testing.assert_close(executor.command(1 / 30, enabled=False), p.actions[1], atol=0, rtol=0)


def test_plan_switch_expiry_reset():
    e = MultiRateExecutor(ResidualConfig(), torch.ones(2))
    assert e.accept_plan(plan(), 0.0)
    assert e.accept_residual(0, torch.ones(2, 2), 0.0, [0.0, 0.0], 0.0)
    assert e.accept_plan(plan(1), 0.0)
    assert not e.pending
    assert not e.accept_residual(0, torch.ones(2, 2), 0.0, [0.0, 0.0], 0.0)
    assert not e.accept_plan(plan(0), 0.0)
    assert e.command(3.0) is None
    e.reset()
    assert e.plan is None and e.command(0.0) is None
    assert not e.accept_plan(plan(), 3.0)


def test_late_missing_unsynchronized_observations():
    cfg = ResidualConfig()
    for times in ([], [0.0], [0.0, float("nan")], [0.0, 0.1], [0.0, 0.0]):
        assert not observation_valid(times, 1.0, cfg)
    assert observation_valid([0.99, 1.0], 1.0, cfg)
    e = MultiRateExecutor(cfg, torch.ones(2))
    e.accept_plan(plan(), 0.0)
    assert not e.accept_residual(0, torch.ones(2, 2), 0.0, [0.1, 0.1], 0.1)
    assert not e.accept_residual(0, torch.full((2, 2), float("nan")), 0.1, [0.0, 0.0], 0.0)


def test_velocity_acceleration_limits():
    e = MultiRateExecutor(
        ResidualConfig(),
        torch.ones(2),
        lower=torch.zeros(2),
        upper=torch.full((2,), 5.0),
        max_velocity=torch.ones(2),
        max_acceleration=torch.ones(2),
    )
    e.accept_plan(plan(), 0.0)
    commands = torch.stack([e.command(t / 30) for t in range(4)])
    velocity = commands.diff(dim=0) * 30
    assert velocity.abs().max() <= 1.00001
    assert (velocity.diff(dim=0) * 30).abs().max() <= 1.0001


def test_partial_late_residual_keeps_original_target_slot():
    e = MultiRateExecutor(ResidualConfig(), torch.ones(2))
    p = plan()
    e.accept_plan(p, 0.0)
    assert e.accept_residual(0, torch.tensor([[0.2, 0.2], [0.7, 0.7]]), 1 / 30, [0.0, 0.0], 0.04)
    assert list(e.pending) == [2]
    assert e.expired_prefix_slots == 1
    torch.testing.assert_close(e.command(2 / 30), p.actions[2] + 0.7)


def episode(offset=0):
    return {
        "features": torch.randn(12, 4) + offset,
        "context": torch.randn(12, 3),
        "base": torch.zeros(12, 2, 2),
        "target": torch.ones(12, 2, 2),
        "valid": torch.ones(12, 2),
        "age": torch.zeros(12),
        "time": torch.arange(12, dtype=torch.float64) / 15,
    }


@pytest.mark.parametrize("stride,horizon", [(2, 2), (6, 9)])
def test_episode_windows_and_online_batch(stride, horizon):
    cfg = ResidualConfig(
        fast_stride=stride,
        correction_steps=horizon,
        plan_steps=max(6, horizon),
        history_gap=0.3 if stride == 6 else None,
    )
    a, b = episode(), episode(1000)
    for ep in (a, b):
        ep["time"] = torch.arange(12, dtype=torch.float64) * stride / 30
    dataset = ResidualSequenceDataset([a, b], cfg)
    assert dataset[12]["lengths"] == 1
    assert torch.equal(dataset[12]["features"][0], b["features"][0])
    assert not dataset[12]["features"][1:].any()
    model = ResidualCorrector(4, 3, 2, torch.ones(2), cfg).eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.02)
    online = OnlineWindow(model)
    for i in range(12):
        result = online.update(a["features"][i], a["context"][i], float(a["time"][i]))
        x = dataset[i]
        expected = model(x["features"][None], x["context"][None], x["lengths"][None])[0]
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    assert online.update(a["features"][0], a["context"][0], float(a["time"][-1])) is None
    online.reset()
    assert len(online.records) == 0


def test_gap_and_tail_loss():
    cfg = ResidualConfig()
    e = episode()
    e["time"][6:] += 1
    assert ResidualSequenceDataset([e], cfg)[6]["lengths"] == 1
    pred = torch.zeros(1, 2, 2)
    b = {
        "base": pred,
        "target": torch.tensor([[[1.0, 1.0], [1e6, 1e6]]]),
        "valid": torch.tensor([[1.0, 0.0]]),
    }
    torch.testing.assert_close(residual_loss(pred, b, torch.ones(2)), torch.tensor(0.95))
    bad = dict(e, time=torch.zeros(12))
    with pytest.raises(ValueError):
        ResidualSequenceDataset([bad], cfg)


def test_semantic_validity_and_missing_masks():
    from mycode.slow_fast_semantic import semantic_descriptor

    reference = torch.zeros(1, 10)
    classes = ["object", "region", "tool", "occluder"]
    missing = semantic_descriptor(None, classes, reference)
    assert missing.shape == (1, 58) and not missing.any()
    p = torch.zeros(1, 5, 16, 16)
    p[:, 0] = 1
    result = semantic_descriptor(p, classes, reference)
    assert not result[:, -8:].any()  # Invisible pairs are not valid zero distances.
    assert not semantic_descriptor(torch.zeros_like(p), classes, reference).any()
    assert not semantic_descriptor(torch.full_like(p, float("nan")), classes, reference).any()
    p.fill_(0.2)
    assert not semantic_descriptor(p, classes, reference)[:, -8:].any()
    p.zero_()
    p[:, 0] = 1
    for index, x in ((1, 1), (2, 6), (3, 11)):
        p[:, 0, 2:5, x : x + 3] = 0
        p[:, index, 2:5, x : x + 3] = 1
    result = semantic_descriptor(p, classes, reference)
    assert result[0, -5] == 1 and result[0, -1] == 1
    assert result[0, -8] < 0 and result[0, -4] < 0
    with pytest.raises(ValueError):
        semantic_descriptor(p[:, :3], classes, reference)


@pytest.mark.parametrize("length", [0, 1, 3, 5])
def test_query_length_validity_and_missing(length):
    from mycode.slow_fast_semantic import query_descriptor

    reference = torch.zeros(2, 10)
    x = torch.ones(2, length, 512)
    out = query_descriptor(x, reference)
    assert out.shape == (2, 1539)
    assert out[:, -3:].sum() == 2 * min(length, 3)
    if length:
        x[:, 0, 0] = float("nan")
        out = query_descriptor(x, reference)
        assert torch.isfinite(out).all() and not out[:, -3].any()
        assert not out[:, :512].any()
    assert not query_descriptor(None, reference).any()
    with pytest.raises(ValueError):
        query_descriptor(torch.ones(2, 3, 128), reference)


def test_query_capture_keeps_original_forward():
    from types import SimpleNamespace

    from mycode.slow_fast_semantic import QueryCapture

    model = SimpleNamespace(
        config=SimpleNamespace(metric_mode="encoder_tokens"),
        metric_heads=torch.nn.ModuleList([torch.nn.Linear(512, 1) for _ in range(3)]),
    )
    x = torch.randn(2, 3, 512)
    before = [head(x[:, i]) for i, head in enumerate(model.metric_heads)]
    capture = QueryCapture(model)
    after = [head(x[:, i]) for i, head in enumerate(model.metric_heads)]
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(capture.descriptor(torch.zeros(2, 10))[:, :-3], x.flatten(1))
    capture.clear()
    assert not capture.descriptor(torch.zeros(2, 10)).any()
    capture.close()


def test_padding_cannot_leak_future_features():
    model = ResidualCorrector(4, 3, 2, torch.ones(2), ResidualConfig()).eval()
    x, c, n = torch.randn(1, 8, 4), torch.randn(1, 8, 3), torch.tensor([3])
    assert not model(x, c, n).any()
    with torch.no_grad():
        model.head.weight.normal_(std=0.1)
    before = model(x, c, n)
    x[:, 3:] = 10000
    c[:, 3:] = -10000
    torch.testing.assert_close(before, model(x, c, n), atol=0, rtol=0)


def test_pending_query_plan_cannot_replace_active_plan():
    e = MultiRateExecutor(ResidualConfig(), torch.ones(2))
    active = plan()
    pending = ActionPlan(30, 1.0, 1.1, active.actions, 30)
    token_by_plan = {0: torch.zeros(1539), 30: torch.ones(1539)}
    assert e.accept_plan(active, 0.0)
    assert not e.accept_plan(pending, 1.0)
    assert not token_by_plan[e.plan.plan_id].any()
    assert e.accept_plan(pending, 1.1)
    assert token_by_plan[e.plan.plan_id].all()


def test_next_update_preserves_current_pending_slot():
    e = MultiRateExecutor(ResidualConfig(), torch.ones(2))
    p = plan()
    e.accept_plan(p, 0.0)
    e.accept_residual(0, torch.tensor([[0.2, 0.2], [0.7, 0.7]]), 1 / 30, [0.0, 0.0], 0.0)
    e.command(1 / 30)
    assert e.accept_residual(0, torch.zeros(2, 2), 3 / 30, [2 / 30, 2 / 30], 2 / 30)
    torch.testing.assert_close(e.command(2 / 30), p.actions[2] + 0.7)
    assert list(e.pending) == [3, 4]


def test_new_update_does_not_refresh_older_observation_age():
    e = MultiRateExecutor(ResidualConfig(), torch.ones(2))
    p = plan()
    e.accept_plan(p, 0.0)
    assert e.accept_residual(0, torch.ones(2, 2), 10 / 30, [0.0, 0.0], 0.0)
    assert e.accept_residual(0, torch.zeros(2, 2), 11 / 30, [0.32, 0.32], 0.32)
    torch.testing.assert_close(e.command(10 / 30), p.actions[10], atol=0, rtol=0)
