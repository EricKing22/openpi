"""Tests for probe capture and the shard writer.

Everything runs on fabricated tensors, so no checkpoint and no GPU are needed. What is not
covered here is whether the hooks are in the right *places* in the model -- that only shows
up when a real policy runs. See docs/probe_capture.md.
"""

import json

import numpy as np
import pytest
import torch

from openpi.probe import capture
from openpi.probe import codec
from openpi.probe import recorder

# Stand-in dimensions for a pi0.5 forward pass, small enough to keep the tests fast.
DEPTH = 5  # transformer depth 4, plus the post-final-norm entry
IMG_TOKENS, CAMERAS, PREFIX_DIM = 6, 3, 8
LANG_TOKENS, VALID_LANG = 4, 2
HORIZON, SUFFIX_LEN, SUFFIX_DIM = 4, 5, 7
PREFIX_LEN = CAMERAS * IMG_TOKENS + LANG_TOKENS
LANG_SPAN = (CAMERAS * IMG_TOKENS, PREFIX_LEN)
IMAGE_SPANS = [(i * IMG_TOKENS, (i + 1) * IMG_TOKENS) for i in range(CAMERAS)]
PAD_MASK = torch.tensor([[True] * (CAMERAS * IMG_TOKENS + VALID_LANG) + [False] * (LANG_TOKENS - VALID_LANG)])


def _encode_bf16(values: np.ndarray) -> np.ndarray:
    """float32 -> raw bfloat16 bits in an int16 array, round-to-nearest-even like torch."""
    u32 = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + bias) >> 16).astype(np.uint16).view(np.int16)


def _simulate_model(num_steps=3, tag=0.0):
    """Replays the call sequence the PyTorch model makes, with random tensors.

    `tag` is written into the first element of every xt_traj snapshot so a test can tell
    which call a stored row came from.
    """
    prefix_hidden = tuple(torch.randn(1, PREFIX_LEN, PREFIX_DIM) for _ in range(DEPTH))
    suffix_hidden = [tuple(torch.randn(1, SUFFIX_LEN, SUFFIX_DIM) for _ in range(DEPTH)) for _ in range(num_steps)]

    capture.set_action_horizon(HORIZON)
    capture.record_observation(
        torch.zeros(1, LANG_TOKENS, dtype=torch.int32),
        PAD_MASK[:, LANG_SPAN[0] :],
        [torch.tensor([True]), torch.tensor([True]), torch.tensor([False])],
        torch.zeros(1, 3),
    )
    for _ in range(CAMERAS):
        capture.record("siglip_tokens", torch.randn(1, IMG_TOKENS, PREFIX_DIM))
    capture.record_prefix_layout(IMAGE_SPANS, LANG_SPAN, PAD_MASK)
    capture.record_prefix_hidden(prefix_hidden)

    def tagged_xt():
        x = torch.randn(1, HORIZON, 3)
        x[0, 0, 0] = tag
        return x

    for step in range(num_steps):
        capture.set_denoise_step(step)
        capture.record("xt_traj", tagged_xt())
        capture.record("adarms_cond", torch.randn(1, SUFFIX_DIM))
        capture.record_suffix_hidden(suffix_hidden[step])
        capture.record("vt", torch.randn(1, HORIZON, 3))
    capture.record("xt_traj", tagged_xt())
    return prefix_hidden, suffix_hidden


def _run_capture(config, num_steps=3):
    torch.manual_seed(0)
    with capture.session(config) as cap:
        prefix_hidden, suffix_hidden = _simulate_model(num_steps)
        arrays, meta = cap.result()
    return arrays, meta, prefix_hidden, suffix_hidden


# =============================================================================
# Capture
# =============================================================================


def test_capture_is_inert_outside_a_session():
    assert not capture.active()
    assert not capture.wants_prefix_hidden()
    assert not capture.wants_suffix_hidden()
    capture.record("xt_traj", torch.zeros(1, 2, 2))  # a no-op, not an error


def test_capture_only_requests_hidden_states_it_will_store():
    # The output_hidden_states flag on each forward is driven by these, so a site that was
    # not asked for costs nothing at all.
    with capture.session(capture.CaptureConfig(sites={"xt_traj"})):
        assert not capture.wants_prefix_hidden()
        assert not capture.wants_suffix_hidden()
    with capture.session(capture.CaptureConfig(sites={"prefix_full"})):
        assert capture.wants_prefix_hidden()
        assert not capture.wants_suffix_hidden()


def test_capture_shapes_and_layer_step_filtering():
    config = capture.CaptureConfig(sites=capture.PILOT_SITES, suffix_layers=(0, -1), denoise_steps=(0, 2))
    arrays, meta, _, _ = _run_capture(config)

    assert arrays["prefix_pool"].shape == (1, DEPTH, CAMERAS + 2, PREFIX_DIM)
    assert arrays["prefix_full"].shape == (1, DEPTH, PREFIX_LEN, PREFIX_DIM)
    assert arrays["siglip_tokens"].shape == (1, CAMERAS, IMG_TOKENS, PREFIX_DIM)
    # 2 of 5 layers, 2 of 3 steps, and the suffix sliced to the last action_horizon tokens.
    assert arrays["suffix_hidden"].shape == (1, 2, 2, HORIZON, SUFFIX_DIM)
    assert arrays["xt_traj"].shape == (1, 4, HORIZON, 3)  # num_steps + 1 snapshots
    assert arrays["vt"].shape == (1, 3, HORIZON, 3)
    assert arrays["adarms_cond"].shape == (1, 3, SUFFIX_DIM)  # not step-filtered

    # Axis ids record which real layer / step each stored position is, so a partial capture
    # stays interpretable.
    assert meta["suffix_hidden"]["axis_ids"] == {"layer": [0, DEPTH - 1], "step": [0, 2]}
    assert meta["prefix_pool"]["axis_ids"]["pool"] == ["img0", "img1", "img2", "lang_mean", "lang_last"]
    assert meta["prefix_pool"]["axis_ids"]["layer"] == list(range(DEPTH))
    assert meta["suffix_hidden"]["shape"] == [2, 2, HORIZON, SUFFIX_DIM]


def test_capture_pooling_matches_the_definition():
    arrays, _, prefix_hidden, suffix_hidden = _run_capture(capture.CaptureConfig(sites=capture.PILOT_SITES))
    pooled = codec.decode(arrays["prefix_pool"], codec.BF16_RAW)[0]  # [layer, pool, dim]

    for layer in range(DEPTH):
        hidden = prefix_hidden[layer][0]
        for camera in range(CAMERAS):
            expected = hidden[camera * IMG_TOKENS : (camera + 1) * IMG_TOKENS].mean(0)
            assert np.allclose(pooled[layer, camera], expected.numpy(), rtol=0.02)
        lang = hidden[CAMERAS * IMG_TOKENS :]
        # The language mean covers only valid prompt positions, not the padding.
        assert np.allclose(pooled[layer, CAMERAS], lang[:VALID_LANG].mean(0).numpy(), rtol=0.02)
        # "lang_last" is the last valid position -- the "Action: " token for pi0.5.
        assert np.allclose(pooled[layer, CAMERAS + 1], lang[VALID_LANG - 1].numpy(), rtol=0.02)

    # The suffix keeps the last action_horizon tokens, which is what feeds action_out_proj.
    suffix = codec.decode(arrays["suffix_hidden"], codec.BF16_RAW)[0]  # [layer, step, token, dim]
    assert np.allclose(suffix[2, 1], suffix_hidden[1][2][0, -HORIZON:].numpy(), rtol=0.02)


def test_capture_rejects_double_recording_and_bad_layers():
    with capture.session(capture.CaptureConfig(sites={"prefix_pool"})) as cap:
        cap.record_prefix_layout([(0, 2)], (2, 4), torch.ones(1, 4, dtype=torch.bool))
        hidden = tuple(torch.randn(1, 4, 3) for _ in range(2))
        cap.record_prefix_hidden(hidden)
        with pytest.raises(RuntimeError, match="recorded twice"):
            cap.record_prefix_hidden(hidden)

    with capture.session(capture.CaptureConfig(sites={"suffix_hidden"}, suffix_layers=(99,))) as cap:
        cap.set_action_horizon(2)
        with pytest.raises(IndexError, match="out of range"):
            cap.record_suffix_hidden(tuple(torch.randn(1, 2, 3) for _ in range(2)))


def test_capture_config_keeps_explicit_sites():
    config = capture.CaptureConfig(sites={"prefix_pool"})
    assert config.sites == {"prefix_pool"}


def test_capture_rejects_nested_sessions():
    with capture.session(capture.CaptureConfig()):
        assert capture.active()
        with pytest.raises(RuntimeError, match="already active"), capture.session(capture.CaptureConfig()):
            pass
    assert not capture.active()


def test_bfloat16_roundtrip_survives_float16_overflow():
    # 65600 is past the float16 ceiling; the later Gemma layers do reach magnitudes like
    # this, which is why activations are not stored as float16.
    values = np.array([1.5, -2.25, 0.0, 65600.0, 6e-5], dtype=np.float32)
    decoded = codec.decode(_encode_bf16(values), codec.BF16_RAW)
    assert np.all(np.isfinite(decoded))
    assert np.allclose(decoded, values, rtol=0.01)


# =============================================================================
# Recorder
# =============================================================================


class _FakePolicy:
    """Stands in for a Policy: its `infer` makes the same capture calls the model would."""

    def __init__(self):
        self.calls = 0

    def infer(self, obs, **kwargs):
        if capture.active():
            _simulate_model(tag=float(self.calls))
        self.calls += 1
        return {"actions": np.zeros((HORIZON, 3), dtype=np.float32), "kwargs": sorted(kwargs)}


def test_recorder_writes_shards_and_reloads_in_call_order(tmp_path):
    torch.manual_seed(0)
    policy = _FakePolicy()
    with recorder.ActivationRecorder(policy, tmp_path, config=capture.CaptureConfig.pilot(), flush_every=4) as rec:
        for _ in range(10):
            rec.infer({"prompt": "pick up the block"})
        assert rec.num_rows == 10

    assert policy.calls == 10
    # 10 rows at 4 per shard: two full shards plus the remainder flushed on close.
    assert len(sorted(tmp_path.glob("shard_*.npz"))) == 3

    hidden = recorder.load_site(tmp_path, "suffix_hidden")
    assert hidden.shape == (10, DEPTH, 3, HORIZON, SUFFIX_DIM)
    assert hidden.dtype == np.float32  # decoded from the raw bfloat16 bits

    # The tag in xt_traj proves the rows come back in the order they were recorded, across
    # the shard boundaries.
    xt = recorder.load_site(tmp_path, "xt_traj")
    assert xt.shape == (10, 4, HORIZON, 3)
    assert np.array_equal(xt[:, 0, 0, 0], np.arange(10, dtype=np.float32))


def test_recorder_meta_describes_every_site(tmp_path):
    torch.manual_seed(0)
    config = capture.CaptureConfig(sites={"prefix_pool", "suffix_hidden"}, suffix_layers=(0, -1))
    with recorder.ActivationRecorder(_FakePolicy(), tmp_path, config=config, extra_meta={"ckpt": "pi05_libero"}) as rec:
        rec.infer({})

    meta = json.loads((tmp_path / "capture_meta.json").read_text(encoding="utf-8"))
    assert meta["num_rows"] == 1
    assert meta["num_shards"] == 1
    assert meta["ckpt"] == "pi05_libero"
    assert meta["capture_config"]["suffix_layers"] == [0, -1]
    assert meta["sites"]["suffix_hidden"]["axes"] == ["layer", "step", "token", "dim"]
    assert meta["sites"]["suffix_hidden"]["axis_ids"]["layer"] == [0, DEPTH - 1]
    assert set(meta["sites"]) == {"prefix_pool", "suffix_hidden"}


def test_recorder_raw_load_skips_decoding(tmp_path):
    torch.manual_seed(0)
    with recorder.ActivationRecorder(_FakePolicy(), tmp_path, config=capture.CaptureConfig.lean()) as rec:
        rec.infer({})
    raw = recorder.load_site(tmp_path, "prefix_pool", decode=False)
    decoded = recorder.load_site(tmp_path, "prefix_pool")
    assert raw.dtype == np.int16
    assert np.array_equal(codec.decode(raw, codec.BF16_RAW), decoded)


def test_recorder_disabled_is_a_pass_through(tmp_path):
    policy = _FakePolicy()
    rec = recorder.ActivationRecorder(policy, tmp_path / "unused", enabled=False)
    assert rec.infer({}, noise=1)["kwargs"] == ["noise"]  # kwargs still reach the policy
    rec.close()
    assert policy.calls == 1
    assert not (tmp_path / "unused").exists()  # nothing is even created


def test_recorder_forwards_kwargs(tmp_path):
    with recorder.ActivationRecorder(_FakePolicy(), tmp_path, config=capture.CaptureConfig.lean()) as rec:
        assert rec.infer({}, noise=np.zeros(3))["kwargs"] == ["noise"]


def test_recorder_rejects_a_changed_layout(tmp_path):
    class _ShrinkingPolicy(_FakePolicy):
        def infer(self, obs, **kwargs):
            # Second call reports one camera fewer, which would make the shards unstackable.
            cameras = CAMERAS if self.calls == 0 else CAMERAS - 1
            capture.set_action_horizon(HORIZON)
            capture.record_prefix_layout(IMAGE_SPANS[:cameras], LANG_SPAN, PAD_MASK)
            capture.record_prefix_hidden(tuple(torch.randn(1, PREFIX_LEN, PREFIX_DIM) for _ in range(DEPTH)))
            self.calls += 1
            return {}

    torch.manual_seed(0)
    with recorder.ActivationRecorder(
        _ShrinkingPolicy(), tmp_path, config=capture.CaptureConfig(sites={"prefix_pool"})
    ) as rec:
        rec.infer({})
        with pytest.raises(RuntimeError, match="layout changed mid-run"):
            rec.infer({})


def test_load_site_rejects_a_site_that_was_not_captured(tmp_path):
    torch.manual_seed(0)
    with recorder.ActivationRecorder(_FakePolicy(), tmp_path, config=capture.CaptureConfig.lean()) as rec:
        rec.infer({})
    with pytest.raises(ValueError, match="was not captured"):
        recorder.load_site(tmp_path, "prefix_full")
