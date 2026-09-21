"""Tests for probe capture and the recorder.

Everything runs on fabricated tensors, so no checkpoint and no GPU are needed. What is not
covered here is whether the hooks are in the right *places* in the model -- that only shows
up when a real policy runs. See docs/probe_capture.md.
"""

import json

import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from openpi.probe import capture
from openpi.probe import recorder

# Stand-in dimensions for a pi0.5 forward pass, small enough to keep the tests fast.
DEPTH = 5  # transformer depth 4, plus the post-final-norm entry
IMG_TOKENS, CAMERAS, PREFIX_DIM = 6, 3, 8
LANG_TOKENS, VALID_LANG = 4, 2
HORIZON, SUFFIX_LEN, SUFFIX_DIM = 4, 5, 7
PREFIX_LEN = CAMERAS * IMG_TOKENS + LANG_TOKENS
LANG_SPAN = (CAMERAS * IMG_TOKENS, PREFIX_LEN)
IMAGE_SPANS = [(i * IMG_TOKENS, (i + 1) * IMG_TOKENS) for i in range(CAMERAS)]
TOKEN_MASK = torch.tensor([[True] * VALID_LANG + [False] * (LANG_TOKENS - VALID_LANG)])


def _simulate_model(num_steps=3, tag=0.0):
    """Replays the call sequence the PyTorch model makes, with random tensors.

    `tag` is written into the first element of every xt_traj snapshot so a test can tell
    which call a stored step came from.
    """
    prefix_hidden = tuple(torch.randn(1, PREFIX_LEN, PREFIX_DIM) for _ in range(DEPTH))
    suffix_hidden = [tuple(torch.randn(1, SUFFIX_LEN, SUFFIX_DIM) for _ in range(DEPTH)) for _ in range(num_steps)]

    capture.set_action_horizon(HORIZON)
    capture.record_observation(
        torch.zeros(1, LANG_TOKENS, dtype=torch.int32),
        TOKEN_MASK,
        [torch.tensor([True]), torch.tensor([True]), torch.tensor([False])],
        torch.zeros(1, 3),
    )
    for _ in range(CAMERAS):
        capture.record("siglip_tokens", torch.randn(1, IMG_TOKENS, PREFIX_DIM))
    capture.record_prefix_layout(IMAGE_SPANS, LANG_SPAN)
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
        tensors, meta = cap.result()
    return tensors, meta, prefix_hidden, suffix_hidden


def _bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16)


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
    for site in ("prefix_image", "prefix_lang"):
        with capture.session(capture.CaptureConfig(sites={site})):
            assert capture.wants_prefix_hidden()
            assert not capture.wants_suffix_hidden()


def test_capture_shapes_dtypes_and_layer_step_filtering():
    config = capture.CaptureConfig(sites=capture.ALL_SITES, suffix_layers=(0, -1), denoise_steps=(0, 2))
    tensors, meta, _, _ = _run_capture(config)

    assert tensors["prefix_image"].shape == (1, DEPTH, CAMERAS, IMG_TOKENS, PREFIX_DIM)
    assert tensors["prefix_lang"].shape == (1, DEPTH, LANG_TOKENS, PREFIX_DIM)
    assert tensors["siglip_tokens"].shape == (1, CAMERAS, IMG_TOKENS, PREFIX_DIM)
    # 2 of 5 layers, 2 of 3 steps, and the suffix sliced to the last action_horizon tokens.
    assert tensors["suffix_hidden"].shape == (1, 2, 2, HORIZON, SUFFIX_DIM)
    assert tensors["xt_traj"].shape == (1, 4, HORIZON, 3)  # num_steps + 1 snapshots
    assert tensors["vt"].shape == (1, 3, HORIZON, 3)
    assert tensors["adarms_cond"].shape == (1, 3, SUFFIX_DIM)  # not step-filtered

    # Hidden states stay bfloat16; everything is on CPU.
    assert tensors["prefix_image"].dtype == torch.bfloat16
    assert tensors["xt_traj"].dtype == torch.float32
    assert tensors["image_mask"].dtype == torch.bool
    assert all(t.device.type == "cpu" for t in tensors.values())
    assert meta["prefix_image"]["dtype"] == "bfloat16"

    # Axis ids record which real layer / step / camera each stored position is, so a
    # partial capture stays interpretable.
    assert meta["suffix_hidden"]["axis_ids"] == {"layer": [0, DEPTH - 1], "step": [0, 2]}
    assert meta["prefix_image"]["axis_ids"] == {"layer": list(range(DEPTH)), "camera": list(range(CAMERAS))}
    assert meta["prefix_lang"]["axis_ids"] == {"layer": list(range(DEPTH))}
    assert meta["suffix_hidden"]["shape"] == [2, 2, HORIZON, SUFFIX_DIM]


def test_capture_prefix_sites_are_exact_unpooled_slices():
    tensors, _, prefix_hidden, suffix_hidden = _run_capture(capture.CaptureConfig.high())
    image = tensors["prefix_image"][0]  # [layer, camera, token, dim]
    lang = tensors["prefix_lang"][0]  # [layer, token, dim]

    for layer in range(DEPTH):
        hidden = _bf16(prefix_hidden[layer][0])
        for camera, (start, end) in enumerate(IMAGE_SPANS):
            assert torch.equal(image[layer, camera], hidden[start:end])
        assert torch.equal(lang[layer], hidden[LANG_SPAN[0] : LANG_SPAN[1]])
        # Image blocks followed by the language block give back the whole prefix.
        assert torch.equal(torch.cat([image[layer].flatten(0, 1), lang[layer]]), hidden)
    # Camera 2 is masked in _simulate_model and is still stored, so the shape stays fixed.
    assert image.shape[1] == CAMERAS

    # The suffix keeps the last action_horizon tokens, which is what feeds action_out_proj.
    assert torch.equal(tensors["suffix_hidden"][0, 2, 1], _bf16(suffix_hidden[1][2][0, -HORIZON:]))


def test_capture_rejects_double_recording_and_bad_layers():
    with capture.session(capture.CaptureConfig(sites={"prefix_image"})) as cap:
        cap.record_prefix_layout([(0, 2)], (2, 4))
        hidden = tuple(torch.randn(1, 4, 3) for _ in range(2))
        cap.record_prefix_hidden(hidden)
        with pytest.raises(RuntimeError, match="recorded twice"):
            cap.record_prefix_hidden(hidden)

    with capture.session(capture.CaptureConfig(sites={"suffix_hidden"}, suffix_layers=(99,))) as cap:
        cap.set_action_horizon(2)
        with pytest.raises(IndexError, match="out of range"):
            cap.record_suffix_hidden(tuple(torch.randn(1, 2, 3) for _ in range(2)))


def test_capture_config_keeps_explicit_sites():
    config = capture.CaptureConfig(sites={"prefix_image"})
    assert config.sites == {"prefix_image"}


def test_presets_are_low_mid_high_with_mid_as_default():
    low, mid, high = capture.CaptureConfig.low(), capture.CaptureConfig.mid(), capture.CaptureConfig.high()
    assert capture.CaptureConfig() == mid

    # low and mid capture the same sites; low keeps only the action expert's final layer.
    assert low.sites == mid.sites == capture.CORE_SITES
    assert low.suffix_layers == (-1,)
    assert mid.suffix_layers is None
    assert {"prefix_image", "suffix_hidden", "xt_traj", "vt"} <= mid.sites
    assert "prefix_lang" not in mid.sites

    # high adds the language tokens, the SigLIP output and the adaRMS conditioning.
    assert high.sites == capture.ALL_SITES
    assert high.sites - mid.sites == {"prefix_lang", "siglip_tokens", "adarms_cond"}
    # Every preset keeps every prefix layer and every denoise step.
    assert all(c.prefix_layers is None and c.denoise_steps is None for c in (low, mid, high))


def test_capture_rejects_nested_sessions():
    with capture.session(capture.CaptureConfig()):
        assert capture.active()
        with pytest.raises(RuntimeError, match="already active"), capture.session(capture.CaptureConfig()):
            pass
    assert not capture.active()


# =============================================================================
# Recorder
# =============================================================================


class _FakePolicy:
    """Stands in for a Policy: its `infer` makes the same capture calls the model would."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def infer(self, obs, **kwargs):
        if capture.active():
            _simulate_model(tag=float(self.calls))
        self.calls += 1
        return {"actions": np.zeros((HORIZON, 3), dtype=np.float32), "kwargs": sorted(kwargs)}

    def reset(self):
        self.resets += 1


def _record_episodes(out_dir, lengths, config=None):
    """Records one episode per entry in `lengths`, calling reset() before each like a rollout loop."""
    torch.manual_seed(0)
    policy = _FakePolicy()
    rec = recorder.ActivationRecorder(policy, out_dir, config=config or capture.CaptureConfig.high())
    for length in lengths:
        rec.reset()
        for _ in range(length):
            rec.infer({"prompt": "pick up the block"})
    return policy


def test_recorder_writes_one_file_per_call_grouped_by_episode(tmp_path):
    policy = _record_episodes(tmp_path, [3, 2])

    assert policy.calls == 5
    assert policy.resets == 2  # reset() is forwarded to the wrapped policy
    # The first reset() comes before any step, so it does not leave an empty episode 0.
    assert sorted(p.name for p in tmp_path.glob("ep_*")) == ["ep_00000", "ep_00001"]
    assert sorted(p.name for p in (tmp_path / "ep_00000").iterdir()) == [
        "t_00000.safetensors",
        "t_00001.safetensors",
        "t_00002.safetensors",
    ]
    assert len(list((tmp_path / "ep_00001").iterdir())) == 2


def test_recorder_files_load_with_plain_safetensors(tmp_path):
    _record_episodes(tmp_path, [1])
    step = load_file(str(tmp_path / "ep_00000" / "t_00000.safetensors"))

    assert set(step) == set(capture.ALL_SITES)
    # Batch axis dropped, bfloat16 kept as bfloat16.
    assert step["prefix_image"].shape == (DEPTH, CAMERAS, IMG_TOKENS, PREFIX_DIM)
    assert step["prefix_image"].dtype == torch.bfloat16
    assert step["image_mask"].tolist() == [True, True, False]


def test_load_episode_stacks_steps_in_call_order(tmp_path):
    _record_episodes(tmp_path, [3, 2])

    first = recorder.load_episode(tmp_path, 0)
    assert first["suffix_hidden"].shape == (3, DEPTH, 3, HORIZON, SUFFIX_DIM)
    assert first["prefix_image"].dtype == torch.bfloat16
    # The tag in xt_traj proves steps come back in the order they were recorded.
    assert first["xt_traj"][:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert recorder.load_episode(tmp_path, 1)["xt_traj"][:, 0, 0, 0].tolist() == [3.0, 4.0]


def test_load_episode_reads_only_the_requested_sites(tmp_path):
    _record_episodes(tmp_path, [2])

    episode = recorder.load_episode(tmp_path, 0, sites=["suffix_hidden", "image_mask"])
    assert set(episode) == {"suffix_hidden", "image_mask"}
    with pytest.raises(ValueError, match="were not captured"):
        recorder.load_episode(tmp_path, 0, sites=["no_such_site"])
    with pytest.raises(FileNotFoundError):
        recorder.load_episode(tmp_path, 7)


def test_recorder_meta_describes_every_site(tmp_path):
    torch.manual_seed(0)
    config = capture.CaptureConfig(sites={"prefix_image", "suffix_hidden"}, suffix_layers=(0, -1))
    rec = recorder.ActivationRecorder(_FakePolicy(), tmp_path, config=config, extra_meta={"checkpoint": "ckpt-1"})
    rec.infer({})

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta == recorder.load_meta(tmp_path)
    assert meta["checkpoint"] == "ckpt-1"
    assert meta["capture_config"]["suffix_layers"] == [0, -1]
    assert meta["sites"]["suffix_hidden"]["axes"] == ["layer", "step", "token", "dim"]
    assert meta["sites"]["suffix_hidden"]["axis_ids"]["layer"] == [0, DEPTH - 1]
    assert set(meta["sites"]) == {"prefix_image", "suffix_hidden"}


def test_recorder_refuses_a_directory_that_already_has_episodes(tmp_path):
    (tmp_path / "ep_00000").mkdir()
    with pytest.raises(FileExistsError, match="already holds episodes"):
        recorder.ActivationRecorder(_FakePolicy(), tmp_path)


def test_recorder_forwards_kwargs(tmp_path):
    rec = recorder.ActivationRecorder(_FakePolicy(), tmp_path, config=capture.CaptureConfig.low())
    assert rec.infer({}, noise=np.zeros(3))["kwargs"] == ["noise"]


def test_recorder_rejects_a_changed_layout(tmp_path):
    class _ShrinkingPolicy(_FakePolicy):
        def infer(self, obs, **kwargs):
            # Second call reports one camera fewer, which would make the episode unstackable.
            cameras = CAMERAS if self.calls == 0 else CAMERAS - 1
            capture.set_action_horizon(HORIZON)
            capture.record_prefix_layout(IMAGE_SPANS[:cameras], LANG_SPAN)
            capture.record_prefix_hidden(tuple(torch.randn(1, PREFIX_LEN, PREFIX_DIM) for _ in range(DEPTH)))
            self.calls += 1
            return {}

    torch.manual_seed(0)
    rec = recorder.ActivationRecorder(
        _ShrinkingPolicy(), tmp_path, config=capture.CaptureConfig(sites={"prefix_image"})
    )
    rec.infer({})
    with pytest.raises(RuntimeError, match="layout changed mid-run"):
        rec.infer({})
