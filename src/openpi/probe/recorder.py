"""Writes captured activations to disk: one safetensors file per inference call.

    rec = recorder.ActivationRecorder(policy, "probe_data/run1")
    for episode in recorded_episodes:
        rec.reset()                          # the next call starts a new ep_ folder
        for obs in episode:
            rec.infer(obs)

    probe_data/run1/
      meta.json                              # what each key is: axes, shape, dtype, layer ids
      ep_00000/t_00000.safetensors           # episode 0, inference call 0
      ep_00000/t_00001.safetensors
      ep_00001/t_00000.safetensors
      ...

Each file is a flat dict {site: tensor} with the batch axis dropped and bfloat16 kept as
bfloat16, so reading one back is a single call:

    from safetensors.torch import load_file
    step = load_file("probe_data/run1/ep_00003/t_00017.safetensors")
    step["prefix_image"]                     # torch.bfloat16 [19, 3, 256, 2048]

`load_episode` stacks a whole episode along a new leading time axis.
"""

from __future__ import annotations

from collections.abc import Iterable
import contextlib
import json
import logging
import pathlib
from typing import Any

from openpi_client import base_policy as _base_policy
import safetensors
import safetensors.torch
import torch

from openpi.probe import capture as _capture

logger = logging.getLogger("openpi")

META_FILENAME = "meta.json"


def episode_dir(out_dir: str | pathlib.Path, episode: int) -> pathlib.Path:
    """Folder holding one episode's files."""
    return pathlib.Path(out_dir) / f"ep_{episode:05d}"


class ActivationRecorder(_base_policy.BasePolicy):
    """Runs each `infer` under a capture session and writes that call to its own file."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        out_dir: str | pathlib.Path,
        *,
        config: _capture.CaptureConfig | None = None,
        extra_meta: dict[str, Any] | None = None,
    ):
        """
        Args:
            policy: Policy to wrap.
            out_dir: Run directory. It must not already hold episodes, so two runs never
                mix in one folder.
            config: What to capture. Defaults to the mid preset, `CaptureConfig.mid()`.
            extra_meta: Written verbatim into meta.json (checkpoint id, git sha...).
        """
        self._policy = policy
        self._config = config or _capture.CaptureConfig.mid()
        self._extra_meta = dict(extra_meta or {})

        self._out_dir = pathlib.Path(out_dir)
        self._episode = 0
        self._step = 0
        self._site_meta: dict[str, dict] | None = None

        if any(self._out_dir.glob("ep_*")):
            raise FileExistsError(f"{self._out_dir} already holds episodes; record each run into a new directory")
        self._out_dir.mkdir(parents=True, exist_ok=True)
        if _disable_torch_compile(policy):
            logger.warning(
                "Probe capture: unwrapped torch.compile on sample_actions, because capture works by "
                "Python-level side effects inside the sampling loop that dynamo would trace away. "
                "Set Pi0Config.pytorch_compile_mode=None to skip the wasted compile."
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", {})

    def reset(self) -> None:
        """Starts a new episode and resets the wrapped policy.

        A reset before the current episode has any steps does not advance the episode, so
        calling this at the start of every episode never leaves an empty folder.
        """
        self._policy.reset()
        if self._step > 0:
            self._episode += 1
            self._step = 0

    def infer(self, obs: dict, **kwargs: Any) -> dict:  # type: ignore[override]
        """Runs the wrapped policy and writes the activations from that call to one file.

        Extra keyword arguments are forwarded to the policy, so `noise=` still works if you
        want to control the flow-matching noise yourself.
        """
        with _capture.session(self._config) as cap:
            outputs = self._policy.infer(obs, **kwargs)
            tensors, site_meta = cap.result()

        if self._site_meta is None:
            self._site_meta = site_meta
            self._write_meta()
        elif site_meta != self._site_meta:
            # Shapes must stay constant or an episode cannot be stacked. This catches a
            # changed camera count or prompt length rather than letting it corrupt the run.
            raise RuntimeError(f"Capture layout changed mid-run: expected {self._site_meta}, got {site_meta}")

        path = episode_dir(self._out_dir, self._episode) / f"t_{self._step:05d}.safetensors"
        path.parent.mkdir(exist_ok=True)
        # Policy.infer always works on a batch of one (it adds that axis itself), so drop it.
        safetensors.torch.save_file({site: tensor[0] for site, tensor in tensors.items()}, str(path))
        self._step += 1
        return outputs

    def _write_meta(self) -> None:
        meta = {
            "format_version": 3,
            "layout": "ep_XXXXX/t_XXXXX.safetensors: one file per inference call, batch axis dropped",
            "capture_config": {
                "sites": sorted(self._config.sites),
                "prefix_layers": self._config.prefix_layers,
                "suffix_layers": self._config.suffix_layers,
                "denoise_steps": self._config.denoise_steps,
            },
            "sites": self._site_meta,
            **self._extra_meta,
        }
        (self._out_dir / META_FILENAME).write_text(
            json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )


def load_meta(out_dir: str | pathlib.Path) -> dict:
    """Reads meta.json."""
    return json.loads((pathlib.Path(out_dir) / META_FILENAME).read_text(encoding="utf-8"))


def load_episode(
    out_dir: str | pathlib.Path, episode: int, sites: Iterable[str] | None = None
) -> dict[str, torch.Tensor]:
    """Loads one episode as {site: tensor[T, ...]}, steps stacked in call order.

    bfloat16 sites stay bfloat16. `sites` limits which keys are read, so loading only the
    small sites never touches the large ones on disk.
    """
    files = sorted(episode_dir(out_dir, episode).glob("t_*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No steps found for episode {episode} in {out_dir}")

    steps = []
    for path in files:
        with safetensors.safe_open(str(path), framework="pt") as f:
            available = set(f.keys())
            wanted = sorted(available) if sites is None else list(sites)
            if missing := set(wanted) - available:
                raise ValueError(f"Site(s) {sorted(missing)} were not captured; available: {sorted(available)}")
            steps.append({site: f.get_tensor(site) for site in wanted})
    return {site: torch.stack([step[site] for step in steps]) for site in steps[0]}


def _disable_torch_compile(policy) -> bool:
    """Unwraps a torch.compile'd `sample_actions`, returning whether anything changed.

    Dynamo would either trace away the capture side effects or graph-break on every one of
    them, so a compiled sampling loop silently records nothing.
    """
    model = getattr(policy, "_model", None)
    compiled = getattr(model, "sample_actions", None)
    original = getattr(compiled, "_torchdynamo_orig_callable", None)
    if original is None:
        return False
    with contextlib.suppress(AttributeError):
        model.sample_actions = original
    if getattr(policy, "_sample_actions", None) is compiled:
        policy._sample_actions = original  # noqa: SLF001
    return True
