"""Writes captured activations to disk.

    with recorder.ActivationRecorder(policy, "probe_data/raw", config=capture.CaptureConfig.pilot()) as rec:
        for obs in observations:
            rec.infer(obs)

    out_dir/
      capture_meta.json     # per-site axes / shapes / dtypes / axis ids, and the row count
      shard_000000.npz      # stacked arrays, first axis is the row
      shard_000001.npz
      ...

Rows are in call order, and shards are in name order, so row i of the whole run is row
`i % flush_every` of shard `i // flush_every`. Nothing about episodes, timesteps or
rollout structure is recorded: this only caches the vectors.

Read it back with `load_site`:

    hidden = load_site("probe_data/raw", "suffix_hidden")   # [N, layer, step, token, dim]
"""

from __future__ import annotations

import contextlib
import json
import logging
import pathlib
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy

from openpi.probe import capture as _capture
from openpi.probe import codec

logger = logging.getLogger("openpi")

META_FILENAME = "capture_meta.json"


class ActivationRecorder(_base_policy.BasePolicy):
    """Runs each `infer` under a capture session and buffers the result into shards."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        out_dir: str | pathlib.Path,
        *,
        config: _capture.CaptureConfig | None = None,
        enabled: bool = True,
        flush_every: int = 32,
        extra_meta: dict[str, Any] | None = None,
    ):
        """
        Args:
            policy: Policy to wrap. When disabled, `infer` is forwarded unchanged.
            out_dir: Where the shards and capture_meta.json go.
            config: What to capture. Defaults to `CaptureConfig()`.
            enabled: Master switch. False makes this a transparent pass-through, so it is
                safe to leave wired into an eval script permanently.
            flush_every: Rows per shard. Peak RAM is about this times the per-row size, and
                a crash loses at most this many rows.
            extra_meta: Written verbatim into capture_meta.json (checkpoint id, git sha...).
        """
        self._policy = policy
        self._enabled = enabled
        self._config = config or _capture.CaptureConfig()
        self._flush_every = int(flush_every)
        self._extra_meta = dict(extra_meta or {})

        self._out_dir = pathlib.Path(out_dir)
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._buffered = 0
        self._num_rows = 0
        self._shard_idx = 0
        self._site_meta: dict[str, dict] | None = None

        if not enabled:
            return

        self._out_dir.mkdir(parents=True, exist_ok=True)
        if _disable_torch_compile(policy):
            logger.warning(
                "Probe capture: unwrapped torch.compile on sample_actions, because capture works by "
                "Python-level side effects inside the sampling loop that dynamo would trace away. "
                "Set Pi0Config.pytorch_compile_mode=None to skip the wasted compile."
            )

    @property
    def num_rows(self) -> int:
        """Inference calls recorded so far, flushed or not."""
        return self._num_rows

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", {})

    def infer(self, obs: dict, **kwargs: Any) -> dict:  # type: ignore[override]
        """Runs the wrapped policy and caches the activations from that call.

        Extra keyword arguments are forwarded to the policy, so `noise=` still works if you
        want to control the flow-matching noise yourself.
        """
        if not self._enabled:
            return self._policy.infer(obs, **kwargs)

        with _capture.session(self._config) as cap:
            outputs = self._policy.infer(obs, **kwargs)
            arrays, site_meta = cap.result()

        if self._site_meta is None:
            self._site_meta = site_meta
            self._write_meta()
        elif site_meta != self._site_meta:
            # Shapes must stay constant or the shards cannot be concatenated. This catches a
            # changed camera count or prompt length rather than letting it corrupt the run.
            raise RuntimeError(f"Capture layout changed mid-run: expected {self._site_meta}, got {site_meta}")

        # Policy.infer always works on a batch of one (it adds that axis itself), so drop it.
        for site, array in arrays.items():
            self._buffers.setdefault(site, []).append(array[0])
        self._buffered += 1
        self._num_rows += 1

        if self._buffered >= self._flush_every:
            self.flush()
        return outputs

    def flush(self) -> None:
        """Writes the buffered rows as one shard."""
        if not self._buffered:
            return
        path = self._out_dir / f"shard_{self._shard_idx:06d}.npz"
        # Uncompressed on purpose: activations barely compress, so savez_compressed would
        # only burn CPU.
        np.savez(path, **{site: np.stack(chunks, axis=0) for site, chunks in self._buffers.items()})
        logger.info("Probe capture: wrote %s (%d rows)", path.name, self._buffered)
        self._shard_idx += 1
        self._buffered = 0
        self._buffers = {}
        self._write_meta()

    def close(self) -> None:
        if self._enabled:
            self.flush()

    def __enter__(self) -> ActivationRecorder:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _write_meta(self) -> None:
        """Rewritten after every flush so the row count on disk stays current."""
        meta = {
            "format_version": 2,
            "num_rows": self._num_rows - self._buffered,  # only what is actually on disk
            "num_shards": self._shard_idx,
            "flush_every": self._flush_every,
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
            json.dumps(meta, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8"
        )


def load_meta(out_dir: str | pathlib.Path) -> dict:
    """Reads capture_meta.json."""
    return json.loads((pathlib.Path(out_dir) / META_FILENAME).read_text(encoding="utf-8"))


def load_site(out_dir: str | pathlib.Path, site: str, *, decode: bool = True) -> np.ndarray:
    """Concatenates one site across every shard, in call order.

    With `decode=True` bfloat16 sites come back as float32; with False you get the raw
    int16 bit patterns, which is half the memory if you only need to move them around.
    """
    out_dir = pathlib.Path(out_dir)
    meta = load_meta(out_dir)
    if site not in meta["sites"]:
        raise ValueError(f"Site {site!r} was not captured; available: {sorted(meta['sites'])}")

    blocks = []
    for path in sorted(out_dir.glob("shard_*.npz")):
        with np.load(path) as npz:
            blocks.append(npz[site])
    if not blocks:
        raise FileNotFoundError(f"No shards found in {out_dir}")
    array = np.concatenate(blocks, axis=0)
    return codec.decode(array, meta["sites"][site]["store_dtype"]) if decode else array


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


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
