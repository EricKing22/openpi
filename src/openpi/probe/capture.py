"""Model-side activation capture for pi0 / pi0.5.

Capture is off unless a `session` is open. The model calls into this module on every
inference, but each call is one `ContextVar` read when nothing is being captured, so a
normal run pays nothing.

    with capture.session(capture.CaptureConfig(sites={"prefix_pool"})) as cap:
        policy.infer(obs)
    arrays, site_meta = cap.result()

`openpi.probe.recorder.ActivationRecorder` is the normal entry point; open a session by
hand only when you want the arrays in memory.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses

import numpy as np
import torch

from openpi.probe import codec


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """描述一个可采集点的数据格式。

    每个 ``SiteSpec`` 定义一种中间张量的轴名称、落盘数据类型和重复采集轴。
    ``Capture`` 会根据这些信息把同一次推理中分多次产生的张量（例如不同相机
    或不同去噪步骤）拼接成形状稳定的数组。
    """

    axes: tuple[str, ...]  # per-sample axes, excluding the leading batch/row axis
    store_dtype: str
    # Sites recorded once per inference have repeat_axis=None. The rest are recorded
    # repeatedly (once per denoise step, or once per camera) and the pieces are stacked
    # into this axis at the end.
    repeat_axis: str | None
    doc: str


# The capture menu. `layer` is axis 0 everywhere on purpose: it makes "give me layer 17"
# the outermost slice, which is the one selection that can be served without touching the
# rest of a row -- whether that is a numpy fancy-index or a later memmap reader.
SITES: dict[str, SiteSpec] = {
    "prefix_pool": SiteSpec(
        ("layer", "pool", "dim"),
        codec.BF16_RAW,
        None,
        "Pooled prefix hidden states: per-camera mean, language mean, last prompt token.",
    ),
    "prefix_full": SiteSpec(
        ("layer", "token", "dim"),
        codec.BF16_RAW,
        None,
        "Every prefix token position. 75 MB per row for pi0.5; pilot runs only.",
    ),
    "suffix_hidden": SiteSpec(
        ("layer", "step", "token", "dim"),
        codec.BF16_RAW,
        "step",
        "Action expert hidden states, last action_horizon tokens, one entry per denoise step.",
    ),
    "siglip_tokens": SiteSpec(
        ("camera", "token", "dim"),
        codec.BF16_RAW,
        "camera",
        "Image tokens before the LLM. The vision-only baseline.",
    ),
    "xt_traj": SiteSpec(
        ("step", "token", "dim"),
        codec.FLOAT32,
        "step",
        "Flow-matching trajectory, num_steps + 1 snapshots. [0] is the noise, [-1] the action.",
    ),
    "vt": SiteSpec(("step", "token", "dim"), codec.FLOAT32, "step", "Predicted velocity at each denoise step."),
    "adarms_cond": SiteSpec(
        ("step", "dim"),
        codec.FLOAT32,
        "step",
        "adaRMS timestep conditioning. A function of t alone, so it is a control variable.",
    ),
    # Alignment sites. Tiny, and without them a prefix slice cannot be interpreted:
    # you would not know which positions are images, language, or padding.
    "tokens": SiteSpec(("token",), codec.INT32, None, "Tokenized prompt (task + discretized state for pi0.5)."),
    "token_mask": SiteSpec(("token",), codec.BOOL, None, "Valid-token mask for the prompt."),
    "image_mask": SiteSpec(("camera",), codec.BOOL, None, "Per-camera validity; masked cameras are padding."),
    "state": SiteSpec(("dim",), codec.FLOAT32, None, "Normalized robot state fed to the model."),
}

ALIGNMENT_SITES = frozenset({"tokens", "token_mask", "image_mask", "state"})
DEFAULT_SITES = frozenset({"prefix_pool", "suffix_hidden", "xt_traj", "vt"}) | ALIGNMENT_SITES
PILOT_SITES = frozenset(SITES)


@dataclasses.dataclass
class CaptureConfig:
    """一次推理的激活采集配置；层或步骤为 ``None`` 时表示全部保留。"""

    sites: set[str] | frozenset[str] = DEFAULT_SITES
    prefix_layers: tuple[int, ...] | None = None
    suffix_layers: tuple[int, ...] | None = None
    denoise_steps: tuple[int, ...] | None = None

    @classmethod
    def pilot(cls) -> CaptureConfig:
        """采集全部站点、层和步骤；pi0.5 每次推理约 98 MB。"""
        return cls(sites=PILOT_SITES)

    @classmethod
    def lean(cls) -> CaptureConfig:
        """采集 pooled prefix 和动作专家最后一层；每次推理约 1.5 MB。"""
        return cls(sites=DEFAULT_SITES, suffix_layers=(-1,))


class Capture:
    """收集并整理单次推理产生的全部激活。

    该对象由 ``session`` 创建，并通过模块级 ``ContextVar`` 被模型钩子访问。
    推理过程中它暂存各采集点的张量、prefix 布局和真实层/步骤编号；调用
    ``result`` 时再把重复记录沿声明的轴堆叠，并返回数组及其元数据。
    """

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._layout: tuple[tuple[tuple[int, int], ...], tuple[int, int], torch.Tensor] | None = None
        self._denoise_step = 0
        self._action_horizon: int | None = None
        # Which layer / step / pool each stored position corresponds to. Recorded so that a
        # partial capture (say prefix_layers=(0, 8, 17)) stays selectable by true layer id.
        self._ids: dict[str, dict[str, list]] = {}

    def wants(self, site: str) -> bool:
        return site in self.config.sites

    # -- model-side recording ---------------------------------------------------

    def record(self, site: str, tensor: torch.Tensor) -> None:
        """Stores one tensor for a site that needs no special handling."""
        if not self.wants(site):
            return
        spec = SITES[site]
        self._buffers.setdefault(site, []).append(_to_numpy(tensor, spec.store_dtype))
        if spec.repeat_axis == "camera":
            self._ids.setdefault(site, {})["camera"] = list(range(len(self._buffers[site])))

    def set_action_horizon(self, action_horizon: int) -> None:
        self._action_horizon = int(action_horizon)

    def set_denoise_step(self, step: int) -> None:
        self._denoise_step = int(step)

    def record_prefix_layout(self, image_spans, lang_span, pad_mask: torch.Tensor) -> None:
        """Tells the pooler where the image and language blocks sit in the prefix."""
        self._layout = (tuple(tuple(s) for s in image_spans), tuple(lang_span), pad_mask)

    def record_prefix_hidden(self, hidden_states) -> None:
        """hidden_states: the HF tuple of depth+1 tensors, each [B, prefix_len, width]."""
        if self._layout is None:
            raise RuntimeError("record_prefix_layout must be called before record_prefix_hidden")
        selected, ids = _select_layers(hidden_states, self.config.prefix_layers)
        num_cameras = len(self._layout[0])
        pool_names = [*(f"img{i}" for i in range(num_cameras)), "lang_mean", "lang_last"]
        for site in ("prefix_pool", "prefix_full"):
            if self.wants(site):
                self._ids[site] = {"layer": ids}
        if self.wants("prefix_pool"):
            self._ids["prefix_pool"]["pool"] = pool_names
            self._store_once("prefix_pool", torch.stack([_pool_prefix(h, self._layout) for h in selected], dim=1))
        if self.wants("prefix_full"):
            self._store_once("prefix_full", torch.stack(selected, dim=1))

    def record_suffix_hidden(self, hidden_states) -> None:
        """hidden_states: the HF tuple of depth+1 tensors, each [B, suffix_len, width]."""
        if not self.wants("suffix_hidden"):
            return
        if self.config.denoise_steps is not None and self._denoise_step not in self.config.denoise_steps:
            return
        selected, ids = _select_layers(hidden_states, self.config.suffix_layers)
        if self._action_horizon is not None:
            # pi0 prepends a state token to the suffix, pi0.5 does not. Keeping the last
            # action_horizon tokens makes both models produce the same shape, and those are
            # exactly the tokens that feed action_out_proj.
            selected = [h[:, -self._action_horizon :] for h in selected]
        entry = self._ids.setdefault("suffix_hidden", {"layer": ids, "step": []})
        entry["step"].append(self._denoise_step)
        self._buffers.setdefault("suffix_hidden", []).append(_to_numpy(torch.stack(selected, dim=1), codec.BF16_RAW))

    def record_observation(self, tokens, token_mask, image_masks, state) -> None:
        """Stores the four alignment sites in one go."""
        if isinstance(image_masks, (list, tuple)):
            image_masks = torch.stack([m.reshape(-1) for m in image_masks], dim=-1)
        self._store_once("tokens", tokens)
        self._store_once("token_mask", token_mask)
        self._store_once("image_mask", image_masks.reshape(image_masks.shape[0], -1))
        self._store_once("state", state)

    # -- results ----------------------------------------------------------------

    def result(self) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Returns (arrays, site metadata), both keyed by site.

        Arrays keep the leading batch axis. Repeated recordings (one per denoise step, or
        one per camera) are stacked into the site's `repeat_axis` here, so a site's shape
        does not depend on how many times `record` happened to be called.
        """
        arrays, meta = {}, {}
        for site, chunks in self._buffers.items():
            spec = SITES[site]
            if spec.repeat_axis is None:
                array = chunks[0]
            else:
                array = np.stack(chunks, axis=spec.axes.index(spec.repeat_axis) + 1)  # +1 for batch
            arrays[site] = array
            meta[site] = {
                "axes": list(spec.axes),
                "shape": list(array.shape[1:]),
                "store_dtype": spec.store_dtype,
                "axis_ids": self._ids.get(site, {}),
                "doc": spec.doc,
            }
        return arrays, meta

    # -- internals --------------------------------------------------------------

    def _store_once(self, site: str, tensor: torch.Tensor) -> None:
        if not self.wants(site):
            return
        if site in self._buffers:
            raise RuntimeError(f"Site {site!r} is recorded once per inference but was recorded twice")
        self._buffers[site] = [_to_numpy(tensor, SITES[site].store_dtype)]


def _select_layers(hidden_states, layers: tuple[int, ...] | None):
    """Picks layers out of an HF `hidden_states` tuple, returning (tensors, resolved ids).

    The tuple has depth+1 entries: index i < depth is the *input* to layer i, and the last
    entry is the output after the final norm. Negative indices work as usual, so -1 is the
    post-final-norm output.
    """
    n = len(hidden_states)
    if layers is None:
        return list(hidden_states), list(range(n))
    ids = []
    for layer in layers:
        resolved = layer + n if layer < 0 else layer
        if not 0 <= resolved < n:
            raise IndexError(f"Layer {layer} out of range for {n} hidden states (0..{n - 1}, or negative)")
        ids.append(resolved)
    return [hidden_states[i] for i in ids], ids


def _pool_prefix(hidden: torch.Tensor, layout) -> torch.Tensor:
    """[B, prefix_len, D] -> [B, num_cameras + 2, D].

    The prefix uses full bidirectional attention, so unlike a causal LM there is no single
    token that summarizes it. These five vectors are the useful stand-ins: one mean per
    camera, the mean over valid language tokens, and the last valid prompt position (the
    "Action: " token for pi0.5).
    """
    image_spans, (lang_start, lang_end), pad_mask = layout
    # Pool in float32: averaging a few hundred bfloat16 values accumulates real error.
    h = hidden.float()
    pooled = [h[:, start:end].mean(dim=1) for start, end in image_spans]

    lang = h[:, lang_start:lang_end]
    mask = pad_mask[:, lang_start:lang_end].to(h.dtype).unsqueeze(-1)
    pooled.append((lang * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0))

    last_valid = (pad_mask[:, lang_start:lang_end].sum(dim=1).long() - 1).clamp(min=0)
    pooled.append(lang[torch.arange(lang.shape[0], device=lang.device), last_valid])

    return torch.stack(pooled, dim=1).to(hidden.dtype)


def _to_numpy(tensor: torch.Tensor, store_dtype: str) -> np.ndarray:
    """Moves a tensor to CPU in its store dtype. bfloat16 goes across as raw int16 bits."""
    tensor = tensor.detach()
    if store_dtype == codec.BF16_RAW:
        return tensor.to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy()
    torch_dtype = {codec.FLOAT32: torch.float32, codec.INT32: torch.int32, codec.BOOL: torch.bool}[store_dtype]
    return tensor.to(torch_dtype).contiguous().cpu().numpy()


# -- ambient session ------------------------------------------------------------
# The model does not take a capture object as an argument; it reads this ContextVar. That
# keeps the capture hooks to one line each and costs a single lookup when capture is off.

_ACTIVE: contextvars.ContextVar[Capture | None] = contextvars.ContextVar("openpi_probe_capture", default=None)


@contextlib.contextmanager
def session(config: CaptureConfig | None = None):
    """Enables capture for the duration of the block. Deliberately not reentrant."""
    if _ACTIVE.get() is not None:
        raise RuntimeError("A probe capture session is already active")
    cap = Capture(config or CaptureConfig())
    token = _ACTIVE.set(cap)
    try:
        yield cap
    finally:
        _ACTIVE.reset(token)


def active() -> bool:
    return _ACTIVE.get() is not None


def wants_prefix_hidden() -> bool:
    """Whether the prefix forward should be asked for output_hidden_states."""
    cap = _ACTIVE.get()
    return cap is not None and (cap.wants("prefix_pool") or cap.wants("prefix_full"))


def wants_suffix_hidden() -> bool:
    """Whether the action expert forward should be asked for output_hidden_states."""
    cap = _ACTIVE.get()
    return cap is not None and cap.wants("suffix_hidden")


def record(site: str, tensor: torch.Tensor) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record(site, tensor)


def set_action_horizon(action_horizon: int) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.set_action_horizon(action_horizon)


def set_denoise_step(step: int) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.set_denoise_step(step)


def record_prefix_layout(image_spans, lang_span, pad_mask) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_prefix_layout(image_spans, lang_span, pad_mask)


def record_prefix_hidden(hidden_states) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_prefix_hidden(hidden_states)


def record_suffix_hidden(hidden_states) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_suffix_hidden(hidden_states)


def record_observation(tokens, token_mask, image_masks, state) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_observation(tokens, token_mask, image_masks, state)
