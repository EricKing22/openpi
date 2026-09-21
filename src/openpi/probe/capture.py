"""Model-side activation capture for pi0 / pi0.5.

Capture is off unless a `session` is open. The model calls into this module on every
inference, but each call is one `ContextVar` read when nothing is being captured, so a
normal run pays nothing.

    with capture.session(capture.CaptureConfig(sites={"prefix_image"})) as cap:
        policy.infer(obs)
    tensors, site_meta = cap.result()

`openpi.probe.recorder.ActivationRecorder` is the normal entry point; open a session by
hand only when you want the tensors in memory.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """描述一个可采集点的数据格式。

    每个 ``SiteSpec`` 定义一种中间张量的轴名称、落盘数据类型和重复采集轴。
    ``Capture`` 会根据这些信息把同一次推理中分多次产生的张量（例如不同相机
    或不同去噪步骤）拼接成形状稳定的张量。
    """

    axes: tuple[str, ...]  # per-sample axes, excluding the leading batch/row axis
    dtype: torch.dtype
    # Sites recorded once per inference have repeat_axis=None. The rest are recorded
    # repeatedly (once per denoise step, or once per camera) and the pieces are stacked
    # into this axis at the end.
    repeat_axis: str | None
    doc: str


# The capture menu. `layer` is axis 0 everywhere on purpose: it makes "give me layer 17"
# the outermost slice. Nothing is pooled -- every site is a tensor as the model produced
# it, or a plain slice of one.
SITES: dict[str, SiteSpec] = {
    "prefix_image": SiteSpec(
        ("layer", "camera", "token", "dim"),
        torch.bfloat16,
        None,
        "Image-token hidden states at every prefix layer, one block per camera.",
    ),
    "prefix_lang": SiteSpec(
        ("layer", "token", "dim"),
        torch.bfloat16,
        None,
        "Language-token hidden states at every prefix layer: the prompt (plus discretized state "
        "for pi0.5), padded to max_token_len. prefix_image + prefix_lang is the whole prefix.",
    ),
    "suffix_hidden": SiteSpec(
        ("layer", "step", "token", "dim"),
        torch.bfloat16,
        "step",
        "Action expert hidden states, last action_horizon tokens, one entry per denoise step.",
    ),
    "siglip_tokens": SiteSpec(
        ("camera", "token", "dim"),
        torch.bfloat16,
        "camera",
        "Image tokens before the LLM. The vision-only baseline.",
    ),
    "xt_traj": SiteSpec(
        ("step", "token", "dim"),
        torch.float32,
        "step",
        "Flow-matching trajectory, num_steps + 1 snapshots. [0] is the noise, [-1] the action.",
    ),
    "vt": SiteSpec(("step", "token", "dim"), torch.float32, "step", "Predicted velocity at each denoise step."),
    "adarms_cond": SiteSpec(
        ("step", "dim"),
        torch.float32,
        "step",
        "adaRMS timestep conditioning. A function of t alone, so it is a control variable.",
    ),
    # Alignment sites. Tiny, and without them a prefix slice cannot be interpreted:
    # you would not know which cameras are real or which prompt positions are padding.
    "tokens": SiteSpec(("token",), torch.int32, None, "Tokenized prompt (task + discretized state for pi0.5)."),
    "token_mask": SiteSpec(("token",), torch.bool, None, "Valid-token mask for the prompt."),
    "image_mask": SiteSpec(("camera",), torch.bool, None, "Per-camera validity; masked cameras are padding."),
    "state": SiteSpec(("dim",), torch.float32, None, "Normalized robot state fed to the model."),
}

ALIGNMENT_SITES = frozenset({"tokens", "token_mask", "image_mask", "state"})
# Image tokens, action expert, flow trajectory, alignment: what low and mid capture.
CORE_SITES = frozenset({"prefix_image", "suffix_hidden", "xt_traj", "vt"}) | ALIGNMENT_SITES
ALL_SITES = frozenset(SITES)


@dataclasses.dataclass
class CaptureConfig:
    """一次推理的激活采集配置；层或步骤为 ``None`` 时表示全部保留。

    三档预设 ``low`` / ``mid`` / ``high``。直接构造 ``CaptureConfig()`` 即为 ``mid``。
    下面的大小按 pi0.5、3 个相机槽、``action_horizon=50`` 计算。
    """

    sites: set[str] | frozenset[str] = CORE_SITES
    prefix_layers: tuple[int, ...] | None = None
    suffix_layers: tuple[int, ...] | None = None
    denoise_steps: tuple[int, ...] | None = None

    @classmethod
    def low(cls) -> CaptureConfig:
        """每层图像 token + 动作专家最后一层 + 去噪轨迹 + 对齐信息。约 61 MB/次。"""
        return cls(sites=CORE_SITES, suffix_layers=(-1,))

    @classmethod
    def mid(cls) -> CaptureConfig:
        """默认档。每层图像 token + 动作专家全部层 + 去噪轨迹 + 对齐信息。约 79 MB/次。"""
        return cls(sites=CORE_SITES)

    @classmethod
    def high(cls) -> CaptureConfig:
        """全部站点。在 mid 之上再加每层语言 token、SigLIP 输出和 adaRMS 条件。约 98 MB/次。"""
        return cls(sites=ALL_SITES)


class Capture:
    """收集并整理单次推理产生的全部激活。

    该对象由 ``session`` 创建，并通过模块级 ``ContextVar`` 被模型钩子访问。
    推理过程中它暂存各采集点的张量、prefix 布局和真实层/步骤编号；调用
    ``result`` 时再把重复记录沿声明的轴堆叠，并返回张量及其元数据。
    """

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._buffers: dict[str, list[torch.Tensor]] = {}
        self._layout: tuple[tuple[tuple[int, int], ...], tuple[int, int]] | None = None
        self._denoise_step = 0
        self._action_horizon: int | None = None
        # Which layer / step / camera each stored position corresponds to. Recorded so that a
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
        self._buffers.setdefault(site, []).append(_to_cpu(tensor, spec.dtype))
        if spec.repeat_axis == "camera":
            self._ids.setdefault(site, {})["camera"] = list(range(len(self._buffers[site])))

    def set_action_horizon(self, action_horizon: int) -> None:
        self._action_horizon = int(action_horizon)

    def set_denoise_step(self, step: int) -> None:
        self._denoise_step = int(step)

    def record_prefix_layout(self, image_spans, lang_span) -> None:
        """Tells the capture where each camera's image block and the language block sit in the prefix."""
        self._layout = (tuple(tuple(s) for s in image_spans), tuple(lang_span))

    def record_prefix_hidden(self, hidden_states) -> None:
        """hidden_states: the HF tuple of depth+1 tensors, each [B, prefix_len, width]."""
        if self._layout is None:
            raise RuntimeError("record_prefix_layout must be called before record_prefix_hidden")
        selected, ids = _select_layers(hidden_states, self.config.prefix_layers)
        image_spans, (lang_start, lang_end) = self._layout
        if self.wants("prefix_image"):
            self._ids["prefix_image"] = {"layer": ids, "camera": list(range(len(image_spans)))}
            self._store_once("prefix_image", torch.stack([_image_tokens(h, image_spans) for h in selected], dim=1))
        if self.wants("prefix_lang"):
            self._ids["prefix_lang"] = {"layer": ids}
            self._store_once("prefix_lang", torch.stack([h[:, lang_start:lang_end] for h in selected], dim=1))

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
        self._buffers.setdefault("suffix_hidden", []).append(_to_cpu(torch.stack(selected, dim=1), torch.bfloat16))

    def record_observation(self, tokens, token_mask, image_masks, state) -> None:
        """Stores the four alignment sites in one go."""
        if isinstance(image_masks, (list, tuple)):
            image_masks = torch.stack([m.reshape(-1) for m in image_masks], dim=-1)
        self._store_once("tokens", tokens)
        self._store_once("token_mask", token_mask)
        self._store_once("image_mask", image_masks.reshape(image_masks.shape[0], -1))
        self._store_once("state", state)

    # -- results ----------------------------------------------------------------

    def result(self) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
        """Returns (tensors, site metadata), both keyed by site.

        Tensors are on CPU and keep the leading batch axis. Repeated recordings (one per
        denoise step, or one per camera) are stacked into the site's `repeat_axis` here, so
        a site's shape does not depend on how many times `record` happened to be called.
        """
        tensors, meta = {}, {}
        for site, chunks in self._buffers.items():
            spec = SITES[site]
            if spec.repeat_axis is None:
                tensor = chunks[0]
            else:
                tensor = torch.stack(chunks, dim=spec.axes.index(spec.repeat_axis) + 1)  # +1 for batch
            tensors[site] = tensor
            meta[site] = {
                "axes": list(spec.axes),
                "shape": list(tensor.shape[1:]),
                "dtype": str(spec.dtype).removeprefix("torch."),
                "axis_ids": self._ids.get(site, {}),
                "doc": spec.doc,
            }
        return tensors, meta

    # -- internals --------------------------------------------------------------

    def _store_once(self, site: str, tensor: torch.Tensor) -> None:
        if not self.wants(site):
            return
        if site in self._buffers:
            raise RuntimeError(f"Site {site!r} is recorded once per inference but was recorded twice")
        self._buffers[site] = [_to_cpu(tensor, SITES[site].dtype)]


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


def _image_tokens(hidden: torch.Tensor, image_spans) -> torch.Tensor:
    """[B, prefix_len, D] -> [B, num_cameras, tokens_per_camera, D].

    The image positions of the prefix, split per camera and otherwise untouched. Masked
    cameras stay in so the shape does not depend on which cameras were valid; the
    `image_mask` site says which ones were.
    """
    return torch.stack([hidden[:, start:end] for start, end in image_spans], dim=1)


def _to_cpu(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Detaches a tensor and moves it to CPU in its stored dtype. bfloat16 stays bfloat16."""
    return tensor.detach().to(dtype).contiguous().cpu()


# -- ambient session ------------------------------------------------------------
# The model does not take a capture object as an argument; it reads this ContextVar. That
# keeps the capture hooks to one line each and costs a single lookup when capture is off.

_ACTIVE: contextvars.ContextVar[Capture | None] = contextvars.ContextVar("openpi_probe_capture", default=None)


@contextlib.contextmanager
def session(config: CaptureConfig | None = None):
    """Enables capture for the duration of the block. Deliberately not reentrant."""
    if _ACTIVE.get() is not None:
        raise RuntimeError("A probe capture session is already active")
    cap = Capture(config or CaptureConfig.mid())
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
    return cap is not None and (cap.wants("prefix_image") or cap.wants("prefix_lang"))


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


def record_prefix_layout(image_spans, lang_span) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_prefix_layout(image_spans, lang_span)


def record_prefix_hidden(hidden_states) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_prefix_hidden(hidden_states)


def record_suffix_hidden(hidden_states) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_suffix_hidden(hidden_states)


def record_observation(tokens, token_mask, image_masks, state) -> None:
    if (cap := _ACTIVE.get()) is not None:
        cap.record_observation(tokens, token_mask, image_masks, state)
