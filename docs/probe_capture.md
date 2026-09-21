# Caching hidden states during inference

Records pi0 / pi0.5 activations and writes them to disk, one safetensors file per
inference call, grouped into one folder per episode.

Capture is **offline only**. The robot runs the normal, compiled policy with no recorder;
the recorded observations are replayed through a capture run afterwards (see
[Offline workflow](#offline-workflow)). Capture is off unless you ask for it, and costs
nothing when off.

Only the **PyTorch** inference path is instrumented. The JAX path wraps its layers in
`nn.scan` with `remat(policy=nothing_saveable)`, so pulling out per-layer hidden states
there means restructuring the scan; the PyTorch path already supports
`output_hidden_states`, so all 19 hidden states fall out of the same forward pass at no
extra compute cost.

## Presets: low / mid / high

Three levels. `CaptureConfig()` is `mid`, and `ActivationRecorder` uses `mid` when no
`config` is given.

| what is stored | `CaptureConfig.low()` | `CaptureConfig.mid()` (default) | `CaptureConfig.high()` |
| --- | --- | --- | --- |
| VLM image tokens, `prefix_image` | all 19 layers | all 19 layers | all 19 layers |
| action expert, `suffix_hidden` | final layer only, 10 denoise steps | all 19 layers, 10 denoise steps | all 19 layers, 10 denoise steps |
| flow trajectory, `xt_traj` + `vt` | yes | yes | yes |
| alignment, `tokens` `token_mask` `image_mask` `state` | yes | yes | yes |
| VLM language tokens, `prefix_lang` | — | — | all 19 layers |
| SigLIP output, `siglip_tokens` | — | — | yes |
| adaRMS conditioning, `adarms_cond` | — | — | yes |
| **per file** | **~61 MB** | **~79 MB** | **~98 MB** |

In words:

- **low**: every VLM layer's image tokens, plus the action expert's final layer (the one
  that feeds `action_out_proj`).
- **mid**: low, plus every other action-expert layer.
- **high**: mid, plus the language tokens, the SigLIP output and the adaRMS conditioning.
  Everything there is.

Every preset keeps every prefix layer and every denoise step. The image tokens dominate
all three sizes; only the action-expert sites scale with `action_horizon`.

## Sites

Sizes are for pi0.5 with three camera slots (`gemma_2b` prefix + `gemma_300m` action
expert, both depth 18, `action_horizon=50`, `max_token_len=200`, `num_steps=10`).
Nothing is pooled anywhere: every site stores the tensor as the model produced it, or a
plain slice of it.

| site | where it hooks | shape per call | dtype | size | filtered by |
| --- | --- | --- | --- | --- | --- |
| `prefix_image` | VLM, every layer, the 768 image positions, split per camera | `[19, 3, 256, 2048]` | bf16 | 59.8 MB | `prefix_layers` |
| `prefix_lang` | VLM, every layer, the 200 language positions (prompt + discretized state, padded) | `[19, 200, 2048]` | bf16 | 15.6 MB | `prefix_layers` |
| `siglip_tokens` | SigLIP output after the projector, before Gemma | `[3, 256, 2048]` | bf16 | 3.1 MB | — |
| `suffix_hidden` | action expert, every layer × every denoise step, last `action_horizon` tokens | `[19, 10, 50, 1024]` | bf16 | 19.5 MB | `suffix_layers`, `denoise_steps` |
| `xt_traj` | `x_t` at every denoise step; `[0]` is the noise, `[-1]` the action | `[11, 50, 32]` | f32 | 70 KB | — |
| `vt` | predicted velocity at every denoise step | `[10, 50, 32]` | f32 | 64 KB | — |
| `adarms_cond` | adaRMS timestep conditioning; a function of `t` alone, so a control variable | `[10, 1024]` | f32 | 40 KB | — |
| `tokens`, `token_mask`, `image_mask`, `state` | model inputs, for telling real cameras and prompt positions from padding | — | — | ~1 KB | — |

`prefix_image` and `prefix_lang` split the prefix without overlap: the prefix is the
three camera blocks followed by the language block, so concatenating the flattened
`prefix_image` with `prefix_lang` gives back the whole prefix at each layer. Camera order
is fixed by `IMAGE_KEYS`: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`. Masked
cameras and padded prompt positions are kept so every call has the same shape;
`image_mask` and `token_mask` say which ones are real. A slot fed a zeroed image without
being masked looks real to `image_mask`, so filter it out at training time yourself.

Reading the layer axis: `layer` is axis 0 everywhere, and each hidden-state tuple has
`depth + 1 = 19` entries. Index `i < 18` is the **input** to layer `i`; index `18` is the
output after the final norm.

- On the VLM side, entries 0–17 are what the action expert's per-layer K/V are computed
  from. Entry 18 is never read by the action expert.
- On the action-expert side, entry 18 is what feeds `action_out_proj`: the "last layer
  before the velocity head".

`denoise_steps` filters `suffix_hidden` only. `xt_traj`, `vt` and `adarms_cond` always
hold every step.

## Custom configs

The presets are plain `CaptureConfig`s; build your own when none fits:

```python
from openpi.probe import capture

capture.CaptureConfig(
    sites={"prefix_image", "suffix_hidden"} | capture.ALIGNMENT_SITES,  # which sites
    prefix_layers=None,    # which VLM layers (None = all 19)
    suffix_layers=(-1,),   # which action-expert layers; -1 is the post-final-norm output
    denoise_steps=(9,),    # which denoise steps for suffix_hidden (None = all 10)
)
```

That one stores every VLM layer's image tokens plus the action expert's final layer at the
last denoise step, about 59.9 MB per file. `capture.CORE_SITES` is the site set of low and
mid; `capture.ALL_SITES` is high's.

## Offline workflow

1. **On the robot**, serve the normal compiled policy and keep the observations. openpi's
   own `scripts/serve_policy.py --record` wraps the policy in `PolicyRecorder`, which dumps
   every observation the server receives (and the actions it returned) to
   `policy_records/step_N.npy`. Episode boundaries come from your robot-side logs.
2. **Offline**, load the same checkpoint with `torch.compile` off and replay those
   observations through `ActivationRecorder`, calling `reset()` at each episode boundary.

The prefix sites (`prefix_image`, `prefix_lang`) depend only on images, prompt and state,
so replay reproduces them exactly. The action-expert sites (`suffix_hidden`, `xt_traj`,
`vt`) also depend on the flow-matching noise, which the online run does not save: on
replay they are a fresh sample for the same observation, not the one whose actions the
robot executed. If you need that match, log the noise on the server during the online run
and pass it back with `rec.infer(obs, noise=...)`.

## Recording

```python
import dataclasses

from openpi.policies import policy_config
from openpi.probe import capture, recorder
from openpi.training import config as _config

train_config = _config.get_config(config_name)  # the pi0.5 config your checkpoint uses
# torch.compile must be off: capture works by Python-level side effects inside the sampling
# loop, which dynamo would trace away. The recorder detects and unwraps a compiled
# sample_actions anyway, but setting this avoids a wasted compile.
train_config = dataclasses.replace(
    train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=None)
)
policy = policy_config.create_trained_policy(train_config, checkpoint_dir)

rec = recorder.ActivationRecorder(
    policy, "probe_data/run1",
    config=capture.CaptureConfig.mid(),
    extra_meta={"checkpoint": str(checkpoint_dir)},
)
for episode in recorded_episodes:
    rec.reset()                              # next call goes into a new ep_ folder
    for obs in episode:
        rec.infer(obs)
```

`reset()` also resets the wrapped policy. A reset before the current episode has any
steps does nothing, so calling it at the start of every episode never leaves an empty
folder. Without any `reset()`, everything goes into `ep_00000`.

Output:

```
probe_data/run1/
  meta.json                     # per-site axes / shape / dtype / layer ids, capture config
  ep_00000/
    t_00000.safetensors         # episode 0, inference call 0
    t_00001.safetensors
  ep_00001/
    t_00000.safetensors
```

Each file is a flat `{site: tensor}` dict for one inference call, batch axis dropped. The
recorder refuses a directory that already holds `ep_*` folders, so two runs never mix.

## Reading it back

One step is one call:

```python
from safetensors.torch import load_file

step = load_file("probe_data/run1/ep_00003/t_00017.safetensors")
step["prefix_image"]            # torch.bfloat16 [19, 3, 256, 2048]
step["image_mask"]              # torch.bool [3]
```

A whole episode, stacked along a new leading time axis:

```python
from openpi.probe import recorder

ep = recorder.load_episode("probe_data/run1", 3)                     # every site
ep = recorder.load_episode("probe_data/run1", 3, sites=["suffix_hidden", "image_mask"])
ep["suffix_hidden"]             # torch.bfloat16 [T, 19, 10, 50, 1024]
```

`sites=` reads only those keys from each file, so loading the small sites never touches
the large ones on disk. Step `t` of episode `e` is `t_{t:05d}.safetensors` in
`ep_{e:05d}`, which is what your labels line up with.

**Layer ids matter.** If you captured `suffix_layers=(0, 8, 17)`, the stored tensor has
three layer positions, and `meta["sites"]["suffix_hidden"]["axis_ids"]["layer"] == [0, 8, 17]`
is the only record of which is which. Read it from `recorder.load_meta(...)` rather than
assuming positions are layer numbers.

## Things worth knowing

- **`torch.compile`.** Set `Pi0Config.pytorch_compile_mode=None` for capture runs. A
  compiled sampling loop silently records nothing. This is also why capture is offline:
  the uncompiled loop is slower than the one the robot runs.
- **Hidden states stay bfloat16.** Gemma's residual stream reaches magnitudes past the
  float16 ceiling of 65504 in the later layers, so float16 is not an option; safetensors
  stores bfloat16 natively. Call `.float()` after loading if your probe wants float32.
- **Scale before probing.** Layer 0 and layer 17 of a residual stream differ in scale by
  orders of magnitude, so a probe trained across layers on unstandardized inputs barely
  moves. Compute the statistics once and reuse them at eval time.
- **`x_t` is an input, not just an output.** It feeds the action expert at every denoise
  step, so a probe predicting actions from `suffix_hidden` may just be decoding it back
  out. `xt_traj` is captured so you can run that control.
- **The denoise-step axis is a confound.** adaRMS modulates every action-expert layer by
  the timestep, so mixing steps in one probe set lets the probe read the clock.
- **Memory.** `output_hidden_states` keeps all 19 layer outputs alive until they are copied
  to CPU, about 75 MB of extra GPU memory for the pi0.5 prefix.
- **Disk.** At `mid`, an episode of 200 calls is about 16 GB. Plan the output directory's
  disk space before a long replay.

## Tests

```bash
uv run pytest src/openpi/probe/probe_test.py
```

Covers capture and the recorder on fabricated tensors — no checkpoint, no GPU. Whether
the hooks sit in the right places in the model only shows up when a real policy runs.
