# Caching hidden states during inference

Records pi0 / pi0.5 activations while a policy runs and writes them to disk. That is all
it does — there is no episode bookkeeping, no index, and no dataset. Capture is off unless
you ask for it, and costs nothing when off.

Only the **PyTorch** inference path is instrumented. The JAX path wraps its layers in
`nn.scan` with `remat(policy=nothing_saveable)`, so pulling out per-layer hidden states
there means restructuring the scan; the PyTorch path already supports
`output_hidden_states`, so all 19 hidden states fall out of the same forward pass at no
extra compute cost.

## What can be captured

For pi0.5 with three cameras (`gemma_2b` prefix + `gemma_300m` action expert, both depth
18, `action_horizon=50`, `max_token_len=200`, `num_steps=10`):

| site | shape per row | dtype | size | what it is |
| --- | --- | --- | --- | --- |
| `prefix_pool` | `[19, 5, 2048]` | bf16 | 389 KB | per-camera mean, language mean, last prompt token, every layer |
| `suffix_hidden` | `[19, 10, 50, 1024]` | bf16 | 19.5 MB | action expert hidden states, every layer × every denoise step |
| `xt_traj` | `[11, 50, 32]` | f32 | 70 KB | flow-matching trajectory; `[0]` is the noise, `[-1]` the action |
| `vt` | `[10, 50, 32]` | f32 | 64 KB | predicted velocity per denoise step |
| `prefix_full` | `[19, 968, 2048]` | bf16 | 75.3 MB | every prefix token position; needed for spatial probes |
| `siglip_tokens` | `[3, 256, 2048]` | bf16 | 3.1 MB | image tokens *before* the LLM — the vision-only baseline |
| `adarms_cond` | `[10, 1024]` | f32 | 40 KB | adaRMS timestep conditioning (a function of `t` alone, so a control variable) |
| `tokens`, `token_mask`, `image_mask`, `state` | — | — | ~1 KB | input alignment data; included by the default presets |

`layer` is axis 0 everywhere, and the hidden-state tuple has `depth + 1 = 19` entries:
index `i < 18` is the **input** to layer `i`, and index `18` is the output after the final
norm.

Preset budgets, per inference call:

| preset | sites | size |
| --- | --- | --- |
| `CaptureConfig.lean()` | pooled prefix, action-expert final layer × 10 steps | **1.5 MB** |
| `CaptureConfig()` | default: pooled prefix, all suffix layers × steps | **20 MB** |
| `CaptureConfig.pilot()` | everything | **98 MB** |

## Switches

```python
from openpi.probe import capture

capture.CaptureConfig(
    sites={"prefix_pool", "suffix_hidden"},   # 1. which sites
    prefix_layers=(0, 4, 8, 12, 17, 18),      # 2. which layers (None = all 19)
    suffix_layers=(-1,),                      #    -1 is the post-final-norm output
    denoise_steps=(0, 5, 9),                  # 3. which denoise steps (None = all 10)
)
```

Plus a master switch on the recorder: `ActivationRecorder(..., enabled=False)` makes it a
transparent pass-through, so it is safe to leave wired into an eval script permanently.

## Recording

```python
import dataclasses

from openpi.policies import policy_config
from openpi.probe import capture, recorder
from openpi.training import config as _config

train_config = _config.get_config("pi05_libero")
# torch.compile must be off: capture works by Python-level side effects inside the sampling
# loop, which dynamo would trace away. The recorder detects and unwraps a compiled
# sample_actions anyway, but setting this avoids a wasted compile.
train_config = dataclasses.replace(
    train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=None)
)
policy = policy_config.create_trained_policy(train_config, checkpoint_dir)

with recorder.ActivationRecorder(
    policy, "probe_data/raw",
    config=capture.CaptureConfig.pilot(),
    extra_meta={"checkpoint": str(checkpoint_dir)},
) as rec:
    for obs in observations:
        action_chunk = rec.infer(obs)["actions"]
```

Output:

```
probe_data/raw/
  capture_meta.json     # per-site axes / shapes / dtypes / axis ids, and the row count
  shard_000000.npz      # stacked arrays, first axis is the row
  shard_000001.npz
```

Rows are in call order and shards are in name order, so row `i` of the run is row
`i % flush_every` of shard `i // flush_every`.

## Reading it back

```python
from openpi.probe import recorder

hidden = recorder.load_site("probe_data/raw", "suffix_hidden")   # [N, 19, 10, 50, 1024] f32
meta = recorder.load_meta("probe_data/raw")
meta["sites"]["suffix_hidden"]["axis_ids"]["layer"]              # which real layer each position is
```

`load_site` concatenates every shard and decodes bfloat16 to float32. Pass `decode=False`
to get the raw `int16` bit patterns instead, which is half the memory if you are only
moving them around; `openpi.probe.codec.decode` widens them later.

For a run big enough that `load_site` will not fit in RAM, read the shards one at a time:

```python
for path in sorted(pathlib.Path("probe_data/raw").glob("shard_*.npz")):
    with np.load(path) as npz:
        block = codec.decode(npz["suffix_hidden"], codec.BF16_RAW)
```

**Axis ids matter.** If you captured `suffix_layers=(0, 8, 17)`, the stored array has three
layer positions, and `axis_ids["layer"] == [0, 8, 17]` is the only record of which is which.
Read it from the meta rather than assuming positions are layer numbers.

## Things worth knowing

- **`torch.compile`.** Set `Pi0Config.pytorch_compile_mode=None` for capture runs. A
  compiled sampling loop silently records nothing.
- **`float16` is not safe here.** Gemma's residual stream reaches magnitudes past the
  float16 ceiling of 65504 in the later layers. Activations are stored as raw bfloat16 bit
  patterns in an `int16` array — lossless, 2 bytes, plain numpy.
- **The noise is not fixed.** `sample_actions` draws fresh Gaussian noise on every call and
  it feeds straight back into the action expert, so repeated captures of the same
  observation differ. Pass your own `noise=` through `rec.infer(obs, noise=...)` if you
  need reproducible suffix activations.
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

## Tests

```bash
uv run pytest src/openpi/probe/probe_test.py
```

Covers capture and the shard writer on fabricated tensors — no checkpoint, no GPU. Whether
the hooks sit in the right places in the model only shows up when a real policy runs.
