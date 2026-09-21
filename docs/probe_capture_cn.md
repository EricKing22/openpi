# 推理期间缓存隐藏状态

记录 pi0 / pi0.5 的激活并写入磁盘。每次推理调用生成一个 safetensors 文件，每个 episode 单独存放在一个文件夹中。

采集**只在离线进行**。机器人照常运行编译过的策略，不接记录器；之后再把记录下来的观测回放一遍来采集（见[离线流程](#离线流程)）。除非显式启用，否则采集功能保持关闭；关闭时不会产生额外开销。

目前只对 **PyTorch** 推理路径进行了插桩。JAX 路径使用带有 `remat(policy=nothing_saveable)` 的 `nn.scan` 包装各层，因此若要从中提取逐层隐藏状态，需要重构 scan；而 PyTorch 路径已经支持 `output_hidden_states`，所以同一次前向传播即可得到全部 19 组隐藏状态，不增加额外计算量。

## 预设：low / mid / high

共三档。直接构造 `CaptureConfig()` 即为 `mid`；未传入 `config` 时，`ActivationRecorder` 也使用 `mid`。

| 保存内容 | `CaptureConfig.low()` | `CaptureConfig.mid()`（默认） | `CaptureConfig.high()` |
| --- | --- | --- | --- |
| VLM 图像词元 `prefix_image` | 全部 19 层 | 全部 19 层 | 全部 19 层 |
| 动作专家 `suffix_hidden` | 仅最终层，10 个去噪步骤 | 全部 19 层，10 个去噪步骤 | 全部 19 层，10 个去噪步骤 |
| 去噪轨迹 `xt_traj` + `vt` | 保存 | 保存 | 保存 |
| 对齐数据 `tokens` `token_mask` `image_mask` `state` | 保存 | 保存 | 保存 |
| VLM 语言词元 `prefix_lang` | — | — | 全部 19 层 |
| SigLIP 输出 `siglip_tokens` | — | — | 保存 |
| adaRMS 条件 `adarms_cond` | — | — | 保存 |
| **每个文件** | **约 61 MB** | **约 79 MB** | **约 98 MB** |

简单地说：

- **low**：VLM 每一层的图像词元，加上动作专家的最终层（也就是送入 `action_out_proj` 的那一层）。
- **mid**：在 low 的基础上，加上动作专家的其余各层。
- **high**：在 mid 的基础上，加上语言词元、SigLIP 输出和 adaRMS 条件，即全部采集点。

三档都保存 prefix 的全部层和全部去噪步骤。三档的大小主要都来自图像词元；只有动作专家侧的采集点会随 `action_horizon` 变化。

## 采集点

以下大小以具有三个相机槽位的 pi0.5 为例（`gemma_2b` prefix + `gemma_300m` 动作专家，两者深度均为 18，`action_horizon=50`、`max_token_len=200`、`num_steps=10`）。任何位置都不会进行池化：每个采集点保存模型原样产生的张量，或者该张量的普通切片。

| 采集点 | 插桩位置 | 每次调用的形状 | 数据类型 | 大小 | 过滤参数 |
| --- | --- | --- | --- | --- | --- |
| `prefix_image` | VLM 每一层的 768 个图像位置，并按相机拆分 | `[19, 3, 256, 2048]` | bf16 | 59.8 MB | `prefix_layers` |
| `prefix_lang` | VLM 每一层的 200 个语言位置（提示词 + 离散化状态，包含填充） | `[19, 200, 2048]` | bf16 | 15.6 MB | `prefix_layers` |
| `siglip_tokens` | SigLIP 经投影器处理后的输出，位于 Gemma 之前 | `[3, 256, 2048]` | bf16 | 3.1 MB | — |
| `suffix_hidden` | 动作专家每一层 × 每个去噪步骤中最后 `action_horizon` 个词元 | `[19, 10, 50, 1024]` | bf16 | 19.5 MB | `suffix_layers`、`denoise_steps` |
| `xt_traj` | 每个去噪步骤的 `x_t`；`[0]` 是噪声，`[-1]` 是动作 | `[11, 50, 32]` | f32 | 70 KB | — |
| `vt` | 每个去噪步骤的预测速度 | `[10, 50, 32]` | f32 | 64 KB | — |
| `adarms_cond` | adaRMS 时间步条件；它仅由 `t` 决定，因此可作为控制变量 | `[10, 1024]` | f32 | 40 KB | — |
| `tokens`、`token_mask`、`image_mask`、`state` | 模型输入，用于区分真实相机、提示词位置和填充位置 | — | — | 约 1 KB | — |

`prefix_image` 与 `prefix_lang` 将 prefix 无重叠地拆成两部分：prefix 先包含三个相机块，随后是语言块。因此，在每一层把展平后的 `prefix_image` 与 `prefix_lang` 拼接起来，就能恢复完整 prefix。相机顺序由 `IMAGE_KEYS` 固定：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。被屏蔽的相机和经过填充的提示词位置仍会保留，以确保每次调用的形状一致；`image_mask` 和 `token_mask` 用来标记哪些位置是真实数据。如果某个槽位输入的是全零图像但没有被 mask，`image_mask` 会把它当作真实相机，训练时需要自己将其过滤掉。

层轴的读取方式：凡是包含 `layer` 轴的采集点，该轴都位于第 0 维。每个隐藏状态元组包含 `depth + 1 = 19` 个元素。索引 `i < 18` 表示第 `i` 层的**输入**；索引 `18` 表示最终归一化后的输出。

- 在 VLM 一侧，编号 0–17 的状态用于计算动作专家每一层的 K/V。编号 18 从不被动作专家读取。
- 在动作专家一侧，编号 18 会送入 `action_out_proj`，即“速度头之前的最后一层”。

`denoise_steps` 只过滤 `suffix_hidden`。`xt_traj`、`vt` 和 `adarms_cond` 始终保存全部步骤。

## 自定义配置

三档预设本身就是普通的 `CaptureConfig`；都不合适时可以自己组合：

```python
from openpi.probe import capture

capture.CaptureConfig(
    sites={"prefix_image", "suffix_hidden"} | capture.ALIGNMENT_SITES,  # 采集哪些位置
    prefix_layers=None,    # 采集 VLM 哪些层（None = 全部 19 层）
    suffix_layers=(-1,),   # 采集动作专家哪些层；-1 表示最终归一化后的输出
    denoise_steps=(9,),    # suffix_hidden 采集哪些去噪步骤（None = 全部 10 步）
)
```

上面这个配置保存 VLM 每一层的图像词元，以及动作专家在最后一个去噪步骤的最终层，每个文件约 59.9 MB。`capture.CORE_SITES` 是 low 和 mid 使用的采集点集合，`capture.ALL_SITES` 是 high 使用的集合。

## 离线流程

1. **在机器人上**，照常运行编译过的策略，并保留观测。openpi 自带的 `scripts/serve_policy.py --record` 会用 `PolicyRecorder` 包装策略，把服务端收到的每个观测（以及返回的动作）写到 `policy_records/step_N.npy`。episode 的边界从你们机器人侧的日志中获取。
2. **离线时**，加载同一个 checkpoint 并关闭 `torch.compile`，把这些观测回放给 `ActivationRecorder`，在每个 episode 的边界调用 `reset()`。

prefix 侧的采集点（`prefix_image`、`prefix_lang`）只取决于图像、提示词和状态，因此回放可以精确复现。动作专家侧的采集点（`suffix_hidden`、`xt_traj`、`vt`）还取决于流匹配噪声，而在线运行时不会保存这个噪声：回放时它们是对同一观测的一次新采样，并不是机器人实际执行的那一次。如果需要两者一致，请在线运行时在服务端记录噪声，回放时通过 `rec.infer(obs, noise=...)` 传回去。

## 记录数据

```python
import dataclasses

from openpi.policies import policy_config
from openpi.probe import capture, recorder
from openpi.training import config as _config

train_config = _config.get_config(config_name)  # checkpoint 对应的 pi0.5 配置
# 必须关闭 torch.compile：采集依赖采样循环中的 Python 级副作用，
# dynamo 会在跟踪时将这些副作用消除。记录器虽然能够检测并解包已编译的
# sample_actions，但在这里直接关闭可以避免一次无用的编译。
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
    rec.reset()                              # 下一次调用写入新的 ep_ 文件夹
    for obs in episode:
        rec.infer(obs)
```

`reset()` 还会重置被包装的策略。如果当前 episode 尚未包含任何步骤，调用 reset 不会执行任何操作。因此，在每个 episode 开头调用它不会留下空文件夹。如果始终不调用 `reset()`，所有内容都会写入 `ep_00000`。

输出目录结构：

```text
probe_data/run1/
  meta.json                     # 各采集点的轴、形状、数据类型、层 ID 和采集配置
  ep_00000/
    t_00000.safetensors         # episode 0，第 0 次推理调用
    t_00001.safetensors
  ep_00001/
    t_00000.safetensors
```

每个文件都是一次推理调用对应的扁平 `{site: tensor}` 字典，批次轴已移除。如果目标目录中已经存在 `ep_*` 文件夹，记录器会拒绝写入，从而避免两次运行的数据混在一起。

## 读取数据

一个步骤对应一次调用：

```python
from safetensors.torch import load_file

step = load_file("probe_data/run1/ep_00003/t_00017.safetensors")
step["prefix_image"]            # torch.bfloat16 [19, 3, 256, 2048]
step["image_mask"]              # torch.bool [3]
```

读取整个 episode，并沿新的时间轴堆叠：

```python
from openpi.probe import recorder

ep = recorder.load_episode("probe_data/run1", 3)                     # 读取所有采集点
ep = recorder.load_episode("probe_data/run1", 3, sites=["suffix_hidden", "image_mask"])
ep["suffix_hidden"]             # torch.bfloat16 [T, 19, 10, 50, 1024]
```

`sites=` 只从每个文件读取指定的键，因此加载小型采集点时不会触碰磁盘上的大型采集点。episode `e` 的第 `t` 步位于 `ep_{e:05d}` 中的 `t_{t:05d}.safetensors`，这正是标签应当对齐的位置。

**层 ID 非常重要。** 如果使用 `suffix_layers=(0, 8, 17)`，保存的张量只有三个层位置，而 `meta["sites"]["suffix_hidden"]["axis_ids"]["layer"] == [0, 8, 17]` 是这些位置分别对应哪一层的唯一记录。应从 `recorder.load_meta(...)` 读取它，不要假设数组位置就是层编号。

## 注意事项

- **`torch.compile`。** 采集运行时应设置 `Pi0Config.pytorch_compile_mode=None`。经过编译的采样循环会在没有任何提示的情况下记录不到数据。这也是采集放在离线进行的原因：未编译的采样循环比机器人实际运行的版本慢。
- **隐藏状态保持为 bfloat16。** Gemma 残差流在后几层的数值幅度会超过 float16 的上限 65504，因此不能使用 float16；safetensors 原生支持存储 bfloat16。如果探针需要 float32，请在加载后调用 `.float()`。
- **训练探针前应进行缩放。** 残差流第 0 层和第 17 层的尺度可能相差多个数量级，因此在未标准化的输入上训练跨层探针时，探针几乎无法有效更新。请只计算一次统计量，并在评估时复用。
- **`x_t` 不只是输出，也是输入。** 它会在每个去噪步骤送入动作专家，因此从 `suffix_hidden` 预测动作的探针可能只是在重新解码它。采集 `xt_traj` 是为了能够执行这项控制实验。
- **去噪步骤轴是一个混杂因素。** adaRMS 会根据时间步调制动作专家的每一层，因此把多个步骤混入同一个探针数据集，会让探针有机会直接读出“时钟”。
- **内存。** `output_hidden_states` 会让全部 19 层输出一直保留到复制到 CPU 为止。对于 pi0.5 的 prefix，这会额外占用约 75 MB GPU 显存。
- **磁盘。** 在 `mid` 档下，一个包含 200 次调用的 episode 约为 16 GB。长时间回放之前请先规划好输出目录的磁盘空间。

## 测试

```bash
uv run pytest src/openpi/probe/probe_test.py
```

这些测试使用构造的张量覆盖采集逻辑和记录器，不需要 checkpoint 或 GPU。只有运行真实策略时，才能验证钩子是否位于模型中的正确位置。
