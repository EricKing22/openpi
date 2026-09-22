"""Build a PyTorch token dataset from recorded Astra requests.

Validate inputs only (no checkpoint/GPU): python scripts/cache_astra.py --index-only
Capture: python scripts/cache_astra.py --checkpoint PATH --output PATH
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from openpi.probe.astra import catalog


def run(args):
    root = args.dataset.expanduser().resolve()
    samples = catalog(root, args.task)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No labelled hybrid requests selected")
    print(json.dumps({"samples": len(samples), "labels": dict(Counter(s.info["mode"] for s in samples))}, indent=2))

    if args.index_only:
        # Exercise the real preview/state/prompt inputs without loading the model.
        for sample in samples:
            sample.observation(root)
        print("All indexed inputs validated.")
        return

    import numpy as np
    import torch

    from openpi.policies import policy_config
    from openpi.probe import capture
    from openpi.probe.astra import to_numpy
    from openpi.training import config

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = config.get_config(args.config)
    capture_config = capture.CaptureConfig.high()
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint.expanduser().resolve(), sample_kwargs={"num_steps": args.num_steps}
    )
    rng = np.random.default_rng(args.seed)
    arrays, rows = {}, []
    for index, sample in enumerate(samples):
        noise = rng.standard_normal((cfg.model.action_horizon, cfg.model.action_dim), dtype=np.float32)
        with torch.inference_mode(), capture.session(capture_config) as session:
            policy.infer(sample.observation(root), noise=noise)
            captured, sites = session.result()
        # Policy.infer works on a batch of one, so drop that axis.
        request = {site: to_numpy(tensor[0]) for site, tensor in captured.items()}
        if not arrays:
            # One file per site, row i being request i. Preallocated, so each request is
            # written in place and only that request is ever in memory.
            arrays = {
                site: np.lib.format.open_memmap(
                    output / f"{site}.npy", mode="w+", shape=(len(samples), *array.shape), dtype=array.dtype
                )
                for site, array in request.items()
            }
            meta = {
                "sites": sites,
                "bfloat16_sites": sorted(s for s, t in captured.items() if t.dtype is torch.bfloat16),
                "config": args.config,
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
            }
            (output / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        for site, array in request.items():
            arrays[site][index] = array
        rows.append(dict(sample.info, request=sample.request))
        print(f"Cached {index + 1}/{len(samples)}: {sample.info['episode_id']} request {sample.info['request_id']}")
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("~/dataset/astra-robodojo-rollouts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", default="pi05_robodojo")
    parser.add_argument("--task")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if not args.index_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --index-only")
    run(args)


if __name__ == "__main__":
    main()
