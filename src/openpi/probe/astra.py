"""Decision-time Astra inputs and intervention labels, read directly from release ZIPs.

One labelled hybrid request becomes one sample: the recorded decision-time previews,
state and instruction go in, `student=0` / `edit,eef=1` comes out. Responses whose mode
carries no intervention label are skipped, never relabelled.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LABELS = {"student": 0, "edit": 1, "eef": 1}


def _read_rows(archive: zipfile.ZipFile, member: str) -> list[dict]:
    if member not in archive.namelist():
        return []
    return [json.loads(line) for line in archive.read(member).splitlines() if line.strip()]


@dataclasses.dataclass
class Sample:
    info: dict
    request: dict

    def observation(self, root: Path) -> dict:
        """Only pre-decision images/state/instruction enter the policy."""
        with zipfile.ZipFile(Path(root) / self.info["core_path"]) as archive:
            images = {}
            for camera in CAMERAS:
                with Image.open(io.BytesIO(archive.read(self.info["preview_members"][camera]))) as image:
                    images[camera] = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).copy()
        return {
            "images": images,
            "state": np.asarray(self.request["current_state"], dtype=np.float32),
            "prompt": self.request["instruction"],
        }


def catalog(root: str | Path, task: str | None = None) -> list[Sample]:
    """Index labelled hybrid requests, executed and unexecuted alike.

    Direct episodes are excluded because they do not label intervention on a student policy.
    """
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "astra.robodojo.dataset.v1":
        raise ValueError("Unsupported Astra dataset schema")
    samples = []
    for episode in manifest["episodes"]:
        if episode["method"] != "hybrid" or (task and episode["task"] != task):
            continue
        core_path = episode["files"]["core"]["path"]
        with zipfile.ZipFile(root / core_path) as archive:
            by_index = {row["index"]: row for row in json.loads(archive.read("observation_index.json"))}
            for member in ("decisions.jsonl", "unexecuted_decisions.jsonl"):
                for decision in _read_rows(archive, member):
                    mode = decision["response"]["mode"]
                    if mode not in LABELS:
                        continue
                    observation = by_index[decision["observation_index"]]
                    samples.append(
                        Sample(
                            {
                                "episode_id": episode["episode_id"],
                                "task": episode["task"],
                                "request_id": decision["request"]["request_id"],
                                "observation_index": observation["index"],
                                "mode": mode,
                                "label": LABELS[mode],
                                "core_path": core_path,
                                "preview_members": observation["preview_members"],
                            },
                            decision["request"],
                        )
                    )
    return samples


def to_numpy(tensor):
    """`.npy` has no bfloat16, so such tensors are stored as their uint16 bit pattern."""
    import torch

    if tensor.dtype is torch.bfloat16:
        return tensor.view(torch.uint16).numpy()
    return tensor.numpy()


class CachedAstraDataset:
    """Lazy per-request tensor reader for the cache written by `scripts/cache_astra.py`.

    Each site is one memory-mapped `<site>.npy` whose row i is request i, so a sample
    reads only the rows it needs and `samples.jsonl` row i describes it.
    """

    def __init__(self, root: str | Path, sites: list[str] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.rows = [json.loads(line) for line in (self.root / "samples.jsonl").read_text().splitlines() if line]
        self.bfloat16_sites = set(json.loads((self.root / "meta.json").read_text())["bfloat16_sites"])
        paths = [self.root / f"{site}.npy" for site in sites] if sites else sorted(self.root.glob("*.npy"))
        self.arrays = {path.stem: np.load(path, mmap_mode="r") for path in paths}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        features = {}
        for site, array in self.arrays.items():
            tensor = torch.from_numpy(np.array(array[index]))
            features[site] = tensor.view(torch.bfloat16) if site in self.bfloat16_sites else tensor
        return {"features": features, "label": self.rows[index]["label"], "info": self.rows[index]}
