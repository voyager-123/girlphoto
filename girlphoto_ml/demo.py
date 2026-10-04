"""Synthetic plumbing checks only. Generated labels are NOT aesthetic ground truth."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import write_image
from .renderer import render


def make_demo(folder):
    folder = Path(folder).resolve()
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    samples, pairs, tasks = [], [], []
    styles = ["retro", "fresh"]
    for group in range(6):
        split = "train" if group < 4 else "validation" if group == 4 else "test"
        source_id = f"g{group}_original"
        array = np.zeros((3, 64, 64), dtype=np.float32)
        ramp = np.linspace(.15, .75, 64, dtype=np.float32)
        for channel in range(3):
            array[channel] = ramp[None, :] + rng.uniform(-.08, .08)
        array[:, 15:52, 23:45] = np.array([.65, .48, .38], dtype=np.float32)[:, None, None]
        source = torch.from_numpy(array.clip(0, 1))

        def record(sid, image, role, quality, style_labels, strengths):
            write_image(image, images / f"{sid}.png")
            samples.append({"id": sid, "file": f"images/{sid}.png", "group_id": f"g{group}",
                            "cohort_id": f"synthetic-scene-{group}", "split": split, "role": role,
                            "quality": quality, "styles": style_labels, "strengths": strengths})

        record(source_id, source, "original", 2, {s: "no" for s in styles}, {s: 0 for s in styles})
        for index, style in enumerate(styles):
            strength = .65
            good = torch.tensor([.05, -.18, .45, -.08, -.25, .22] if index == 0 else [.20, -.10, -.20, .05, .12, .10]) * strength
            bad = good.clone()
            bad[0] = .65 if group % 2 == 0 else -.65
            for name, parameters, quality in (("good", good, 3), ("bad", bad, 1)):
                sid = f"g{group}_{style}_{name}"
                result = render(source[None], parameters[None])[0]
                record(sid, result, "edited", quality, {s: "yes" if s == style else "no" for s in styles},
                       {s: strength if s == style else 0 for s in styles})
            target = f"g{group}_{style}_good"
            pairs.append({"source": source_id, "left": target, "right": f"g{group}_{style}_bad", "style": style, "outcome": "left"})
            tasks.append({"source": source_id, "style": style, "strength": strength, "parameters": good.tolist(), "target": target})
    manifest = {"schema_version": 1, "demo_only": True, "styles": styles, "samples": samples, "pairs": pairs, "tasks": tasks,
                "note": "Synthetic functional-test labels, not real human preferences. Do not mix with real-photo training."}
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
