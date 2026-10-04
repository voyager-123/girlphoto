"""Explicit manifests; missing labels are masked and split leakage is rejected."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
import torch


def read_image(path, size=None):
    with Image.open(path) as opened:
        icc = opened.info.get("icc_profile")
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if icc:
            try:
                image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(__import__('io').BytesIO(icc)),
                                                 ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError) as exc:
                raise ValueError(f"Cannot convert embedded color profile: {path}") from exc
        if size:
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)


def write_image(image, path):
    array = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


class Manifest:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if self.raw.get("schema_version") != 1:
            raise ValueError("Training manifest schema_version must be 1 (different from annotation export schema).")
        self.styles = self.raw.get("styles", [])
        if not self.styles or len(self.styles) > 20 or any(not isinstance(s, str) or not s.strip() for s in self.styles) or len(set(self.styles)) != len(self.styles):
            raise ValueError("Specify 1-20 unique style IDs.")
        self.samples = self.raw.get("samples", [])
        self.ids = {}
        groups, cohorts = {}, {}
        for sample in self.samples:
            sid = sample.get("id")
            if not isinstance(sid, str) or not sid or sid in self.ids:
                raise ValueError("Every image needs a unique nonempty string id.")
            self.ids[sid] = sample
            split = sample.get("split")
            if split not in ("train", "validation", "test"):
                raise ValueError(f"Invalid split for {sid}.")
            for field, seen in (("group_id", groups), ("cohort_id", cohorts)):
                value = sample.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Missing {field}: {sid}.")
                if value in seen and seen[value] != split:
                    raise ValueError(f"Split leakage: {field} {value} crosses data splits.")
                seen[value] = split
            file = sample.get("file")
            if not isinstance(file, str) or not file:
                raise ValueError(f"Missing image path: {sid}.")
            sample["resolved_file"] = (self.path.parent / file).resolve()
            if not sample["resolved_file"].is_file():
                raise ValueError(f"Image does not exist: {file}")
            quality = sample.get("quality")
            if quality is not None and (type(quality) is not int or quality not in (1, 2, 3)):
                raise ValueError("Quality must be 1, 2, 3 or null.")
            labels = sample.get("styles", {})
            if not isinstance(labels, dict) or any(k not in self.styles or v not in ("yes", "no", "unknown") for k, v in labels.items()):
                raise ValueError("Invalid style labels.")
            strengths = sample.get("strengths", {})
            if not isinstance(strengths, dict):
                raise ValueError("strengths must be an object.")
            for style, value in strengths.items():
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 or labels.get(style) not in ("yes", "no"):
                    raise ValueError("Strength needs a known style judgment and a number in [0,1].")
                if (labels[style] == "no" and value != 0) or (labels[style] == "yes" and value == 0):
                    raise ValueError("Strength zero means absent; a present style requires a positive strength.")
            if sample.get("role", "unknown") not in ("original", "edited", "reference", "unknown"):
                raise ValueError("Invalid image role.")
        if not self.samples:
            raise ValueError("No samples in manifest.")
        self.pairs = []
        for pair in self.raw.get("pairs", []):
            if pair.get("outcome") == "skip":
                continue
            if pair.get("outcome") not in ("left", "right", "tie") or pair.get("style") not in self.styles:
                raise ValueError("Invalid preference pair.")
            records = self.related([pair.get(k) for k in ("source", "left", "right")])
            if records[0].get("role") != "original" or pair["left"] == pair["right"]:
                raise ValueError("Pair needs a marked original and two distinct candidates.")
            if any(s.get("styles", {}).get(pair["style"]) != "yes" for s in records[1:]):
                raise ValueError("Both preference candidates must be confirmed in the target style.")
            self.pairs.append(pair)
        self.tasks = self.raw.get("tasks", [])
        for task in self.tasks:
            source = self.related([task.get("source")])[0]
            if source.get("role") != "original" or task.get("style") not in self.styles:
                raise ValueError("Model 2 tasks require an original image and a known style.")
            strength = task.get("strength")
            if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("Task strength must be in [0,1].")
            params = task.get("parameters")
            if params is not None and (not isinstance(params, list) or len(params) != 6 or any(type(p) not in (int, float) or not math.isfinite(p) or abs(p) > strength + 1e-7 for p in params)):
                raise ValueError("Task parameters must be six normalized values bounded by task strength.")
            if task.get("target"):
                self.related([task["source"], task["target"]])
        self.demo_only = bool(self.raw.get("demo_only", False))

    def related(self, ids):
        if any(not isinstance(sid, str) or sid not in self.ids for sid in ids):
            raise ValueError("A source/candidate/target id does not exist.")
        records = [self.ids[sid] for sid in ids]
        if any(len({s[field] for s in records}) != 1 for field in ("group_id", "cohort_id", "split")):
            raise ValueError("Related images must share original group, cohort and split.")
        return records

    def image(self, sid, size):
        return read_image(self.ids[sid]["resolved_file"], size)

    def select(self, split):
        return [s for s in self.samples if s["split"] == split]
