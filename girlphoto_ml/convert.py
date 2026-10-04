"""Convert one reviewer's annotation export. Never fabricate missing original links."""
import json
from pathlib import Path
import zipfile


def convert_export(export_file, data_dir, reviewer, out, original_marker=None):
    export_file, data_dir, out = Path(export_file), Path(data_dir).resolve(), Path(out).resolve()
    if zipfile.is_zipfile(export_file):
        with zipfile.ZipFile(export_file) as archive:
            bundle = json.loads(archive.read("dataset.json"))
    else:
        bundle = json.loads(export_file.read_text(encoding="utf-8-sig"))
    if bundle.get("schema_version") != 1:
        raise ValueError("This converter currently supports annotation export schema 1.")
    labels = {r["photo_id"]: r for r in bundle["labels"] if r["reviewer"] == reviewer and r["status"] == "labeled"}
    if not labels:
        raise ValueError("No saved labels for this reviewer.")
    samples, originals = [], {}
    for photo in bundle["photos"]:
        if photo.get("is_demo"):
            continue
        label = labels.get(photo["id"], {})
        path = (data_dir / photo["original"]).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError("Export image path escapes the selected data directory.")
        role = "unknown"
        if original_marker and Path(photo["name"]).stem.endswith(original_marker):
            if photo["group_id"] in originals:
                raise ValueError("The original filename marker matched multiple images in one group.")
            originals[photo["group_id"]] = photo["id"]
            role = "original"
        samples.append({"id": photo["id"], "file": path.as_posix(), "group_id": photo["group_id"],
                        "cohort_id": photo["cohort_id"], "split": photo["split"], "role": role,
                        "quality": label.get("quality"), "styles": label.get("styles", {}), "strengths": {}})
    ids = {s["id"]: s for s in samples}
    pairs, unlinked = [], 0
    for pair in bundle["pairs"]:
        if pair["reviewer"] != reviewer or not pair.get("eligible") or pair.get("is_demo") or pair["outcome"] == "skip":
            continue
        if pair["a"] not in ids or pair["b"] not in ids:
            continue
        source = originals.get(ids[pair["a"]]["group_id"])
        if not source:
            unlinked += 1
            continue
        pairs.append({"source": source, "left": pair["a"], "right": pair["b"], "style": pair["style_id"],
                      "outcome": {"a": "left", "b": "right", "tie": "tie"}[pair["outcome"]]})
    result = {"schema_version": 1, "demo_only": False, "styles": [s["id"] for s in bundle["styles"]],
              "samples": samples, "pairs": pairs, "tasks": [],
              "note": "Review original roles and splits before training. Add model 2 tasks explicitly; parameters and strengths are not inferred.",
              "conversion": {"reviewer": reviewer, "pairs_without_original": unlinked}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"samples": len(samples), "pairs": len(pairs), "pairs_without_original": unlinked, "out": str(out)}
