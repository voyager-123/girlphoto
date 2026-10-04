import argparse
import json
from pathlib import Path

import torch

from .convert import convert_export
from .data import Manifest, read_image, write_image
from .demo import make_demo
from .models import load_checkpoint
from .renderer import describe, render
from .train import evaluate_scorer, train_adjuster, train_scorer


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def predict(checkpoint, image, style, strength, out):
    if not 0 <= strength <= 1:
        raise ValueError("Strength must be in [0,1].")
    model, bundle = load_checkpoint(checkpoint, "adjuster")
    if style not in bundle["styles"] or style not in bundle["supervision"]["trained_styles"]:
        raise ValueError("This checkpoint was not trained for the selected style.")
    out = Path(out).resolve()
    if out.suffix.lower() != ".png":
        raise ValueError("Use a .png output filename to preserve the rendered RGB values.")
    if out == Path(image).resolve():
        raise ValueError("Choose a different output path; never overwrite the source photo.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        resized = read_image(image, bundle["image_size"])[None]
        p = model(resized, torch.tensor([bundle["styles"].index(style)]), torch.tensor([strength], dtype=torch.float32))
        full = read_image(image)[None]
        # A global transform allows striped rendering, avoiding a full-resolution graph.
        result = torch.empty_like(full)
        for y in range(0, full.shape[2], 256):
            result[:, :, y:y+256] = render(full[:, :, y:y+256], p)
        write_image(result[0], out)
    parameters = describe(p[0].tolist())
    parameters.update({"style": style, "strength": strength, "demo_only": bundle["demo_only"],
                       "checkpoint": str(Path(checkpoint).resolve())})
    out.with_suffix(".parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"image": str(out), "parameters": str(out.with_suffix('.parameters.json')), "demo_only": bundle["demo_only"]}


def main():
    parser = argparse.ArgumentParser(description="GirlPhoto models: isolated PyTorch baseline, not a pretrained beauty filter.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train1", "train2"):
        command = sub.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--steps", type=positive, default=100)
        command.add_argument("--batch-size", type=positive, default=8)
        command.add_argument("--size", type=positive, default=96)
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        if name == "train2":
            command.add_argument("--scorer")
            command.add_argument("--feedback-weight", type=float, default=.2)
    command = sub.add_parser("predict")
    command.add_argument("--model", required=True)
    command.add_argument("--image", required=True)
    command.add_argument("--style", required=True)
    command.add_argument("--strength", type=float, default=.7)
    command.add_argument("--out", required=True)
    command = sub.add_parser("score")
    command.add_argument("--model", required=True)
    command.add_argument("--image", required=True)
    command = sub.add_parser("evaluate1")
    command.add_argument("--model", required=True)
    command.add_argument("--data", required=True)
    command.add_argument("--split", choices=["validation", "test"], default="validation")
    command = sub.add_parser("check-data")
    command.add_argument("--data", required=True)
    command = sub.add_parser("convert-export")
    command.add_argument("--export", required=True)
    command.add_argument("--data-dir", required=True)
    command.add_argument("--reviewer", required=True)
    command.add_argument("--original-marker", help="Explicit filename stem suffix identifying the original, e.g. __original")
    command.add_argument("--out", required=True)
    command = sub.add_parser("demo")
    command.add_argument("--out", default="ml_runs/demo")
    command.add_argument("--steps", type=positive, default=30)
    args = parser.parse_args()
    try:
        if args.command in ("train1", "train2"):
            if args.size < 16:
                raise ValueError("Image size must be at least 16.")
            kwargs = dict(manifest=args.data, out=args.out, steps=args.steps, batch_size=args.batch_size,
                          size=args.size, seed=args.seed, device=args.device)
            if args.command == "train1":
                result = train_scorer(**kwargs)
            else:
                if not 0 <= args.feedback_weight <= 10:
                    raise ValueError("Feedback weight must be between 0 and 10.")
                if args.scorer and args.feedback_weight == 0:
                    raise ValueError("Omit --scorer for supervised-only training; feedback weight must be positive when a scorer is used.")
                result = train_adjuster(**kwargs, scorer_path=args.scorer, feedback_weight=args.feedback_weight)
        elif args.command == "predict":
            result = predict(args.model, args.image, args.style, args.strength, args.out)
        elif args.command in ("score", "evaluate1"):
            model, bundle = load_checkpoint(args.model, "scorer")
            if args.command == "evaluate1":
                data = Manifest(args.data)
                if data.styles != bundle["styles"]:
                    raise ValueError("Manifest style order differs from checkpoint.")
                result = evaluate_scorer(model, data, bundle["image_size"], "cpu", args.split)
            else:
                with torch.no_grad():
                    prediction = model(read_image(args.image, bundle["image_size"])[None])
                counts = bundle["supervision"]
                result = {"style_probabilities": {s: float(prediction["style_logits"][0, i].sigmoid())
                          if counts["styles"][s]["yes"] and counts["styles"][s]["no"] else None for i, s in enumerate(bundle["styles"])},
                          "quality_probabilities_1_2_3": prediction["quality_logits"][0].softmax(0).tolist() if counts["quality"] else None,
                          "strength": {s: float(prediction["strength"][0, i]) if counts["styles"][s]["strength"] else None for i, s in enumerate(bundle["styles"])},
                          "demo_only": bundle["demo_only"], "note": "Probabilities are model estimates, not calibrated confidence; null means no sufficient training labels."}
        elif args.command == "check-data":
            data = Manifest(args.data)
            result = {"samples": len(data.samples), "pairs": len(data.pairs), "tasks": len(data.tasks),
                      "splits": {s: len(data.select(s)) for s in ("train", "validation", "test")}, "demo_only": data.demo_only}
        elif args.command == "convert-export":
            result = convert_export(args.export, args.data_dir, args.reviewer, args.out, args.original_marker)
        else:
            folder = Path(args.out).resolve()
            if (folder / "model1.pt").exists() or (folder / "model2.pt").exists():
                raise ValueError("Demo output already contains models. Choose a new --out directory to keep earlier results.")
            manifest = make_demo(folder)
            train_scorer(manifest, folder/"model1.pt", steps=args.steps, size=48, batch_size=4)
            train_adjuster(manifest, folder/"model2.pt", scorer_path=folder/"model1.pt", steps=args.steps, size=48, batch_size=4)
            result = predict(folder/"model2.pt", folder/"images/g4_original.png", "retro", .65, folder/"preview.png")
            result["warning"] = "Synthetic end-to-end test only. These weights have NOT learned human aesthetics."
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
