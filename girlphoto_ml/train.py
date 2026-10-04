import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .data import Manifest
from .models import Adjuster, Scorer, load_checkpoint, save_checkpoint
from .renderer import render


def setup(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")


def image_batch(data, ids, size, device):
    return torch.stack([data.image(sid, size) for sid in ids]).to(device)


def sample_loss(model, images, samples, styles):
    prediction = model(images)
    terms = []
    for index, style in enumerate(styles):
        known = [i for i, s in enumerate(samples) if s.get("styles", {}).get(style) in ("yes", "no")]
        if known:
            target = images.new_tensor([float(samples[i]["styles"][style] == "yes") for i in known])
            terms.append(F.binary_cross_entropy_with_logits(prediction["style_logits"][known, index], target))
        strong = [i for i, s in enumerate(samples) if style in s.get("strengths", {})]
        if strong:
            target = images.new_tensor([samples[i]["strengths"][style] for i in strong])
            terms.append(F.mse_loss(prediction["strength"][strong, index], target))
    rated = [i for i, s in enumerate(samples) if s.get("quality") is not None]
    if rated:
        target = torch.tensor([samples[i]["quality"]-1 for i in rated], dtype=torch.long, device=images.device)
        terms.append(F.cross_entropy(prediction["quality_logits"][rated], target))
    return sum(terms) / len(terms) if terms else None


def preference_loss(model, data, pairs, size, device):
    source = image_batch(data, [p["source"] for p in pairs], size, device)
    left = image_batch(data, [p["left"] for p in pairs], size, device)
    right = image_batch(data, [p["right"] for p in pairs], size, device)
    style = torch.tensor([data.styles.index(p["style"]) for p in pairs], device=device)
    difference = model.rank(source, left, style) - model.rank(source, right, style)
    target = difference.new_tensor([{"left": 1., "right": 0., "tie": .5}[p["outcome"]] for p in pairs])
    return F.binary_cross_entropy_with_logits(difference, target), difference, target


def supervision_counts(data, samples, pairs):
    return {"quality": sum(s.get("quality") is not None for s in samples), "styles": {
        style: {"yes": sum(s.get("styles", {}).get(style) == "yes" for s in samples),
                "no": sum(s.get("styles", {}).get(style) == "no" for s in samples),
                "strength": sum(style in s.get("strengths", {}) for s in samples),
                "preferences": sum(p["style"] == style and p["outcome"] != "tie" for p in pairs)}
        for style in data.styles}}


@torch.no_grad()
def evaluate_scorer(model, data, size, device, split="validation"):
    model.eval()
    total_loss, count = 0., 0
    style_correct, style_count, quality_correct, quality_count = 0, 0, 0, 0
    for sample in data.select(split):
        images = image_batch(data, [sample["id"]], size, device)
        loss = sample_loss(model, images, [sample], data.styles)
        if loss is not None:
            total_loss += float(loss)
            count += 1
        prediction = model(images)
        for i, style in enumerate(data.styles):
            label = sample.get("styles", {}).get(style)
            if label in ("yes", "no"):
                style_correct += int((prediction["style_logits"][0, i] >= 0).item() == (label == "yes"))
                style_count += 1
        if sample.get("quality") is not None:
            quality_correct += int(prediction["quality_logits"][0].argmax().item()+1 == sample["quality"])
            quality_count += 1
    pairs = [p for p in data.pairs if data.ids[p["source"]]["split"] == split]
    pair_correct, pair_count, losses = 0, 0, []
    for pair in pairs:
        loss, difference, target = preference_loss(model, data, [pair], size, device)
        losses.append(float(loss))
        if pair["outcome"] != "tie":
            pair_correct += int((difference[0] > 0).item() == (target[0] == 1).item())
            pair_count += 1
    return {"split": split, "labeled_samples": count, "sample_loss": total_loss/count if count else None,
            "style_accuracy": style_correct/style_count if style_count else None,
            "quality_accuracy": quality_correct/quality_count if quality_count else None,
            "preference_accuracy": pair_correct/pair_count if pair_count else None,
            "preference_loss": sum(losses)/len(losses) if losses else None, "preference_count": pair_count}


def train_scorer(manifest, out, steps=100, batch_size=8, size=96, seed=17, device="cpu"):
    setup(seed, device)
    data = Manifest(manifest)
    samples = [s for s in data.select("train") if s.get("quality") is not None or
               any(v in ("yes", "no") for v in s.get("styles", {}).values())]
    pairs = [p for p in data.pairs if data.ids[p["source"]]["split"] == "train"]
    if not samples and not pairs:
        raise ValueError("No usable training labels or preference pairs.")
    model = Scorer(len(data.styles)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        terms = []
        if samples:
            selected = random.sample(samples, min(batch_size, len(samples)))
            terms.append(sample_loss(model, image_batch(data, [s["id"] for s in selected], size, device), selected, data.styles))
        if pairs:
            selected_pairs = random.sample(pairs, min(batch_size, len(pairs)))
            terms.append(preference_loss(model, data, selected_pairs, size, device)[0])
        loss = sum(terms)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step == 0 or (step+1) % 20 == 0 or step+1 == steps:
            print(f"Model 1 step {step+1}/{steps}: loss={history[-1]:.4f}", flush=True)
    counts = supervision_counts(data, samples, pairs)
    save_checkpoint(out, model, "scorer", data.styles, size, counts, data.demo_only)
    metrics = evaluate_scorer(model, data, size, device)
    write_report(out, {"loss_history": history, "validation": metrics, "supervision": counts, "demo_only": data.demo_only})
    return metrics


def feedback_loss(scorer, bundle, source, output, styles, strength):
    prediction = scorer(output)
    terms = []
    for i, style_id in enumerate(styles.tolist()):
        style = bundle["styles"][style_id]
        counts = bundle["supervision"]["styles"][style]
        if counts["yes"] and counts["no"]:
            terms.append(.3 * F.softplus(-prediction["style_logits"][i, style_id]))
        if bundle["supervision"]["quality"]:
            expected = (prediction["quality_logits"][i].softmax(0) * output.new_tensor([0., .5, 1.])).sum()
            terms.append(.3 * (1-expected))
        if counts["strength"]:
            terms.append(.2 * (prediction["strength"][i, style_id] - strength[i]).square())
    if terms:
        scalar = sum(terms) / len(styles)
    else:
        scalar = output.sum() * 0
    relative = scorer.rank(source, output, styles) - scorer.rank(source, source, styles)
    return scalar + F.softplus(-relative).mean()


def task_loss(model, data, tasks, size, device, scorer=None, bundle=None, feedback_weight=.2):
    source = image_batch(data, [t["source"] for t in tasks], size, device)
    style_ids = torch.tensor([data.styles.index(t["style"]) for t in tasks], device=device)
    strength = source.new_tensor([t["strength"] for t in tasks])
    parameters = model(source, style_ids, strength)
    output = render(source, parameters)
    terms = []
    labeled = [i for i, t in enumerate(tasks) if t.get("parameters") is not None]
    if labeled:
        target = source.new_tensor([tasks[i]["parameters"] for i in labeled])
        terms.append(F.mse_loss(parameters[labeled], target))
    paired = [i for i, t in enumerate(tasks) if t.get("target")]
    if paired:
        target = image_batch(data, [tasks[i]["target"] for i in paired], size, device)
        terms.append(F.l1_loss(output[paired], target))
    if scorer is not None:
        terms.append(feedback_weight * feedback_loss(scorer, bundle, source, output, style_ids, strength))
    terms += [.01*parameters.square().mean(), .02*F.l1_loss(output, source)]
    return sum(terms)


def train_adjuster(manifest, out, scorer_path=None, steps=100, batch_size=8, size=96, seed=17, device="cpu", feedback_weight=.2):
    setup(seed, device)
    data = Manifest(manifest)
    tasks = [t for t in data.tasks if data.ids[t["source"]]["split"] == "train"]
    if not tasks:
        raise ValueError("No model 2 training tasks; add explicit original/style/strength tasks.")
    scorer, bundle = None, None
    if scorer_path:
        scorer, bundle = load_checkpoint(scorer_path, "scorer", device)
        if bundle["styles"] != data.styles or bundle["image_size"] != size:
            raise ValueError("Model 1 styles and image size must match model 2 training.")
        if bundle["demo_only"] and not data.demo_only:
            raise ValueError("A synthetic-demo scorer cannot guide real-photo training.")
        for style in {t["style"] for t in tasks}:
            if not bundle["supervision"]["styles"][style]["preferences"]:
                raise ValueError(f"Model 1 has no non-tie preference training for {style}; use supervised tasks without --scorer first.")
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
    elif any(t.get("parameters") is None and not t.get("target") for t in tasks):
        raise ValueError("Without --scorer, every training task needs target parameters or a target image.")
    model = Adjuster(len(data.styles)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for step in range(steps):
        selected = random.sample(tasks, min(batch_size, len(tasks)))
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model, data, selected, size, device, scorer, bundle, feedback_weight)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step == 0 or (step+1) % 20 == 0 or step+1 == steps:
            print(f"Model 2 step {step+1}/{steps}: loss={history[-1]:.4f}", flush=True)
    trained_styles = sorted({t["style"] for t in tasks})
    save_checkpoint(out, model, "adjuster", data.styles, size, {"tasks": len(tasks), "trained_styles": trained_styles,
                    "uses_scorer": bool(scorer_path)}, data.demo_only)
    model.eval()
    validation = [t for t in data.tasks if data.ids[t["source"]]["split"] == "validation" and
                  (t.get("parameters") is not None or t.get("target"))]
    with torch.no_grad():
        losses = [float(task_loss(model, data, [t], size, device)) for t in validation]
    report = {"loss_history": history, "validation_tasks": len(losses),
              "validation_supervised_objective": sum(losses)/len(losses) if losses else None,
              "demo_only": data.demo_only, "note": "Low objective loss is not evidence of human aesthetic quality."}
    write_report(out, report)
    return report


def write_report(checkpoint, value):
    Path(str(checkpoint)+".metrics.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
