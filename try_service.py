"""Local, in-memory model testing. PyTorch is optional for the annotation app."""
import base64
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import threading
import time

from PIL import Image

STYLE_NAMES = {"retro": "复古", "fresh": "明亮清新", "cinematic": "暗调电影感"}
MAX_UPLOAD = 25 * 1024 * 1024
MAX_PIXELS = 16_000_000
PREVIEW_SIDE = 1600


class TryService:
    def __init__(self, model_dir):
        self.root = Path(model_dir).resolve()
        self.lock = threading.Lock()
        self.entries = {}
        self.cached = None

    @staticmethod
    def fingerprint(path):
        info = path.stat()
        return (info.st_mtime_ns, info.st_size)

    def catalog(self):
        if importlib.util.find_spec("torch") is None:
            return {"available": False, "models": [], "message": "请关闭当前服务，用 start_try.bat 启动；首次使用先运行 setup_models.bat。"}
        from girlphoto_ml.models import load_checkpoint
        with self.lock:
            entries, skipped = {}, 0
            for path in sorted(self.root.rglob("*.pt"))[:80]:
                path = path.resolve()
                if not path.is_relative_to(self.root) or path.stat().st_size > 100 * 1024 * 1024:
                    skipped += 1
                    continue
                try:
                    model, bundle = load_checkpoint(path, "adjuster")
                    styles = bundle["styles"]
                    trained = bundle["supervision"]["trained_styles"]
                    if not isinstance(styles, list) or not all(isinstance(s, str) and s for s in styles):
                        raise ValueError("Invalid styles")
                    choices = [s for s in styles if s in trained]
                    if not choices or not 16 <= bundle["image_size"] <= 512:
                        raise ValueError("Missing trained styles or invalid image size")
                    stamp = self.fingerprint(path)
                    relative = path.relative_to(self.root).as_posix()
                    model_id = hashlib.sha256(f"{relative}:{stamp}".encode()).hexdigest()[:24]
                    public = {"id": model_id, "name": relative, "demo_only": bool(bundle["demo_only"]),
                              "styles": [{"id": s, "name": STYLE_NAMES.get(s, s)} for s in choices],
                              "training": "监督训练 + 模型 1 反馈" if bundle["supervision"].get("uses_scorer") else "监督训练"}
                    entries[model_id] = {"path": path, "stamp": stamp, "public": public}
                except Exception:
                    # Scorer weights and corrupt/incompatible checkpoints are not runnable adjusters.
                    skipped += 1
            self.entries = entries
            self.cached = None
            models = sorted((v["public"] for v in entries.values()), key=lambda m: (m["demo_only"], not m["name"].endswith("/model2.pt"), m["name"]))
        return {"available": bool(models), "models": models, "skipped": skipped,
                "message": "" if models else "还没有可用的模型 2 权重。先运行 demo_models.bat，或把训练得到的模型 2 放进 ml_runs 后刷新。"}

    def predict(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("请求格式无效。")
        strength = payload.get("strength")
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("强度须为 0–100% 之间的数值。")
        model_id, style = payload.get("model_id"), payload.get("style")
        if not isinstance(model_id, str) or not isinstance(style, str):
            raise ValueError("请先选择模型和风格。")
        encoded = payload.get("image")
        if not isinstance(encoded, str) or len(encoded) > MAX_UPLOAD * 1.4:
            raise ValueError("请上传小于 25 MB 的照片。")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("图片数据无法读取，请重新选择。") from exc
        if not raw or len(raw) > MAX_UPLOAD:
            raise ValueError("请上传小于 25 MB 的照片。")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format not in ("JPEG", "PNG", "WEBP") or getattr(opened, "is_animated", False):
                    raise ValueError("仅支持静态 JPEG、PNG、WebP 图片。")
                if opened.width * opened.height > MAX_PIXELS:
                    raise ValueError("网页测试支持不超过 1600 万像素的照片，请先缩小。")
                original_size = [opened.width, opened.height]
                if opened.getexif().get(274) in (5, 6, 7, 8):
                    original_size.reverse()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("图片已损坏或尺寸过大。") from exc
        if not self.lock.acquire(blocking=False):
            raise ValueError("模型正在处理另一张图片，请稍后重试。")
        try:
            entry = self.entries.get(model_id)
            if entry is None or not entry["path"].exists() or self.fingerprint(entry["path"]) != entry["stamp"]:
                raise ValueError("模型列表已变化，请刷新模型列表后重试。")
            if style not in [s["id"] for s in entry["public"]["styles"]]:
                raise ValueError("当前模型没有训练过这个风格。")
            import torch
            from girlphoto_ml.data import read_image
            from girlphoto_ml.models import load_checkpoint
            from girlphoto_ml.renderer import describe, render
            torch.set_num_threads(min(torch.get_num_threads(), 4))
            start = time.perf_counter()
            if self.cached is None or self.cached[0] != model_id:
                model, bundle = load_checkpoint(entry["path"], "adjuster")
                digest = hashlib.sha256(entry["path"].read_bytes()).hexdigest()
                self.cached = (model_id, model, bundle, digest)
            _, model, bundle, digest = self.cached
            with torch.inference_mode():
                resized = read_image(io.BytesIO(raw), size=bundle["image_size"])[None]
                parameters = model(resized, torch.tensor([bundle["styles"].index(style)]), torch.tensor([strength], dtype=torch.float32))
                before = read_image(io.BytesIO(raw), max_side=PREVIEW_SIDE)[None]
                after = render(before, parameters)
                if not torch.isfinite(parameters).all() or not torch.isfinite(after).all():
                    raise ValueError("当前模型产生了无效数值，请选择其他权重。")
                previews = []
                for tensor in (before[0], after[0]):
                    pixels = tensor.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                    buffer = io.BytesIO()
                    Image.fromarray(pixels).save(buffer, format="PNG")
                    previews.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            exported = describe(parameters[0].tolist())
            exported.update({"style": style, "strength": strength, "demo_only": bundle["demo_only"],
                             "checkpoint": entry["public"]["name"], "checkpoint_sha256": digest,
                             "source_sha256": hashlib.sha256(raw).hexdigest(), "source_size": original_size,
                             "preview_size": [before.shape[3], before.shape[2]], "preview_max_side": PREVIEW_SIDE,
                             "preview_note": "Preview is resized to at most 1600px. Use the CLI for a full-size render; resizing can cause small pixel differences."})
            return {"before_png": previews[0], "after_png": previews[1], "parameters": exported,
                    "model": entry["public"], "elapsed_ms": round((time.perf_counter()-start)*1000)}
        finally:
            self.lock.release()
