"""GirlPhoto: offline photo annotation, Python + SQLite + Pillow."""
from __future__ import annotations

import argparse
import base64
import csv
from contextlib import contextmanager
import hashlib
import io
import itertools
import json
import mimetypes
import os
from pathlib import Path
import random
import secrets
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import webbrowser
import zipfile

from PIL import Image, ImageCms, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
STYLES = [
    {"id": "retro", "name": "复古", "description": "按调色判断胶片或复古影调；不要因服装或老建筑直接归类。"},
    {"id": "fresh", "name": "明亮清新", "description": "明亮、轻盈、柔和，同时保留必要细节；不按人物长相判断。"},
    {"id": "cinematic", "name": "暗调电影感", "description": "暗调中有合理层次与色彩关系；单纯欠曝不等于电影感。"},
]
DEFECTS = ["肤色偏色", "高光细节丢失", "暗部细节丢失", "饱和度不合适", "颜色不协调", "其他"]
MAX_FILE = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def clean_text(value, label, limit=120):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{label}不能为空，且最多 {limit} 个字符。")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{label}包含不可用字符。")
    return value.strip()


def default_group(name):
    parts = name.replace("\\", "/").split("/")
    stem = Path(parts[-1]).stem
    if "__" in stem:
        return stem.split("__", 1)[0][:120]
    if len(parts) > 1:
        return parts[-2][:120]
    return stem[:120]


def split_for(group):
    number = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if number < 80 else "validation" if number < 90 else "test"


class Store:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "originals").mkdir(exist_ok=True)
        (self.root / "previews").mkdir(exist_ok=True)
        self.db_path = self.root / "annotations.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cohorts(id TEXT PRIMARY KEY, split TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS photos(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source_path TEXT NOT NULL,
                    group_id TEXT NOT NULL, cohort_id TEXT NOT NULL,
                    original TEXT NOT NULL, preview TEXT NOT NULL,
                    width INTEGER, height INTEGER, sha256 TEXT UNIQUE NOT NULL,
                    imported_at TEXT NOT NULL, is_demo INTEGER NOT NULL DEFAULT 0,
                    color_note TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS labels(
                    photo_id TEXT NOT NULL, reviewer TEXT NOT NULL, quality INTEGER,
                    styles TEXT NOT NULL, defects TEXT NOT NULL, note TEXT NOT NULL,
                    status TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(photo_id,reviewer));
                CREATE TABLE IF NOT EXISTS pairs(
                    a TEXT NOT NULL, b TEXT NOT NULL, style_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL, outcome TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(a,b,style_id,reviewer));
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, reviewer TEXT NOT NULL,
                    kind TEXT NOT NULL, key_json TEXT NOT NULL, before_json TEXT,
                    after_json TEXT NOT NULL, created_at TEXT NOT NULL, undone INTEGER DEFAULT 0);
            """)
            db.execute("INSERT OR IGNORE INTO settings VALUES ('styles',?)", (json.dumps(STYLES, ensure_ascii=False),))
            db.execute("INSERT OR IGNORE INTO settings VALUES ('schema_version','1')")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def styles(self):
        with self.connect() as db:
            return json.loads(db.execute("SELECT value FROM settings WHERE key='styles'").fetchone()[0])

    def add_style(self, name, description):
        name = clean_text(name, "风格名", 30)
        description = clean_text(description, "风格定义", 300)
        with self.lock, self.connect() as db:
            styles = json.loads(db.execute("SELECT value FROM settings WHERE key='styles'").fetchone()[0])
            if any(s["name"] == name for s in styles):
                raise ValueError("这个风格已经存在。")
            if len(styles) >= 20:
                raise ValueError("最小版最多支持 20 个风格，请先保持少量清晰的类别。")
            styles.append({"id": "s_" + secrets.token_hex(5), "name": name, "description": description})
            db.execute("UPDATE settings SET value=? WHERE key='styles'", (json.dumps(styles, ensure_ascii=False),))

    def import_photo(self, raw, name, group=None, cohort=None, demo=False):
        if not raw or len(raw) > MAX_FILE:
            raise ValueError("单张图片须小于 25 MB。")
        name = clean_text(name, "文件名", 600)
        group = clean_text(group or default_group(name), "原图组")
        cohort = clean_text(cohort or group, "拍摄批次")
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock, self.connect() as db:
            old = db.execute("SELECT id FROM photos WHERE sha256=?", (digest,)).fetchone()
            if old:
                return {"id": old[0], "duplicate": True}
            try:
                with Image.open(io.BytesIO(raw)) as opened:
                    fmt = opened.format
                    if fmt not in ("JPEG", "PNG", "WEBP"):
                        raise ValueError("仅支持 JPEG、PNG、WebP 图片。")
                    if opened.width * opened.height > 40_000_000:
                        raise ValueError("图片超过 4000 万像素，请先缩小。")
                    if getattr(opened, "is_animated", False):
                        raise ValueError("请使用静态图片。")
                    icc = opened.info.get("icc_profile")
                    picture = ImageOps.exif_transpose(opened).copy()
                color_note = "无嵌入配置，按 sRGB 显示"
                if icc:
                    try:
                        picture = ImageCms.profileToProfile(picture, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"), outputMode="RGB")
                        color_note = "预览已转换至 sRGB"
                    except (ImageCms.PyCMSError, OSError, ValueError):
                        color_note = "颜色配置无法转换，需人工检查"
                if picture.mode in ("RGBA", "LA") or "transparency" in picture.info:
                    rgba = picture.convert("RGBA")
                    backdrop = Image.new("RGBA", rgba.size, (240, 240, 240, 255))
                    picture = Image.alpha_composite(backdrop, rgba).convert("RGB")
                else:
                    picture = picture.convert("RGB")
                width, height = picture.size
                picture.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
                preview_bytes = io.BytesIO()
                picture.save(preview_bytes, format="JPEG", quality=95, subsampling=0)
            except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
                raise ValueError("图片无法读取，或尺寸过大。") from exc
            existing = db.execute("SELECT cohort_id FROM photos WHERE group_id=? LIMIT 1", (group,)).fetchone()
            if existing:
                cohort = existing[0]
            photo_id = digest[:24]
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
            original = f"originals/{photo_id}{ext}"
            preview = f"previews/{photo_id}.jpg"
            (self.root / original).write_bytes(raw)
            (self.root / preview).write_bytes(preview_bytes.getvalue())
            db.execute("INSERT OR IGNORE INTO cohorts VALUES (?,?)", (cohort, split_for(cohort)))
            db.execute("INSERT INTO photos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                photo_id, name.replace("\\", "/").split("/")[-1], name, group, cohort, original,
                preview, width, height, digest, timestamp(), int(demo), color_note))
            return {"id": photo_id, "duplicate": False}

    @staticmethod
    def unpack_label(row):
        if row is None:
            return None
        result = dict(row)
        result["styles"] = json.loads(result["styles"])
        result["defects"] = json.loads(result["defects"])
        return result

    def state(self, reviewer):
        reviewer = clean_text(reviewer, "标注者", 60)
        with self.connect() as db:
            photos = [dict(r) for r in db.execute("SELECT p.*,c.split FROM photos p JOIN cohorts c ON p.cohort_id=c.id ORDER BY imported_at,id")]
            labels = {r["photo_id"]: self.unpack_label(r) for r in db.execute("SELECT * FROM labels WHERE reviewer=?", (reviewer,))}
            for photo in photos:
                photo["label"] = labels.get(photo["id"])
                photo["url"] = "/media/" + photo["id"]
            counts = {r[0]: r[1] for r in db.execute("SELECT outcome,COUNT(*) FROM pairs WHERE reviewer=? GROUP BY outcome", (reviewer,))}
            return {"photos": photos, "styles": self.styles(), "defects": DEFECTS, "pair_counts": counts, "data_dir": str(self.root)}

    def event(self, db, reviewer, kind, key, before, after):
        db.execute("INSERT INTO events(reviewer,kind,key_json,before_json,after_json,created_at) VALUES (?,?,?,?,?,?)", (
            reviewer, kind, json.dumps(key), json.dumps(dict(before)) if before else None,
            json.dumps(after, ensure_ascii=False), timestamp()))

    def label(self, payload):
        reviewer = clean_text(payload.get("reviewer"), "标注者", 60)
        photo_id = payload.get("photo_id")
        quality = payload.get("quality")
        if quality is not None and (type(quality) is not int or quality not in (1, 2, 3)):
            raise ValueError("质量仅可为 1、2、3 或未评分。")
        status = payload.get("status", "labeled")
        if status not in ("labeled", "skipped"):
            raise ValueError("无效标注状态。")
        style_values = payload.get("styles", {})
        style_ids = {s["id"] for s in self.styles()}
        if not isinstance(style_values, dict) or any(k not in style_ids or v not in ("yes", "no", "unknown") for k, v in style_values.items()):
            raise ValueError("风格标注无效。")
        # Missing means unreviewed; unknown means explicitly uncertain. Never turn either into a negative.
        defects = payload.get("defects", [])
        if not isinstance(defects, list) or any(d not in DEFECTS for d in defects):
            raise ValueError("问题标签无效。")
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > 1000:
            raise ValueError("备注最多 1000 字。")
        if status == "labeled" and quality is None and not style_values:
            raise ValueError("至少标一项质量或风格；暂时无法判断可点跳过。")
        with self.lock, self.connect() as db:
            if not db.execute("SELECT 1 FROM photos WHERE id=?", (photo_id,)).fetchone():
                raise ValueError("照片不存在。")
            before = db.execute("SELECT * FROM labels WHERE photo_id=? AND reviewer=?", (photo_id, reviewer)).fetchone()
            values = (photo_id, reviewer, quality, json.dumps(style_values), json.dumps(defects, ensure_ascii=False), note, status, timestamp())
            db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?)", values)
            self.event(db, reviewer, "label", [photo_id, reviewer], before, payload)

    def update_group(self, payload):
        group = clean_text(payload.get("group_id"), "原图组")
        cohort = clean_text(payload.get("cohort_id"), "拍摄批次")
        split = payload.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("无效数据集划分。")
        with self.lock, self.connect() as db:
            photo = db.execute("SELECT * FROM photos WHERE id=?", (payload.get("photo_id"),)).fetchone()
            if not photo:
                raise ValueError("照片不存在。")
            if db.execute("SELECT 1 FROM pairs WHERE a=? OR b=?", (photo["id"], photo["id"])).fetchone():
                raise ValueError("该照片已有对比记录，请先撤销相关对比，再改变原图组。")
            target = db.execute("SELECT cohort_id FROM photos WHERE group_id=? AND id!=? LIMIT 1", (group, photo["id"])).fetchone()
            if target:
                cohort = target[0]
                split = db.execute("SELECT split FROM cohorts WHERE id=?", (cohort,)).fetchone()[0]
            elif db.execute("SELECT 1 FROM cohorts WHERE id=?", (cohort,)).fetchone():
                split = db.execute("SELECT split FROM cohorts WHERE id=?", (cohort,)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO cohorts VALUES (?,?)", (cohort, split))
            db.execute("UPDATE photos SET group_id=?,cohort_id=? WHERE id=?", (group, cohort, photo["id"]))

    def update_split(self, payload):
        cohort = clean_text(payload.get("cohort_id"), "拍摄批次")
        split = payload.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("无效划分。")
        with self.lock, self.connect() as db:
            if not db.execute("UPDATE cohorts SET split=? WHERE id=?", (split, cohort)).rowcount:
                raise ValueError("批次不存在。")

    def pair_next(self, reviewer, style_id, include_skipped=False):
        if style_id not in {s["id"] for s in self.styles()}:
            raise ValueError("请先选择目标风格。")
        state = self.state(reviewer)
        groups = {}
        for photo in state["photos"]:
            label = photo["label"]
            if label and label["status"] == "labeled" and label["styles"].get(style_id) == "yes":
                groups.setdefault(photo["group_id"], []).append(photo)
        with self.connect() as db:
            done = {(r["a"], r["b"]) for r in db.execute("SELECT * FROM pairs WHERE style_id=? AND reviewer=?", (style_id, reviewer)) if not (include_skipped and r["outcome"] == "skip")}
        chosen, total = None, 0
        # Reservoir sampling avoids materializing an unbounded O(n^2) pair list.
        for photos in groups.values():
            for a, b in itertools.combinations(photos, 2):
                key = tuple(sorted((a["id"], b["id"])))
                if key in done:
                    continue
                total += 1
                if random.randrange(total) == 0:
                    chosen = [a, b]
        if chosen:
            random.shuffle(chosen)
        return {"pair": chosen, "remaining": total}

    def save_pair(self, payload):
        reviewer = clean_text(payload.get("reviewer"), "标注者", 60)
        left, right = payload.get("left"), payload.get("right")
        style = payload.get("style_id")
        outcome = payload.get("outcome")
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise ValueError("请对比两个不同版本。")
        if outcome not in ("left", "right", "tie", "skip"):
            raise ValueError("无效比较结果。")
        if style not in {s["id"] for s in self.styles()}:
            raise ValueError("无效目标风格。")
        a, b = sorted((left, right))
        normalized = ("a" if (left if outcome == "left" else right) == a else "b") if outcome in ("left", "right") else outcome
        with self.lock, self.connect() as db:
            rows = list(db.execute("SELECT * FROM photos WHERE id IN (?,?)", (a, b)))
            if len(rows) != 2 or rows[0]["group_id"] != rows[1]["group_id"]:
                raise ValueError("只能比较同一原图组的两个版本。")
            for photo_id in (a, b):
                label = db.execute("SELECT * FROM labels WHERE photo_id=? AND reviewer=?", (photo_id, reviewer)).fetchone()
                if not label or label["status"] != "labeled" or json.loads(label["styles"]).get(style) != "yes":
                    raise ValueError("请先确认两张图片都符合本次目标风格。")
            key = (a, b, style, reviewer)
            before = db.execute("SELECT * FROM pairs WHERE a=? AND b=? AND style_id=? AND reviewer=?", key).fetchone()
            db.execute("INSERT OR REPLACE INTO pairs VALUES (?,?,?,?,?,?)", (*key, normalized, timestamp()))
            self.event(db, reviewer, "pair", key, before, payload)

    def undo(self, reviewer):
        reviewer = clean_text(reviewer, "标注者", 60)
        with self.lock, self.connect() as db:
            event = db.execute("SELECT * FROM events WHERE reviewer=? AND undone=0 ORDER BY id DESC LIMIT 1", (reviewer,)).fetchone()
            if not event:
                return {"undone": False}
            key = json.loads(event["key_json"])
            before = json.loads(event["before_json"]) if event["before_json"] else None
            if event["kind"] == "label":
                db.execute("DELETE FROM labels WHERE photo_id=? AND reviewer=?", key)
                if before:
                    db.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?,?)", tuple(before[k] for k in ("photo_id", "reviewer", "quality", "styles", "defects", "note", "status", "updated_at")))
            else:
                db.execute("DELETE FROM pairs WHERE a=? AND b=? AND style_id=? AND reviewer=?", key)
                if before:
                    db.execute("INSERT INTO pairs VALUES (?,?,?,?,?,?)", tuple(before[k] for k in ("a", "b", "style_id", "reviewer", "outcome", "updated_at")))
            db.execute("UPDATE events SET undone=1 WHERE id=?", (event["id"],))
            return {"undone": True, "kind": event["kind"], "key": key}

    def export(self):
        with self.lock, self.connect() as db:
            db.execute("BEGIN")
            photos = [dict(r) for r in db.execute("SELECT p.*,c.split FROM photos p JOIN cohorts c ON p.cohort_id=c.id")]
            labels = [self.unpack_label(r) for r in db.execute("SELECT * FROM labels")]
            pairs = [dict(r) for r in db.execute("SELECT * FROM pairs")]
            events = [dict(r) for r in db.execute("SELECT * FROM events")]
            styles = json.loads(db.execute("SELECT value FROM settings WHERE key='styles'").fetchone()[0])
        ids = {p["id"]: p for p in photos}
        # An old comparison whose style labels were later changed is preserved, but masked out of training.
        label_index = {(r["photo_id"], r["reviewer"]): r for r in labels}
        for pair in pairs:
            pair["group_id"] = ids[pair["a"]]["group_id"]
            pair["split"] = ids[pair["a"]]["split"]
            pair["eligible"] = pair["outcome"] != "skip" and all(
                (label_index.get((p, pair["reviewer"])) or {}).get("status") == "labeled" and
                (label_index.get((p, pair["reviewer"])) or {}).get("styles", {}).get(pair["style_id"]) == "yes"
                for p in (pair["a"], pair["b"]))
            pair["is_demo"] = bool(ids[pair["a"]]["is_demo"] or ids[pair["b"]]["is_demo"])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            bundle = {"schema_version": 1, "exported_at": timestamp(), "styles": styles,
                      "photos": photos, "labels": labels, "pairs": pairs, "events": events}
            archive.writestr("dataset.json", json.dumps(bundle, ensure_ascii=False, indent=2))
            def write_csv(name, rows, fields):
                stream = io.StringIO(newline="")
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    cooked = {}
                    for key in fields:
                        value = row.get(key, "")
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value, ensure_ascii=False)
                        # Spreadsheet formula protection. dataset.json keeps the original exact strings.
                        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                            value = "'" + value
                        cooked[key] = value
                    writer.writerow(cooked)
                archive.writestr(name, stream.getvalue().encode("utf-8-sig"))
            write_csv("photos.csv", photos, ["id", "name", "group_id", "cohort_id", "split", "original", "preview", "sha256", "width", "height", "is_demo", "color_note"])
            write_csv("quality.csv", labels, ["photo_id", "reviewer", "quality", "status", "defects", "note", "updated_at"])
            style_rows = [{"photo_id": r["photo_id"], "reviewer": r["reviewer"], "style_id": s["id"],
                           "value": r["styles"].get(s["id"], "unreviewed"), "status": r["status"]}
                          for r in labels for s in styles]
            write_csv("styles.csv", style_rows, ["photo_id", "reviewer", "style_id", "value", "status"])
            write_csv("preferences.csv", pairs, ["a", "b", "style_id", "reviewer", "outcome", "group_id", "split", "eligible", "is_demo", "updated_at"])
            archive.writestr("README.txt", "导出不含照片文件，请同时备份应用 data 目录。\n原始精确数据在 dataset.json；CSV 对公式开头文本加了单引号。\n风格 unknown 和 unreviewed 都不是负例。质量 null 和 status=skipped 不作为评分。\n比较 a/b 是规范编号，不是屏幕左右；tie 是平局，skip 是跳过。\n只使用 eligible=true 的比较。is_demo=true 为合成流程演示，不用于真实训练。\n按 cohort_id 保持划分；验证与测试不用于训练。\n")
        return buffer.getvalue()

    def demo(self):
        made = 0
        for scene in range(2):
            for variant, tint in enumerate(((12, 0, -12), (-5, 3, 9), (30, 8, -26))):
                image = Image.new("RGB", (700, 900))
                draw = ImageDraw.Draw(image)
                for y in range(900):
                    color = tuple(max(0, min(255, int(v + tint[i]))) for i, v in enumerate((105 + y / 12, 145 + y / 20, 170 - y / 18)))
                    draw.line((0, y, 700, y), fill=color)
                draw.rectangle((60, 440, 250, 770), fill=(68 + variant * 10, 78, 74))
                draw.ellipse((260, 180, 470, 390), fill=(205 + variant * 10, 170, 145))
                draw.rounded_rectangle((240, 370, 500, 820), radius=90, fill=(190, 185 - scene * 25, 163))
                draw.text((35, 30), "DEMO / SYNTHETIC - NOT TRAINING DATA", fill="white", stroke_width=1)
                out = io.BytesIO()
                image.save(out, format="PNG")
                result = self.import_photo(out.getvalue(), f"demo{scene+1}__v{variant+1}.png", cohort=f"demo-session-{scene+1}", demo=True)
                made += not result["duplicate"]
        return {"created": made}


def make_server(store, port=8765, model_dir=None):
    token = secrets.token_urlsafe(32)
    from try_service import TryService
    tester = TryService(model_dir or ROOT / "ml_runs")

    class Handler(BaseHTTPRequestHandler):
        server_version = "GirlPhoto/0.2"

        def log_message(self, fmt, *args):
            # Do not put original filenames, notes, or query strings in logs.
            pass

        def send_bytes(self, body, mime, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def host_ok(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.host_ok():
                return self.json({"error": "仅允许本机访问。"}, 403)
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/api/try/models":
                    result = tester.catalog()
                    result["token"] = token
                    return self.json(result)
                if url.path == "/api/state":
                    result = store.state(query.get("reviewer", ["我"])[0])
                    result["token"] = token
                    return self.json(result)
                if url.path == "/api/pair":
                    return self.json(store.pair_next(query.get("reviewer", ["我"])[0], query.get("style", [""])[0], query.get("retry", ["0"])[0] == "1"))
                if url.path == "/api/export":
                    return self.send_bytes(store.export(), "application/zip", filename="girlphoto-labels.zip")
                if url.path.startswith("/media/"):
                    photo_id = url.path.removeprefix("/media/")
                    with store.connect() as db:
                        row = db.execute("SELECT preview FROM photos WHERE id=?", (photo_id,)).fetchone()
                    if not row:
                        return self.json({"error": "照片不存在。"}, 404)
                    return self.send_bytes((store.root / row[0]).read_bytes(), "image/jpeg")
                static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                          "/try": "try.html", "/try.js": "try.js", "/try.css": "try.css"}
                if url.path in static:
                    path = ROOT / "static" / static[url.path]
                    return self.send_bytes(path.read_bytes(), (mimetypes.guess_type(path)[0] or "text/plain") + "; charset=utf-8")
                return self.json({"error": "页面不存在。"}, 404)
            except ValueError as exc:
                return self.json({"error": str(exc)}, 400)
            except Exception:
                return self.json({"error": "读取失败，请检查数据目录和图片文件。"}, 500)

        def do_POST(self):
            if not self.host_ok() or self.headers.get("X-Girlphoto-Token") != token:
                return self.json({"error": "请求校验失败，请刷新页面。"}, 403)
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                return self.json({"error": "来源校验失败。"}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_FILE * 1.4:
                    raise ValueError("请求过大或为空。")
                payload = json.loads(self.rfile.read(size))
                path = urlparse(self.path).path
                if path == "/api/try/predict":
                    return self.json(tester.predict(payload))
                if path == "/api/import":
                    raw = base64.b64decode(payload["data"], validate=True)
                    return self.json(store.import_photo(raw, payload["name"], payload.get("group_id"), payload.get("cohort_id")))
                if path == "/api/label":
                    store.label(payload)
                elif path == "/api/pair":
                    store.save_pair(payload)
                elif path == "/api/group":
                    store.update_group(payload)
                elif path == "/api/split":
                    store.update_split(payload)
                elif path == "/api/style":
                    store.add_style(payload.get("name"), payload.get("description"))
                elif path == "/api/undo":
                    return self.json(store.undo(payload.get("reviewer")))
                elif path == "/api/demo":
                    return self.json(store.demo())
                else:
                    return self.json({"error": "接口不存在。"}, 404)
                return self.json({"ok": True})
            except (ValueError, KeyError, TypeError) as exc:
                return self.json({"error": str(exc)}, 400)
            except Exception:
                return self.json({"error": "保存失败。已有记录未丢失，请重试或查看数据目录。"}, 500)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="本地照片标注工具；不连接外网。")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--open-try", action="store_true", help="打开模型调色测试网页")
    args = parser.parse_args()
    store = Store(args.data_dir)
    try:
        server = make_server(store, args.port)
    except OSError:
        print(f"端口 {args.port} 不可用。请尝试 python app.py --port 8766")
        return 1
    url = f"http://127.0.0.1:{server.server_port}" + ("/try" if args.open_try else "")
    print(f"GirlPhoto 已启动：{url}\n数据保存到：{store.root}\n关闭此窗口或按 Ctrl+C 停止。", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
