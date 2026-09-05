"""Local web UI for manually labeling collected CAPTCHA images."""

from __future__ import annotations

import argparse
import csv
import html
import mimetypes
import os
import re
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import ddddocr

from collector import make_ocr


LABEL_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MANIFEST_FIELDS = [
    "timestamp",
    "filename",
    "status",
    "predicted_label",
    "ocr_source",
    "verified",
    "sha256",
    "bytes",
    "verification_response",
]


def model_top3(ocr: ddddocr.DdddOcr, image_bytes: bytes) -> list[tuple[str, float]]:
    result = ocr.classification(image_bytes, probability=True)
    probabilities = result["probabilities"]
    charset = result["charset"]
    beam: dict[tuple[str, ...], tuple[float, float]] = {(): (1.0, 0.0)}

    # CTC prefix beam search. This is intentionally small because it is a
    # review aid, not a replacement for the collector's server verification.
    for timestep in probabilities:
        distribution = timestep[0] if len(timestep) == 1 else timestep
        top_indices = sorted(
            range(len(distribution)), key=distribution.__getitem__, reverse=True
        )[:20]
        next_beam: dict[tuple[str, ...], list[float]] = {}
        for prefix, (blank_score, text_score) in beam.items():
            for index in top_indices:
                probability = float(distribution[index])
                if probability <= 0:
                    continue
                if index == 0:
                    scores = next_beam.setdefault(prefix, [0.0, 0.0])
                    scores[0] += (blank_score + text_score) * probability
                    continue
                character = charset[index]
                if not (character.isascii() and character.isalnum()):
                    continue
                if prefix and prefix[-1] == character:
                    same_scores = next_beam.setdefault(prefix, [0.0, 0.0])
                    same_scores[1] += text_score * probability
                    extended_scores = next_beam.setdefault(prefix + (character,), [0.0, 0.0])
                    extended_scores[1] += blank_score * probability
                else:
                    extended_scores = next_beam.setdefault(prefix + (character,), [0.0, 0.0])
                    extended_scores[1] += (blank_score + text_score) * probability
        beam = {
            prefix: (scores[0], scores[1])
            for prefix, scores in sorted(
                next_beam.items(), key=lambda item: sum(item[1]), reverse=True
            )[:20]
        }

    candidates: dict[str, float] = {}
    for prefix, scores in beam.items():
        if not prefix:
            continue
        label = "".join(prefix).lower()
        if LABEL_RE.fullmatch(label):
            candidates[label] = max(candidates.get(label, 0.0), sum(scores))
    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:3]


def safe_filename(value: str) -> str:
    path = Path(value)
    if path.name != value or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("invalid image filename")
    return value


def labeled_filename(filename: str, label: str) -> str:
    path = Path(filename)
    return f"{path.stem}_{label}{path.suffix}"


def update_manifest(root: Path, old_relative: str, new_relative: str, label: str) -> None:
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    changed = False
    for row in rows:
        if row.get("filename") == old_relative:
            row["filename"] = new_relative
            row["status"] = "manually_reviewed"
            row["predicted_label"] = label
            row["verified"] = "false"
            changed = True
            break
    if not changed:
        return

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=manifest.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, manifest)


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.render_index()
            return
        if parsed.path == "/image":
            filename = parse_qs(parsed.query).get("name", [""])[0]
            self.serve_image(filename)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        fields = parse_qs(self.rfile.read(length).decode("utf-8"))
        filename = fields.get("filename", [""])[0]
        action = fields.get("action", ["review"])[0]
        if action == "skip":
            self.redirect("/")
            return
        label = fields.get("label", [""])[0].strip().lower()
        try:
            safe_filename(filename)
            if not LABEL_RE.fullmatch(label):
                raise ValueError("label must contain 1-32 letters or digits")
            source = self.server.review_dir / filename
            if not source.is_file():
                raise FileNotFoundError(filename)
            target_name = labeled_filename(filename, label)
            target = self.server.reviewed_dir / target_name
            if target.exists():
                raise FileExistsError(target_name)
            source.replace(target)
            update_manifest(
                self.server.root,
                f"review/{filename}",
                f"reviewed/{target_name}",
                label,
            )
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            self.render_index(error=str(exc), selected=filename)
            return
        self.redirect("/")

    def render_index(self, error: str = "", selected: str = "") -> None:
        files = sorted(
            path.name
            for path in self.server.review_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        current = selected if selected in files else (files[0] if files else "")
        escaped_current = html.escape(current, quote=True)
        top3_html = ""
        if current:
            try:
                candidates = model_top3(
                    self.server.ocr,
                    (self.server.review_dir / current).read_bytes(),
                )
                top3_html = "<p>当前模型 Top3：</p><ol>" + "".join(
                    f"<li><code>{html.escape(label)}</code> "
                    f"<span class=\"meta\">score={score:.4g}</span></li>"
                    for label, score in candidates
                ) + "</ol>"
            except Exception as exc:
                top3_html = f'<p class="error">模型推理失败：{html.escape(str(exc))}</p>'
        body = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>验证码人工复核</title>
  <style>
    body { font-family: sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
    img { image-rendering: pixelated; max-width: 100%; border: 1px solid #ccc; background: #eee; }
    input { font-size: 1.2rem; padding: .45rem; width: 12rem; }
    button { font-size: 1rem; padding: .5rem 1rem; margin-right: .5rem; }
    .error { color: #b00020; }
    .meta { color: #666; margin: .7rem 0; }
  </style>
</head>
<body>
  <h1>验证码人工复核</h1>
  <p class="meta">待复核：{count} 张；完成后移动到独立的 <code>reviewed/</code> 目录。</p>
  {message}
  {top3}
  {content}
</body>
</html>
"""
        message = f'<p class="error">{html.escape(error)}</p>' if error else ""
        content = (
                f'<img src="/image?name={quote(current)}" alt="验证码"><p class="meta">{escaped_current}</p>'
                f'<form method="post" action="/review">'
                f'<input type="hidden" name="filename" value="{escaped_current}">'
                '<label>识别结果：<input name="label" autofocus pattern="[A-Za-z0-9]{1,32}" required></label>'
                '<p><button type="submit">提交并下一张</button>'
                '<button type="submit" name="action" value="skip">跳过</button></p></form>'
                if current
                else "<p>没有待复核图片。</p>"
            )
        body = body.replace("{count}", str(len(files)))
        body = body.replace("{message}", message)
        body = body.replace("{top3}", top3_html)
        body = body.replace("{content}", content)
        self.send_html(body)

    def serve_image(self, filename: str) -> None:
        try:
            safe_filename(filename)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid image filename")
            return
        path = self.server.review_dir / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], root: Path, model_path: Path):
        self.root = root
        self.review_dir = root / "review"
        self.reviewed_dir = root / "reviewed"
        self.ocr = make_ocr(model_path)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.reviewed_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(address, ReviewHandler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/collected"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=Path("data/common_old.onnx"))
    args = parser.parse_args()
    server = ReviewServer(
        (args.host, args.port), args.output_dir.resolve(), args.model.resolve()
    )
    print(f"review server: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
