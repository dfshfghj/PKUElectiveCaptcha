from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGIN_URL = "https://iaaa.pku.edu.cn/iaaa/oauthlogin.do"
SSO_URL = "https://elective.pku.edu.cn/elective2008/ssoLogin.do"
COURSE_HOME_URL = (
    "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/help/"
    "HelpController.jpf"
)
CAPTCHA_URL = "https://elective.pku.edu.cn/elective2008/DrawServlet"
CAPTCHA_VERIFY_URL = (
    "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/"
    "supplement/validate.do"
)

HELP_TITLE = "<title>帮助-总体流程</title>"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/98 Safari/537.36"
)

LABEL_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")

# CAPTCHA type classifier thresholds.  The V<60 dark-pixel ratio separates
# the three non-blue samples in data/collected/classify:
# class_2 <= 0.18%, class_3 >= 3.42%, and class_1 >= 14.57%.
DARK_VALUE_THRESHOLD = 60
CLASS_1_DARK_RATIO_THRESHOLD = 0.10
CLASS_3_DARK_RATIO_THRESHOLD = 0.015
BLUE_PIXEL_RATIO_THRESHOLD = 0.05
SESSION_BATCH_SIZE = 15
SESSION_RELOGIN_DELAY = 5.0

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

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


@dataclass
class Credentials:
    username: str
    password: Optional[str] = None
    channel: Optional[str] = None


class AuthError(RuntimeError):
    pass


OCR_CHARSETS = {
    21: "2345678abcdefgmnpwxy",
    37: "0123456789abcdefghijklmnopqrstuvwxyz",
    63: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
}


class CaptchaOCR:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"OCR model not found: {model_path}")
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        output_shape = self.session.get_outputs()[0].shape
        class_count = output_shape[-1]
        try:
            self.charset = OCR_CHARSETS[class_count]
        except KeyError as exc:
            raise ValueError(
                f"unsupported OCR output class count: {class_count}; "
                "expected 21, 37, or 63"
            ) from exc
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def prepare_blue_filter_inpaint(image_bytes: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("captcha bytes cannot be decoded as an image")
        image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        line_mask = (hsv[:, :, 2] < 80).astype(np.uint8) * 255
        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        line_mask = cv2.dilate(
            line_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)),
            iterations=1,
        )
        blue_mask = cv2.inRange(
            hsv,
            np.array([100, 50, 50]),
            np.array([130, 255, 255]),
        )
        filtered = cv2.bitwise_and(image, image, mask=blue_mask)
        filtered[blue_mask == 0] = [255, 255, 255]
        return cv2.inpaint(filtered, line_mask, 3, cv2.INPAINT_TELEA)

    @classmethod
    def _prepare(cls, image_bytes: bytes, blue_filter: bool = False) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("captcha bytes cannot be decoded as an image")
        if blue_filter:
            image = cls.prepare_blue_filter_inpaint(image_bytes)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (130, 52), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image[None, None, :, :]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=-1, keepdims=True)
        exponent = np.exp(logits)
        return exponent / exponent.sum(axis=-1, keepdims=True)

    def classification(self, image_bytes: bytes, blue_filter: bool = False) -> str:
        logits = self.session.run(
            None, {self.input_name: self._prepare(image_bytes, blue_filter)}
        )[0]
        if logits.ndim != 3 or logits.shape[1] != 1:
            raise ValueError(f"unsupported OCR output shape: {logits.shape}")
        probabilities = self._softmax(logits[:, 0, :])
        beam: dict[tuple[str, ...], tuple[float, float]] = {(): (1.0, 0.0)}
        for distribution in probabilities:
            top_indices = np.argsort(distribution)[-min(20, len(distribution)) :][::-1]
            next_beam: dict[tuple[str, ...], list[float]] = {}
            for prefix, (blank_score, text_score) in beam.items():
                for index in top_indices:
                    probability = float(distribution[index])
                    if index == 0:
                        scores = next_beam.setdefault(prefix, [0.0, 0.0])
                        scores[0] += (blank_score + text_score) * probability
                        continue
                    character = self.charset[index - 1]
                    if prefix and prefix[-1] == character:
                        same = next_beam.setdefault(prefix, [0.0, 0.0])
                        same[1] += text_score * probability
                        extended = next_beam.setdefault(prefix + (character,), [0.0, 0.0])
                        extended[1] += blank_score * probability
                    else:
                        extended = next_beam.setdefault(prefix + (character,), [0.0, 0.0])
                        extended[1] += (blank_score + text_score) * probability
            beam = {
                prefix: (scores[0], scores[1])
                for prefix, scores in sorted(
                    next_beam.items(), key=lambda item: sum(item[1]), reverse=True
                )[:5]
            }
        if not beam:
            raise ValueError("OCR beam search produced no candidate")
        prefix, _ = max(beam.items(), key=lambda item: sum(item[1]))
        return "".join(prefix)


def mount_get_retries(session: requests.Session) -> None:
    # SSO occasionally closes the TLS connection during a GET.  Retry only
    # idempotent GET requests so the login POST is never replayed.
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        redirect=0,
        status=0,
        backoff_factor=0.5,
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

def build_session() -> requests.Session:
    session = requests.Session()
    session.verify = False
    mount_get_retries(session)
    session.headers.update(
        {"Referer": COURSE_HOME_URL, "Cache-Control": "max-age=0", "User-Agent": USER_AGENT}
    )
    return session


def authenticate(session: requests.Session, credentials: Credentials) -> None:
    response = session.post(
        LOGIN_URL,
        data={
            "appid": "syllabus", "userName": credentials.username,
            "password": credentials.password, "randCode": "", "smsCode": "",
            "otpCode": "", "redirUrl": "http://elective.pku.edu.cn:80/elective2008/agent4Iaaa.jsp/../ssoLogin.do",
        },
        cookies={"userName": credentials.username}, timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise AuthError(f"login failed: {payload}")
    token = payload.get("token")
    if not token:
        raise AuthError("login succeeded but token is missing")

    response = session.get(SSO_URL, params={"rand": "0.1", "token": token}, timeout=20)
    response.raise_for_status()
    body = response.text
    if HELP_TITLE in body:
        return
    if "/scnStAthVef.jsp/" in body:
        if not credentials.channel:
            raise AuthError("identity selection required; pass --channel bzx or --channel bfx")
        sida = body.partition("/ssoLogin.do?sida=")[2].partition("&")[0]
        if not sida.isalnum():
            raise AuthError("unable to extract sida from SSO response")
        response = session.get(
            SSO_URL, params={"sida": sida, "sttp": credentials.channel}, timeout=20
        )
        response.raise_for_status()
        if HELP_TITLE in response.text:
            return
    raise AuthError("after login check did not reach elective home")


def verify_alive(session: requests.Session) -> bool:
    response = session.get(COURSE_HOME_URL, timeout=20)
    response.raise_for_status()
    return HELP_TITLE in response.text


def captcha_class_features(image_bytes: bytes) -> tuple[float, float]:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("captcha bytes cannot be decoded as an image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark_ratio = float((hsv[:, :, 2] < DARK_VALUE_THRESHOLD).mean())
    blue_ratio = float(
        (
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 135)
            & (hsv[:, :, 1] >= 120)
            & (hsv[:, :, 2] >= DARK_VALUE_THRESHOLD)
        ).mean()
    )
    return dark_ratio, blue_ratio


def classify_captcha(image_bytes: bytes) -> str:
    dark_ratio, blue_ratio = captcha_class_features(image_bytes)
    if blue_ratio >= BLUE_PIXEL_RATIO_THRESHOLD:
        return "class_5" if dark_ratio >= CLASS_3_DARK_RATIO_THRESHOLD else "class_4"
    if dark_ratio >= CLASS_1_DARK_RATIO_THRESHOLD:
        return "class_1"
    if dark_ratio >= CLASS_3_DARK_RATIO_THRESHOLD:
        return "class_3"
    return "class_2"


def fetch_captcha_bytes(session: requests.Session) -> bytes:
    response = session.get(CAPTCHA_URL, params={"Rand": "0.1"}, timeout=20)
    response.raise_for_status()
    image_bytes = response.content
    if response.headers.get("Content-Type", "").lower().split(";", 1)[0] == "text/html":
        raise AuthError("captcha response is HTML; login session has expired")
    if cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) is None:
        raise AuthError("captcha response is not a valid image; login session has expired")
    return image_bytes


def verify_captcha_label(session: requests.Session, username: str, label: str) -> tuple[bool, Any]:
    response = session.post(
        CAPTCHA_VERIFY_URL, data={"validCode": label, "xh": username}, timeout=20
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("valid") == "2", payload


def load_cookie_session(cookie_path: Path) -> Optional[requests.Session]:
    if not cookie_path.exists():
        return None
    session = build_session()
    session.cookies.update(json.loads(cookie_path.read_text(encoding="utf-8")).get("cookies", {}))
    return session


def save_cookie_session(session: requests.Session, cookie_path: Path) -> None:
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    cookie_path.write_text(
        json.dumps({"cookies": requests.utils.dict_from_cookiejar(session.cookies)}, indent=2),
        encoding="utf-8",
    )
    cookie_path.chmod(0o600)


def resolve_credentials(args: argparse.Namespace) -> Credentials:
    username = args.username or os.environ.get("HEED_USERNAME", "").strip()
    if not username:
        raise AuthError("missing username; pass --username or set HEED_USERNAME")
    channel = (args.channel or os.environ.get("HEED_CHANNEL") or "").strip().lower() or None
    if channel not in (None, "bzx", "bfx"):
        raise AuthError("channel must be bzx or bfx")
    return Credentials(username=username, channel=channel)


def ensure_password(credentials: Credentials, args: argparse.Namespace) -> None:
    credentials.password = args.password or os.environ.get("HEED_PASSWORD")
    if not credentials.password:
        credentials.password = getpass.getpass("password: ")
    if not credentials.password:
        raise AuthError("missing password")


def safe_label(label: str) -> str:
    normalized = label.strip().lower()
    if not LABEL_RE.fullmatch(normalized):
        raise ValueError(f"trained model returned unsupported label: {label!r}")
    return normalized


def make_sample_name(index: int, image_bytes: bytes) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(image_bytes).hexdigest()[:12]
    return f"{stamp}_{index:05d}_{digest}.png"


def append_manifest(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prepare_dirs(root: Path) -> tuple[Path, Path, Path]:
    raw, auto_labeled, review = root / "raw", root / "auto_labeled", root / "review"
    for directory in (raw, auto_labeled, review):
        directory.mkdir(parents=True, exist_ok=True)
    return raw, auto_labeled, review


def make_ocr(model_path: Path) -> CaptchaOCR:
    return CaptchaOCR(model_path)


def collect(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).resolve()
    raw_dir, auto_dir, review_dir = prepare_dirs(root)
    manifest = root / "manifest.csv"
    ocr = make_ocr(Path(args.model).resolve())
    credentials = resolve_credentials(args)
    cookie_path = Path(args.cookies_file).resolve()
    session = load_cookie_session(cookie_path) if args.reuse_cookies else None
    if session is None:
        ensure_password(credentials, args)
        session = build_session()
        authenticate(session, credentials)
        save_cookie_session(session, cookie_path)
    else:
        try:
            if not verify_alive(session):
                raise AuthError("persisted cookie session expired")
        except Exception:
            ensure_password(credentials, args)
            session = build_session()
            authenticate(session, credentials)
            save_cookie_session(session, cookie_path)

    for index in range(1, args.count + 1):
        image_bytes = fetch_captcha_bytes(session)
        digest = hashlib.sha256(image_bytes).hexdigest()
        sample_name = make_sample_name(index, image_bytes)
        raw_path = raw_dir / sample_name
        raw_path.write_bytes(image_bytes)
        status, prediction, verified = "needs_manual_label", "", False
        response_payload: Any = None
        final_path = review_dir / sample_name
        try:
            prediction = safe_label(ocr.classification(image_bytes))
            verified, response_payload = verify_captcha_label(session, credentials.username, prediction)
            if verified:
                status = "auto_verified"
                label_dir = auto_dir / prediction
                label_dir.mkdir(parents=True, exist_ok=True)
                final_path = label_dir / sample_name
        except Exception as exc:
            response_payload = {"error": f"{type(exc).__name__}: {exc}"}
        shutil.move(str(raw_path), final_path)
        append_manifest(manifest, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": str(final_path.relative_to(root)).replace("\\", "/"),
            "status": status, "predicted_label": prediction,
            "verified": "true" if verified else "false", "sha256": digest,
            "bytes": str(len(image_bytes)),
            "verification_response": json.dumps(response_payload, ensure_ascii=False, separators=(",", ":")),
        })
        print(f"[{index}/{args.count}] {status}: {final_path.relative_to(root)} prediction={prediction or '-'}")
        if index < args.count and args.delay > 0:
            time.sleep(args.delay)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("collect", help="fetch, OCR, server-verify, and archive samples")
    command.add_argument("--username")
    command.add_argument("--password")
    command.add_argument("--channel", choices=["bzx", "bfx"])
    command.add_argument("--count", type=int, default=20)
    command.add_argument("--delay", type=float, default=0.8)
    command.add_argument("--output-dir", default="data/collected")
    command.add_argument("--cookies-file", default=".session_cookies.json")
    command.add_argument("--model", default="data/captcha_mobilenet_ctc.onnx")
    command.add_argument("--reuse-cookies", action="store_true")
    command.set_defaults(handler=collect)
    return parser

def build_session():
    session = requests.Session()
    session.verify = False
    mount_get_retries(session)

    session.headers.update(
        {
            "Referer": COURSE_HOME_URL,
            "Cache-Control": "max-age=0",
            "User-Agent": USER_AGENT,
        }
    )

    return session



def fetch_captcha_bytes(session):
    r = session.get(
        CAPTCHA_URL,
        params={"Rand": "0.1"},
        timeout=20,
    )

    r.raise_for_status()

    image_bytes = r.content
    content_type = r.headers.get("Content-Type", "").lower().split(";", 1)[0]
    if content_type == "text/html":
        raise AuthError("captcha response is HTML; login session has expired")
    if cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) is None:
        raise AuthError("captcha response is not a valid image; login session has expired")

    return image_bytes



def safe_label(label):
    label = label.strip().lower()

    if not LABEL_RE.fullmatch(label):
        raise ValueError(
            f"invalid OCR label {label}"
        )

    return label

def verify_captcha_label(session, username, label):
    response = session.post(
        CAPTCHA_VERIFY_URL,
        data={
            "validCode": label,
            "xh": username,
        },
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    return payload.get("valid") == "2", payload



def load_cookie_session(cookie_path):
    if not cookie_path.exists():
        return None

    session = build_session()

    session.cookies.update(
        json.loads(
            cookie_path.read_text(
                encoding="utf-8"
            )
        ).get("cookies", {})
    )

    return session



def save_cookie_session(session, cookie_path):
    cookie_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cookie_path.write_text(
        json.dumps(
            {
                "cookies":
                    requests.utils.dict_from_cookiejar(
                        session.cookies
                    )
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cookie_path.chmod(0o600)



def make_sample_name(index, image_bytes):
    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    digest = hashlib.sha256(
        image_bytes
    ).hexdigest()[:12]

    return (
        f"{stamp}_{index:05d}_{digest}.png"
    )


def make_labeled_sample_name(sample_name, label):
    path = Path(sample_name)
    return f"{path.stem}_{label}{path.suffix}"



def append_manifest(path, row):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = not path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=MANIFEST_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)



def prepare_dirs(root):

    raw = root / "raw"
    processed = root / "processed"
    auto_labeled = root / "auto_labeled"
    review = root / "review"

    for d in (
        raw,
        processed,
        auto_labeled,
        review,
    ):
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    return (
        raw,
        processed,
        auto_labeled,
        review,
    )



def make_ocr(model_path):
    return CaptchaOCR(model_path)

def collect(args):

    root = Path(
        args.output_dir
    ).resolve()

    (
        raw_dir,
        processed_dir,
        auto_dir,
        review_dir,
    ) = prepare_dirs(root)


    manifest = root / "manifest.csv"


    ocr = None if args.images_only else make_ocr(Path(args.model).resolve())


    credentials = resolve_credentials(args)


    cookie_path = Path(
        args.cookies_file
    ).resolve()


    session = (
        load_cookie_session(cookie_path)
        if args.reuse_cookies
        else None
    )


    if session is None:

        ensure_password(
            credentials,
            args
        )

        session = build_session()

        authenticate(
            session,
            credentials
        )

        save_cookie_session(
            session,
            cookie_path
        )


    verified_count = 0
    for index in range(
        1,
        args.count + 1
    ):

        image_bytes = fetch_captcha_bytes(
            session
        )

        captcha_class = classify_captcha(image_bytes)

        digest = hashlib.sha256(
            image_bytes
        ).hexdigest()


        sample_name = make_sample_name(
            index,
            image_bytes,
        )


        raw_path = raw_dir / sample_name
        raw_path.write_bytes(
            image_bytes
        )

        if args.images_only:
            print(
                f"[{index}/{args.count}] collected: "
                f"{raw_path.relative_to(root)} class={captcha_class}"
            )
            if index < args.count:
                if args.delay > 0:
                    time.sleep(args.delay)
            continue

        status = "needs_manual_label"
        prediction = ""
        ocr_source = "trained_model"
        verified = False
        response_payload = None
        try:
            prediction = safe_label(ocr.classification(image_bytes))
            verified, response_payload = verify_captcha_label(
                session,
                credentials.username,
                prediction,
            )
        except Exception as exc:
            response_payload = {"error": f"{type(exc).__name__}: {exc}"}

        if verified:
            verified_count += 1

        if verified:

            status = "auto_verified"

            final_path = (
                auto_dir / make_labeled_sample_name(sample_name, prediction)
            )

        else:

            final_path = (
                review_dir / sample_name
            )


        shutil.move(
            str(raw_path),
            final_path,
        )


        append_manifest(
            manifest,
            {
                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "filename":
                    str(
                        final_path.relative_to(root)
                    ).replace("\\", "/"),

                "status":
                    status,

                "predicted_label":
                    prediction,

                "ocr_source":
                    ocr_source,

                "verified":
                    str(verified).lower(),

                "sha256":
                    digest,

                "bytes":
                    str(len(image_bytes)),

                "verification_response":
                    json.dumps(
                        response_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
            },
        )


        print(
            f"[{index}/{args.count}] "
            f"{status} "
            f"class={captcha_class} "
            f"label={prediction or '-'} "
            f"source={ocr_source or '-'} "
            f"correct={verified_count}/{index} ({verified_count / index:.2%})"
        )


        if index < args.count:
            if index % SESSION_BATCH_SIZE == 0:
                time.sleep(SESSION_RELOGIN_DELAY)
                session = build_session()
                authenticate(session, credentials)
                save_cookie_session(session, cookie_path)
            elif args.delay > 0:
                time.sleep(args.delay)


    print(
        f"summary: correct={verified_count}/{args.count} "
        f"({verified_count / args.count:.2%})"
    )
    return 0

def build_parser():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    cmd = sub.add_parser(
        "collect"
    )

    cmd.add_argument(
        "--username"
    )

    cmd.add_argument(
        "--password"
    )

    cmd.add_argument(
        "--channel",
        choices=[
            "bzx",
            "bfx",
        ],
    )

    cmd.add_argument(
        "--count",
        type=int,
        default=20,
    )

    cmd.add_argument(
        "--delay",
        type=float,
        default=0.8,
    )

    cmd.add_argument(
        "--output-dir",
        default="data/collected",
    )

    cmd.add_argument(
        "--cookies-file",
        default=".session_cookies.json",
    )

    cmd.add_argument(
        "--model",
        default="data/captcha_mobilenet_ctc.onnx",
    )

    cmd.add_argument(
        "--reuse-cookies",
        action="store_true",
    )

    cmd.add_argument(
        "--images-only",
        action="store_true",
        help="only download and save captcha images; skip preprocessing, OCR, and verification",
    )

    cmd.set_defaults(
        handler=collect
    )

    return parser



def main(argv):

    parser = build_parser()

    args = parser.parse_args(argv)

    try:
        return args.handler(args)

    except KeyboardInterrupt:
        return 130

    except Exception as exc:

        print(
            f"error: {exc}",
            file=sys.stderr,
        )

        return 1



if __name__ == "__main__":
    raise SystemExit(
        main(sys.argv[1:])
    )
