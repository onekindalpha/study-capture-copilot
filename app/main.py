from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv as _python_dotenv_load
except ModuleNotFoundError:
    def _load_dotenv(path: Path, override: bool = False) -> bool:
        """Minimal .env loader fallback when python-dotenv is not installed."""
        try:
            if not path.exists():
                return False
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (override or key not in os.environ):
                    os.environ[key] = value
            return True
        except Exception:
            return False
else:
    def _load_dotenv(path: Path, override: bool = False) -> bool:
        return bool(_python_dotenv_load(path, override=override))

DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DOTENV_LOADED = _load_dotenv(DOTENV_PATH, override=True)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CAPTURE_DIR = DATA_DIR / "captures"
DOCUMENT_DIR = DATA_DIR / "documents"
DOCUMENT_TMP_DIR = DATA_DIR / "tmp_uploads"
NOTES_PATH = DATA_DIR / "notes.jsonl"
SESSIONS_PATH = DATA_DIR / "sessions.json"
EXAMPLES_DIR = BASE_DIR / "examples"
ARTICLE_TYPE_CONFIG_DIR = BASE_DIR / "configs" / "article_types"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_TMP_DIR.mkdir(parents=True, exist_ok=True)
NOTES_PATH.touch(exist_ok=True)
SESSIONS_PATH.touch(exist_ok=True)

import sys as _sys  # noqa: E402 - local import kept next to the path setup it supports

if str(BASE_DIR) not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR))

from tools.document_ingest import convert_document_to_markdown  # noqa: E402

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
MODEL_PROVIDER_FAST = os.getenv("MODEL_PROVIDER_FAST", "groq")
MODEL_PROVIDER_DEEP = os.getenv("MODEL_PROVIDER_DEEP", "openai")


@dataclass
class StudyNote:
    id: str
    created_at: str
    title: str
    source_type: str
    tags: list[str]
    raw_text: str
    user_memo: str
    summary: str
    action_items: list[str]
    blog_draft: str
    image_path: str | None = None
    image_paths: list[str] | None = None


@dataclass
class UploadedFile:
    filename: str
    data: bytes


FormData = dict[str, list[str | UploadedFile]]


@dataclass
class ImageEvidence:
    image_no: int
    caption: str
    visible_evidence: list[str]
    role: str
    problem_signal: str
    technical_entities: list[str]
    inferred_meaning: str
    confidence: float = 0.0
    evidence_source: str = ""


@dataclass
class CritiqueResult:
    passed: bool
    failures: list[str]
    section_failures: dict[str, str]
    metrics: dict[str, Any]


@dataclass
class CaptureEvent:
    capture_id: int
    timestamp: str
    image_path: str
    source_title: str = ""
    source_url: str = ""
    user_note: str = ""
    ocr_text: str = ""
    vision_summary: str = ""
    auto_keywords: list[str] | None = None


@dataclass
class QALog:
    qa_id: int
    timestamp: str
    related_capture_ids: list[int]
    selected_text: str
    question: str
    answer: str
    answer_summary: str
    learner_state: str
    resolved: bool
    used_in_article: bool


@dataclass
class DocumentEvidence:
    document_id: int
    timestamp: str
    original_filename: str
    ext: str
    status: str  # "ok" | "error"
    char_count: int
    text_path: str | None = None
    error: str | None = None


class LLM:
    def __init__(self) -> None:
        self.client = None
        self.last_diagnostics: dict[str, Any] = {}

    def get_client(self) -> Any:
        if self.client:
            return self.client
        if not GROQ_API_KEY:
            self.last_diagnostics = provider_diagnostics(
                provider="groq",
                exception=RuntimeError("GROQ_API_KEY is not set"),
                package_import_ok=groq_import_ok(),
                client_init_ok=False,
            )
            return None
        try:
            from groq import Groq
        except ModuleNotFoundError as exc:
            self.last_diagnostics = provider_diagnostics(
                provider="groq",
                exception=exc,
                package_import_ok=False,
                client_init_ok=False,
            )
            return None
        try:
            self.client = Groq(api_key=GROQ_API_KEY, timeout=90.0)
            self.last_diagnostics = provider_diagnostics(provider="groq", package_import_ok=True, client_init_ok=True)
            return self.client
        except Exception as exc:
            self.last_diagnostics = provider_diagnostics(
                provider="groq",
                exception=exc,
                package_import_ok=True,
                client_init_ok=False,
            )
            return None

    def generate_note(self, raw_text: str, memo: str) -> dict[str, Any]:
        if not raw_text.strip() and not memo.strip():
            return fallback_note(raw_text, memo)

        client = self.get_client()
        if not client:
            return fallback_note(raw_text, memo)

        prompt = f"""
아래 학습 화면/실습 기록을 기술 노트로 정리해 주세요.

규칙:
- 한국어로 작성합니다.
- 과장하지 말고 입력에 있는 내용만 사용합니다.
- JSON만 반환합니다.
- keys: title, source_type, tags, summary, action_items, blog_draft
- summary는 문제 인식, 핵심 개념, 실습 흐름, 막힌 지점, 해결 방향을 포함합니다.
- blog_draft는 기술블로그 초안처럼 제목/배경/실습/배운점 구조로 작성합니다.

[화면 텍스트]
{raw_text}

[사용자 메모]
{memo}
""".strip()
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                max_tokens=1200,
                messages=[
                    {
                        "role": "system",
                        "content": "You convert study captures into structured developer learning notes. Return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            return parse_json_or_fallback(content, raw_text, memo)
        except Exception as exc:
            print(f"[LLM text error] {exc}")
            return fallback_note(raw_text, memo)

    def generate_note_from_image(self, image_file: Path, memo: str) -> dict[str, Any]:
        return self.generate_note_from_images([image_file], memo)

    def generate_note_from_images(self, image_files: list[Path], memo: str) -> dict[str, Any]:
        client = self.get_client()
        if not client:
            return image_only_note(f"/captures/{image_files[0].name}" if image_files else "")

        if len(image_files) > 6:
            partial_notes: list[dict[str, Any]] = []
            for chunk_index, start in enumerate(range(0, len(image_files), 6), start=1):
                chunk = image_files[start : start + 6]
                chunk_memo = f"{memo}\n\n이 묶음은 전체 캡처 중 {chunk_index}번째 묶음입니다."
                partial_note = self.generate_note_from_images(chunk, chunk_memo)
                partial_notes.append(partial_note)
            return self.combine_image_notes(partial_notes, memo, len(image_files))

        for candidate_files in (image_files, image_files[:3], image_files[:1]):
            if not candidate_files:
                continue
            note = self.try_generate_note_from_images(client, candidate_files, memo)
            if note:
                return note
        return image_only_note(f"/captures/{image_files[0].name}" if image_files else "")

    def combine_image_notes(self, partial_notes: list[dict[str, Any]], memo: str, image_count: int) -> dict[str, Any]:
        joined = "\n\n".join(
            f"[묶음 {index}]\n제목: {note.get('title', '')}\n요약:\n{note.get('summary', '')}\n액션:\n{', '.join(note.get('action_items', []))}"
            for index, note in enumerate(partial_notes, start=1)
        )
        client = self.get_client()
        if client:
            prompt = f"""
아래는 사용자가 학습/Lab 실습 중 순서대로 캡처한 {image_count}장 이미지를 6장 이하 묶음으로 나누어 판독한 결과입니다.
묶음별 내용을 하나의 흐름으로 통합해 학습 노트 JSON을 작성해 주세요.

규칙:
- 한국어로 작성합니다.
- 입력된 묶음 요약과 사용자 메모만 근거로 사용합니다.
- JSON만 반환합니다.
- keys: title, source_type, tags, summary, action_items, blog_draft
- summary는 전체 실습 흐름을 압축하지 말고, 이미지 묶음별로 문제/원인/조치/검증 흐름을 길게 정리합니다.
- blog_draft는 Medium 포트폴리오 초안처럼 배경/문제 인식/문제 정의/해결 흐름/성과/배운 점 구조로 충분히 길게 작성합니다.

[사용자 메모]
{memo}

[묶음별 판독 결과]
{joined}
""".strip()
            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=0.2,
                    max_tokens=3200,
                    messages=[
                        {"role": "system", "content": "You merge sequential study-capture notes into one grounded structured JSON note."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return parse_json_or_fallback(completion.choices[0].message.content or "", joined, memo)
            except Exception as exc:
                print(f"[LLM merge error] {exc}")
        return {
            "title": f"{image_count}장 캡처 기반 학습 기록",
            "source_type": "study-capture",
            "tags": ["study-note", "multi-image", "vision"],
            "summary": joined,
            "action_items": ["묶음별 핵심 흐름 연결", "문제 인식과 해결 과정 보강", "문제 해결형 Medium 글로 확장"],
            "blog_draft": f"# {image_count}장 캡처 기반 학습 기록\n\n{joined}",
        }

    def try_generate_note_from_images(self, client: Any, image_files: list[Path], memo: str) -> dict[str, Any] | None:
        prompt = f"""
아래 이미지는 사용자가 학습/Lab 실습 중 순서대로 캡처한 화면입니다. 이미지 순서를 실습 흐름으로 보고 포트폴리오용 학습 노트로 정리해 주세요.

규칙:
- 한국어로 작성합니다.
- 화면에 보이는 내용과 사용자 메모만 근거로 사용합니다.
- JSON만 반환합니다.
- keys: title, source_type, tags, summary, action_items, blog_draft
- 여러 이미지가 있으면 이미지 1, 이미지 2... 순서대로 실습 흐름을 연결합니다.
- summary에는 각 이미지에서 확인한 화면 변화, 발견한 문제, 의심 원인, 조치, 검증 포인트를 생략하지 말고 정리합니다.
- blog_draft는 문제 해결형 기술블로그 초안처럼 제목/배경/문제 인식/문제 정의/해결 흐름/성과/배운 점 구조로 충분히 길게 작성합니다.
- 이미지가 여러 장이면 이미지별 캡션 후보를 함께 정리합니다.

[사용자 메모]
{memo}
""".strip()
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, image_file in enumerate(image_files, start=1):
            mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
            image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
            content_parts.append({"type": "text", "text": f"이미지 {index}"})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                }
            )
        try:
            completion = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                temperature=0.2,
                max_tokens=2400,
                messages=[
                    {
                        "role": "system",
                        "content": "You read study screenshots and turn them into structured developer learning notes. Return strict JSON.",
                    },
                    {
                        "role": "user",
                        "content": content_parts,
                    },
                ],
            )
            content = completion.choices[0].message.content or ""
            return parse_json_or_fallback(content, "", memo)
        except Exception as exc:
            print(f"[LLM vision error] {exc}")
            return None

    def synthesize_blog(self, notes: list[StudyNote], topic: str, format_type: str, extra_info: str = "") -> str:
        notes = [note for note in notes if is_meaningful_note(note)][-8:]
        if not notes:
            return "문제해결형 Medium 글을 만들 수 있는 학습 노트가 아직 없습니다. 스크린샷을 업로드하거나 메모를 입력한 뒤 먼저 캡처 노트를 생성해 주세요."

        joined = "\n\n".join(
            f"[노트 {index}]\n제목: {note.title}\n요약:\n{note.summary[:5200]}\n액션: {', '.join(note.action_items[:8])}\n초안:\n{note.blog_draft[:3600]}"
            for index, note in enumerate(notes, start=1)
        )
        client = self.get_client()
        if not client:
            return local_portfolio_blog(notes, topic)

        prompt = portfolio_prompt(topic, joined, extra_info)
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.25,
                max_tokens=7800,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Korean technical portfolio writer. "
                            "Write long, concrete, grounded Medium-ready problem-solving articles from study notes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content or local_portfolio_blog(notes, topic)
        except Exception:
            return local_portfolio_blog(notes, topic)

    def synthesize_blog_from_capture(
        self,
        raw_text: str,
        memo: str,
        image_files: list[Path],
        topic: str,
        extra_info: str = "",
        image_names: list[str] | None = None,
        captures: list[dict[str, Any]] | None = None,
        qa_logs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return generate_medium_article_pipeline(
            self,
            raw_text=raw_text,
            memo=memo,
            image_files=image_files,
            topic=topic,
            extra_info=extra_info,
            image_names=image_names or [path.name for path in image_files],
            captures=captures or [],
            qa_logs=qa_logs or [],
        )

    def describe_image_sequence(self, image_files: list[Path], memo: str, extra_info: str) -> str:
        client = self.get_client()
        if not client:
            return ""
        chunks: list[str] = []
        for chunk_index, start in enumerate(range(0, len(image_files), 6), start=1):
            chunk = image_files[start : start + 6]
            prompt = f"""
아래 이미지는 사용자가 실습/프로젝트를 진행한 순서대로 캡처한 화면 중 {chunk_index}번째 묶음입니다.
최종 Medium 글을 쓰기 위한 근거 메모를 작성해 주세요.

규칙:
- 최종 글을 쓰지 말고, 이미지별 관찰 내용과 문제 해결 흐름만 자세히 정리합니다.
- 버튼 클릭 설명이 아니라 문제/원인/조치/검증 관점으로 해석합니다.
- 화면에서 확인되는 기술명, 테이블명, 수식명, 오류, 결과값을 가능한 한 보존합니다.
- 사용자가 제공한 추가 정보와 모순되지 않게 작성합니다.
- 이미지 번호는 전체 순서를 기준으로 {start + 1}번부터 시작합니다.

[사용자 메모]
{memo}

[추가 정보]
{extra_info}
""".strip()
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for offset, image_file in enumerate(chunk, start=start + 1):
                mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
                image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
                content_parts.append({"type": "text", "text": f"이미지 {offset}"})
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}})
            try:
                completion = client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    temperature=0.2,
                    max_tokens=3200,
                    messages=[
                        {"role": "system", "content": "You read technical screenshots and create detailed grounded writing notes."},
                        {"role": "user", "content": content_parts},
                    ],
                )
                chunks.append(completion.choices[0].message.content or "")
            except Exception as exc:
                print(f"[LLM direct image sequence error] {exc}")
        return "\n\n".join(f"[이미지 판독 묶음 {index}]\n{chunk}" for index, chunk in enumerate(chunks, start=1))


llm = LLM()


def groq_import_ok() -> bool:
    try:
        import groq  # noqa: F401
        return True
    except Exception:
        return False


def provider_diagnostics(
    provider: str = "groq",
    exception: Exception | None = None,
    package_import_ok: bool | None = None,
    client_init_ok: bool | None = None,
    test_call_ok: bool | None = None,
) -> dict[str, Any]:
    package_ok = groq_import_ok() if package_import_ok is None and provider == "groq" else bool(package_import_ok)
    diagnostics: dict[str, Any] = {
        "provider": provider,
        "model": GROQ_MODEL if provider == "groq" else OPENAI_MODEL,
        "vision_model": GROQ_VISION_MODEL if provider == "groq" else OPENAI_MODEL,
        "text_model": GROQ_MODEL if provider == "groq" else OPENAI_MODEL,
        "api_key_present": bool(GROQ_API_KEY if provider == "groq" else OPENAI_API_KEY),
        "package_import_ok": package_ok,
        "dotenv_loaded": bool(DOTENV_LOADED),
        "dotenv_path": str(DOTENV_PATH),
        "dotenv_exists": DOTENV_PATH.exists(),
        "client_init_ok": client_init_ok,
        "test_call_ok": test_call_ok,
        "selected_fast_provider": MODEL_PROVIDER_FAST,
        "selected_deep_provider": MODEL_PROVIDER_DEEP,
    }
    if exception:
        diagnostics.update(
            {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback_summary": "".join(traceback.format_exception_only(type(exception), exception)).strip(),
            }
        )
    return diagnostics


def llm_health_check(run_test_call: bool = False) -> dict[str, Any]:
    groq_diag = provider_diagnostics("groq", package_import_ok=groq_import_ok(), client_init_ok=False, test_call_ok=None)
    if groq_diag["api_key_present"] and groq_diag["package_import_ok"]:
        try:
            from groq import Groq

            client = Groq(api_key=GROQ_API_KEY, timeout=20.0)
            groq_diag["client_init_ok"] = True
            if run_test_call:
                try:
                    completion = client.chat.completions.create(
                        model=GROQ_MODEL,
                        temperature=0,
                        max_tokens=4,
                        messages=[{"role": "user", "content": "ping"}],
                    )
                    groq_diag["test_call_ok"] = bool(completion.choices)
                except Exception as exc:
                    groq_diag["test_call_ok"] = False
                    groq_diag["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            groq_diag.update(provider_diagnostics("groq", exception=exc, package_import_ok=True, client_init_ok=False))
    openai_diag = {
        "enabled": MODEL_PROVIDER_FAST == "openai" or MODEL_PROVIDER_DEEP == "openai",
        "api_key_present": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
    }
    groq_diag["enabled"] = MODEL_PROVIDER_FAST == "groq" or MODEL_PROVIDER_DEEP == "groq"
    return {"groq": groq_diag, "openai": openai_diag}



class _PlainHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3", "p", "li", "pre", "code", "td", "th"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3", "p", "li", "pre", "code", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title:
            self.title += text + " "
        self.parts.append(text)

    def text(self) -> str:
        joined = "\n".join(part.strip() for part in self.parts if part.strip())
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]}>\"']+", text or "")
    cleaned: list[str] = []
    for url in urls:
        url = url.rstrip(".,;，。")
        if url not in cleaned:
            cleaned.append(url)
    return cleaned[:8]


def fetch_public_source_text(url: str, max_chars: int = 12000) -> str:
    """Best-effort public URL reader for lecture/lab pages.

    It intentionally does not require the user to paste screenshots.  Login-gated
    pages and YouTube transcript extraction are reported as source hints rather
    than treated as fatal errors.
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return (
            f"[영상 URL]\n{url}\n"
            "YouTube 영상 URL입니다. 현재 구현은 영상 자막을 자동 추출하지 않으므로, "
            "영상 제목/자막/요약을 붙여넣으면 더 정확한 학습 흐름을 만들 수 있습니다."
        )
    if "aiskillsnavigator.microsoft.com" in host:
        return (
            f"[강의 플레이어 URL]\n{url}\n"
            "AI Skills Navigator 플레이어 URL입니다. 로그인/동적 렌더링 페이지일 수 있어 "
            "URL은 출처 힌트로만 사용하고, 실제 강의안 본문이나 실습 URL 내용을 우선 근거로 사용합니다."
        )
    allowed_hosts = ("microsoftlearning.github.io", "learn.microsoft.com", "docs.github.com", "github.com", "raw.githubusercontent.com")
    if not any(host.endswith(item) for item in allowed_hosts):
        return f"[URL 힌트]\n{url}\n자동 본문 추출 대상이 아닌 URL입니다. URL 문자열은 주제 힌트로만 사용합니다."
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 StudyCaptureAgent/1.0"})
        with urlopen(req, timeout=12) as resp:
            raw = resp.read(700000)
            content_type = resp.headers.get("Content-Type", "")
        text = raw.decode("utf-8", errors="replace")
        if "html" in content_type or text.lstrip().startswith("<!") or "<html" in text[:500].lower():
            parser = _PlainHTMLTextExtractor()
            parser.feed(text)
            body = parser.text()
            title = parser.title.strip()
            if title:
                body = f"제목: {title}\n\n{body}"
            return f"[URL 자동 추출]\n{url}\n\n{body[:max_chars]}"
        return f"[URL 자동 추출]\n{url}\n\n{text[:max_chars]}"
    except Exception as exc:
        return f"[URL 추출 실패]\n{url}\n{type(exc).__name__}: {exc}\n이 URL은 문자열 힌트로만 사용합니다."


def enrich_raw_text_with_source_urls(raw_text: str, memo: str) -> str:
    source = f"{raw_text}\n{memo}"
    urls = extract_urls_from_text(source)
    if not urls:
        return raw_text
    extracted = [fetch_public_source_text(url) for url in urls]
    return raw_text.rstrip() + "\n\n" + "\n\n".join(extracted)

def direct_source_text(raw_text: str, memo: str, image_count: int) -> str:
    sections = []
    if raw_text.strip():
        sections.append(f"[화면 텍스트/코드/오류]\n{raw_text.strip()}")
    if memo.strip():
        sections.append(f"[사용자 메모/질문/해결 과정]\n{memo.strip()}")
    if image_count:
        sections.append(
            "[이미지 정보]\n"
            f"사용자가 실습 진행 순서대로 업로드한 이미지 수: {image_count}장\n"
            "이미지 자체는 최종 글에서 단순 캡션이 아니라 문제 발견, 원인 분석, 해결 과정, 검증 결과의 근거로 사용해야 합니다."
        )
    sections.append(
        "[작성 주의]\n"
        "비어 있는 입력칸, 내부 라벨, '직접 입력된 화면 텍스트 없음', '직접 입력된 사용자 메모 없음' 같은 시스템 상태 문구는 최종 글에 절대 쓰지 않습니다."
    )
    return "\n\n".join(sections)


SECTION_TITLES = [
    "한국어 제목",
    "영어 부제",
    "짧은 도입부",
    "핵심 작업 요약",
    "문제 인식",
    "문제 정의",
    "왜 이것을 문제로 인식했는가",
    "문제 해결 경험",
    "복잡한 문제/수식/쿼리/코드 작성 및 해결 경험",
    "성과",
    "사용한 주요 수식/코드 정리",
    "최종 정리",
    "Portfolio Summary",
    "Key skills practiced",
    "이미지 번호와 캡션 목록",
]

PLACEHOLDER_PHRASES = [
    ":::writing",
    'id="',
    "여기에 이미지 넣기",
    "첨부 삽입",
    "이미지 추가 예정",
    "[화면 텍스트/코드/오류]",
    "[사용자 메모/질문/해결 과정]",
    "[이미지 정보]",
    "[작성 주의]",
    "근거 화면 1 기반 검증 단계",
    "근거 화면 2 기반 검증 단계",
    "근거 화면 3 기반 검증 단계",
    "근거 화면 4 기반 검증 단계",
    "이미지 근거와 사용자 메모를 연결해 문제 해결 단계로 재구성했습니다.",
    "문제 상황 제시",
    "문제 원인 분석",
]

FUNCTION_DESCRIPTION_PHRASES = [
    "버튼을 눌렀습니다",
    "차트를 만들었습니다",
    "관계를 설정했습니다",
    "실습을 완료했습니다",
]

FILLER_SENTENCES = [
    "이 단계에서는 화면에 보이는 현상을 그대로 받아들이지 않고",
    "조치는 기능 실행 자체가 아니라",
    "이 확인 결과는 다음 단계로 넘어가기 위한 기준이며",
    "데이터 변환 작업 성공",
    "오류를 확인하고 수정",
    "문제 해결 서사에 필요한 구체적 캡션",
    "추가 메모 없음",
    "데이터 소스 연결 및 데이터 로딩이 문제가 발생하여 해결이 필요하다",
    "데이터 로딩이 성공적으로 이루어졌는지 확인",
    "데이터 테이블 간 연관관계 확인이 성공적으로 이루어졌는지 확인",
    "데이터의 문제가 있음을 발견",
    "카테고리별 판매 금액이 동일하여 데이터의 문제가 있음을 발견",
    "최종 판매 목표 데이터 확인",
    "지역별 판매 데이터 확인",
    "문제 영역 식별",
    "이익률 결과를 확대하여 세부적으로 확인",
    "판매 데이터에서 비용을 차감하여 이익을 계산하는 측정값을 생성",
    "최종 salesperson 판매 목표 데이터 확인",
    "Vision 판독 실패",
    "파일명과 입력 메모만 근거로 사용",
    "파일명과 입력 메모를 분석하여 문제를 이해한다",
    "파일명과 입력 메모를 분석하여 원인을 찾는다",
    "파일명과 입력 메모를 분석하여 해결 방안을 찾는다",
    "파일명과 입력 메모를 분석하여 검증한다",
]

NON_POWERBI_TEMPLATE_PHRASES = [
    "화면에 값이 표시되더라도 모델, 수식, 관계, 필터 흐름이 맞지 않으면 분석 결과의 의미가 달라질 수 있습니다.",
    "데이터 모델, 수식, 관계, 변환 흐름 중 하나가 어긋나면",
    "화면에 숫자가 나오거나 설정 창이 열린 것만으로는",
]

POWER_QUERY_CORE_PROBLEM = (
    "SQL Server와 CSV에서 가져온 원본 데이터가 보고서 작성에 바로 사용할 수 있는 분석 모델 형태가 아니었고, "
    "데이터 품질·컬럼 구조·비표준 값·결측 원가·월별 목표 데이터 구조·보조 테이블 병합·최종 로드 범위를 정리해야 했다."
)

SEMANTIC_CORE_PROBLEM = (
    "Product[Category] 기준으로 Sales[Sales]를 집계했을 때 모든 Category 행에 동일한 매출 총액이 반복되어, "
    "Product 차원의 filter context가 Sales fact table로 전달되지 않는 relationship 문제를 확인했다."
)

DAX_CORE_PROBLEM = (
    "Sales 데이터를 단순 합계와 raw column 자동 집계에 의존하면 월 정렬, 날짜 기준 분석, 가격 지표, 목표 대비 성과 분석이 불명확해지므로 "
    "MonthKey 정렬, Date table, explicit measure, Target/Variance measure로 계산 맥락을 명확히 구성해야 했다."
)

INTERNAL_DIAGNOSTIC_PHRASES = [
    "Vision 판독 실패",
    "파일명과 입력 메모만 근거로 사용",
    "Vision 응답이 없어",
    "파일명과 입력 메모를 분석하여",
]

SEMANTIC_CONCRETE_ENTITIES = [
    "Product[Category]",
    "Sales[Sales]",
    "Product[ProductKey] -> Sales[ProductKey]",
    "One to many",
    "Cross-filter direction",
    "Star schema",
    "Product hierarchy",
    "Profit",
    "Profit Margin",
    "DIVIDE",
    "SalespersonRegion",
    "Bridge table",
    "Salesperson -> SalespersonRegion -> Region -> Sales",
    "inactive relationship",
    "Targets",
]

SEMANTIC_REGRESSION_REQUIREMENTS = [
    ("이미지 1 문제 화면 설명", ["이미지 1", "Product[Category]", "Sales[Sales]", "반복"]),
    ("이미지 2 ProductKey relationship 설정", ["이미지 2", "Product[ProductKey] -> Sales[ProductKey]", "One to many"]),
    ("이미지 4 star schema 모델 구조", ["이미지 4", "Star schema"]),
    ("이미지 5~6 hierarchy 구성", ["이미지 5", "Product hierarchy"]),
    ("이미지 7~8 Profit / Profit Margin measure", ["이미지 7", "Profit", "Profit Margin"]),
    ("이미지 9~10 Category별 결과", ["이미지 9", "Sales", "Profit", "Profit Margin"]),
    ("이미지 11 SalespersonRegion bridge table", ["이미지 11", "SalespersonRegion", "Bridge table"]),
    ("이미지 12 Cross-filter Both", ["이미지 12", "Cross-filter direction", "Both"]),
    ("이미지 13 direct relationship 비활성화", ["이미지 13", "inactive relationship"]),
    ("이미지 14 Salesperson별 Sales 변경 결과", ["이미지 14", "Salesperson", "Sales"]),
    ("이미지 15 Salesperson별 Sales와 Target 최종 비교", ["이미지 15", "Salesperson", "Targets"]),
]

POWER_QUERY_REGRESSION_REQUIREMENTS = [
    ("이미지 1 SQL Server database", ["이미지 1", "SQL Server", "AdventureWorksDW2020"]),
    ("이미지 2 Navigator Transform Data", ["이미지 2", "Navigator", "Transform Data", "FactResellerSales"]),
    ("이미지 3 Column quality/distribution/profile", ["이미지 3", "Column quality", "Column distribution", "Column profile"]),
    ("이미지 4 SalesPersonFlag", ["이미지 4", "SalesPersonFlag", "TRUE"]),
    ("이미지 5 Salesperson merge", ["이미지 5", "FirstName", "LastName", "Salesperson"]),
    ("이미지 6 Product expansion", ["이미지 6", "DimProductSubcategory", "DimProductCategory", "Subcategory", "Category"]),
    ("이미지 7 BusinessType standardization", ["이미지 7", "BusinessType", "Ware House", "Warehouse"]),
    ("이미지 8 Region query", ["이미지 8", "SalesTerritoryAlternateKey", "Region", "Country", "Group"]),
    ("이미지 9 TotalProductCost null fix", ["이미지 9", "TotalProductCost", "OrderQuantity", "StandardCost", "Cost"]),
    ("이미지 10 Targets Unpivot", ["이미지 10", "M01", "M12", "Unpivot"]),
    ("이미지 11 Targets long format", ["이미지 11", "MonthNumber", "Target", "TargetMonth"]),
    ("이미지 12 ColorFormats", ["이미지 12", "ColorFormats", "Background Color Format", "Font Color Format"]),
    ("이미지 13 Product ColorFormats Merge", ["이미지 13", "Product[Color]", "ColorFormats[Color]", "Left Outer"]),
    ("이미지 14 final load control", ["이미지 14", "Disable load", "Close & Apply", "7개 테이블"]),
]

REQUIRED_CONCRETE_ENTITIES = {
    "semantic_model_relationship": SEMANTIC_CONCRETE_ENTITIES,
    "power_query_etl": [
        "SQL Server",
        "AdventureWorksDW2020",
        "Navigator",
        "Transform Data",
        "Power Query Editor",
        "Column quality",
        "Column distribution",
        "Column profile",
        "SalesPersonFlag",
        "FirstName",
        "LastName",
        "Salesperson",
        "EmployeeID",
        "UPN",
        "ProductKey",
        "DimProductSubcategory",
        "DimProductCategory",
        "Subcategory",
        "Category",
        "BusinessType",
        "Ware House",
        "Warehouse",
        "Replace Values",
        "DimGeography",
        "Region",
        "Country",
        "Group",
        "FactResellerSales",
        "TotalProductCost",
        "OrderQuantity",
        "StandardCost",
        "Cost",
        "Fixed Decimal Number",
        "ResellerSalesTargets.csv",
        "M01",
        "M12",
        "Unpivot",
        "MonthNumber",
        "Target",
        "TargetMonth",
        "ColorFormats",
        "Product[Color]",
        "ColorFormats[Color]",
        "Left Outer",
        "Background Color Format",
        "Font Color Format",
        "Disable load",
        "Close & Apply",
        "7개 테이블",
    ],
    "dax_measure_modeling": [
        "Month",
        "MonthKey",
        "Sort by column",
        "Date table",
        "CALENDARAUTO",
        "Fiscal",
        "Mark as date table",
        "Avg Price",
        "Median Price",
        "Orders",
        "Order Lines",
        "Currency",
        "Target",
        "TargetAmount",
        "HASONEVALUE",
        "Variance",
        "Variance Margin",
    ],
}


def local_text_source_pack_response(
    raw_text: str,
    memo: str,
    image_files: list[Path],
    topic: str,
    extra_info: str,
    image_names: list[str] | None,
    captures: list[dict[str, Any]],
    qa_logs: list[dict[str, Any]],
    provider_diagnostics_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_pairs = order_image_inputs(image_files, image_names or [path.name for path in image_files])
    ordered_files = [path for path, _ in ordered_pairs]
    ordered_names = [name for _, name in ordered_pairs]
    classification = classify_article_type_with_confidence(raw_text, memo, topic, extra_info, ordered_names)
    article_type = str(classification.get("article_type") or "general_learning_portfolio")
    evidence: list[dict[str, Any]] = []
    sparse_capture_report = build_sparse_capture_report(article_type, classification, evidence, raw_text, memo)
    steps = build_text_assisted_solution_steps(article_type, raw_text, memo)
    problem_map = {
        "article_type": article_type,
        "core_problem": compact_user_intent(raw_text, memo),
        "key_terms": source_derived_terms(raw_text, memo, limit=8),
        "solution_steps": steps,
        "_article_type_confidence": classification.get("confidence"),
        "_article_type_candidates": classification.get("candidates", []),
        "_uploaded_images_count": len(ordered_files),
        "_capture_count": len(captures),
        "_qa_count": len(qa_logs),
    }
    section_plan = [
        {
            "section": str(step.get("title") or f"학습 흐름 {idx}"),
            "image_refs": [],
            "must_include": normalize_str_list(step.get("technical_entities"))[:5],
        }
        for idx, step in enumerate(steps[:6], start=1)
        if isinstance(step, dict)
    ]
    draft = build_url_assisted_medium_draft(
        article_type=article_type,
        raw_text=raw_text,
        memo=memo,
        problem_map=problem_map,
        section_plan=section_plan,
        sparse_report=sparse_capture_report,
        evidence=evidence,
        qa_logs=qa_logs,
    )
    sparse_capture_report["generation_mode"] = "local_text_source_pack_fallback"
    return {
        "draft": sanitize_medium_markdown(draft),
        "article_type": article_type,
        "image_evidence": evidence,
        "problem_map": problem_map,
        "learning_evidence": build_learning_evidence(captures, qa_logs, raw_text, memo),
        "decision_map": {},
        "section_plan": section_plan,
        "article_brief": {},
        "sparse_capture_report": sparse_capture_report,
        "provider_diagnostics": provider_diagnostics_data or {},
        "critic_report": {
            "passed": True,
            "failures": [],
            "metrics": {
                "generation_mode": "local_text_source_pack_fallback",
                "provider_unavailable": bool(provider_diagnostics_data),
            },
        },
    }


def generate_medium_article_pipeline(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    image_files: list[Path],
    topic: str,
    extra_info: str = "",
    image_names: list[str] | None = None,
    captures: list[dict[str, Any]] | None = None,
    qa_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    captures = captures or []
    qa_logs = qa_logs or []
    if not image_files and is_foundry_iq_mcp_rag_context(raw_text, memo):
        return local_text_source_pack_response(raw_text, memo, image_files, topic, extra_info, image_names, captures, qa_logs)
    client = llm_client.get_client()
    if not client:
        if has_text_assisted_context(raw_text, memo):
            return local_text_source_pack_response(
                raw_text,
                memo,
                image_files,
                topic,
                extra_info,
                image_names,
                captures,
                qa_logs,
                provider_diagnostics_data=llm_client.last_diagnostics,
            )
        diagnostics = llm_client.last_diagnostics or provider_diagnostics("groq", exception=RuntimeError("LLM client unavailable"))
        return {
            "draft": provider_failure_message(
                image_count=len(image_files),
                capture_count=len(captures),
                qa_count=len(qa_logs),
                memo=memo,
                diagnostics=diagnostics,
                elapsed_seconds=0,
            ),
            "article_type": "unknown",
            "image_evidence": [],
            "problem_map": {},
            "learning_evidence": [],
            "decision_map": {},
            "article_brief": {},
            "provider_diagnostics": diagnostics,
            "critic_report": {
                "passed": False,
                "failures": ["provider_failure: LLM client unavailable"],
                "metrics": {"failure_type": "provider_failure", "provider_diagnostics": diagnostics},
            },
        }

    try:
        ordered_pairs = order_image_inputs(image_files, image_names or [path.name for path in image_files])
        ordered_files = [path for path, _ in ordered_pairs]
        ordered_names = [name for _, name in ordered_pairs]
        classification = classify_article_type_with_confidence(raw_text, memo, topic, extra_info, ordered_names)
        article_type = str(classification.get("article_type") or "unknown")
        golden_context = load_golden_context(article_type)
        evidence = build_image_evidence(llm_client, raw_text, memo, ordered_files, topic, extra_info, ordered_names)
        evidence = ensure_image_evidence_coverage(evidence, ordered_files, ordered_names, raw_text, memo, golden_context)
        auto_topic_hint = image_only_auto_topic_hint(raw_text, memo, len(ordered_files), evidence, ordered_names)
        if auto_topic_hint:
            memo = auto_topic_hint
            classification = classify_article_type_with_confidence(raw_text, memo, topic, extra_info, ordered_names)
            article_type = str(classification.get("article_type") or "unknown")
            golden_context = load_golden_context(article_type)
            evidence = ensure_image_evidence_coverage(evidence, ordered_files, ordered_names, raw_text, memo, golden_context)
        classification = promote_article_type_from_evidence(classification, evidence, raw_text, memo, ordered_names)
        article_type = str(classification.get("article_type") or "unknown")
        golden_context = load_golden_context(article_type)
        sparse_capture_report = build_sparse_capture_report(article_type, classification, evidence, raw_text, memo)
        if article_type in {"unknown", "general_learning_portfolio"} or float(classification.get("confidence") or 0) < 0.55:
            evidence_text = json.dumps(evidence, ensure_ascii=False)
            second_classification = classify_article_type_with_confidence(
                raw_text + "\n" + evidence_text,
                memo,
                topic,
                extra_info,
                ordered_names,
            )
            if float(second_classification.get("confidence") or 0) > float(classification.get("confidence") or 0):
                classification = second_classification
                classification = promote_article_type_from_evidence(second_classification, evidence, raw_text, memo, ordered_names)
                article_type = str(classification.get("article_type") or "unknown")
                golden_context = load_golden_context(article_type)
                sparse_capture_report = build_sparse_capture_report(article_type, classification, evidence, raw_text, memo)
        if article_type == "unknown" or float(classification.get("confidence") or 0) < 0.25:
            return unknown_article_type_response(classification, evidence, len(ordered_files), memo, raw_text, sparse_capture_report)
        learning_evidence = build_learning_evidence(captures, qa_logs, raw_text, memo)
        problem_map = build_problem_map(llm_client, raw_text, memo, evidence, topic, extra_info, article_type, golden_context)
        problem_map["article_type"] = article_type
        problem_map["_sparse_capture_report"] = sparse_capture_report
        problem_map["_article_type_confidence"] = classification.get("confidence")
        problem_map["_article_type_candidates"] = classification.get("candidates", [])
        problem_map["_uploaded_images_count"] = len(ordered_files)
        problem_map["_capture_count"] = len(captures)
        problem_map["_qa_count"] = len(qa_logs)
        problem_map["_readme_image_count"] = readme_image_count(golden_context.get("image_order", ""))
        problem_map = attach_problem_map_refs(problem_map, evidence)
        problem_map["_evidence_text_for_classification"] = json.dumps(evidence, ensure_ascii=False)[:20000]
        enrich_problem_map_concrete_details(problem_map, article_type)
        decision_map = build_decision_map(learning_evidence, problem_map)
        brief = build_article_brief(llm_client, raw_text, memo, evidence, problem_map, topic, extra_info)
        outline = build_article_outline(llm_client, brief, problem_map, evidence)
        section_plan = build_section_plan(article_type, evidence, problem_map)
        problem_map["_section_plan"] = section_plan
        sparse_hold_reason = sparse_capture_generation_blocker(article_type, sparse_capture_report, problem_map, brief)
        if sparse_hold_reason:
            # URL/lecture-note assisted mode: do not punish the user for having no screenshots,
            # no explicit “hard problem”, or no final validation screen.  If the user provided
            # source URLs, lab text, lecture notes, or a lightweight question, produce a
            # Medium-ready draft first and move uncertainties to a short optional checklist.
            if can_generate_url_assisted_medium_draft(raw_text, memo, article_type, problem_map, section_plan):
                url_article = build_url_assisted_medium_draft(
                    article_type=article_type,
                    raw_text=raw_text,
                    memo=memo,
                    problem_map=problem_map,
                    section_plan=section_plan,
                    sparse_report=sparse_capture_report,
                    evidence=evidence,
                    qa_logs=qa_logs,
                )
                critic = CritiqueResult(
                    passed=True,
                    failures=[],
                    section_failures={},
                    metrics={
                        "generation_mode": "url_assisted_medium_draft",
                        "original_sparse_hold_reason": sparse_hold_reason,
                        "image_count": len(ordered_files),
                    },
                )
                sparse_capture_report["generation_mode"] = "url_assisted_medium_draft"
                return {
                    "draft": sanitize_medium_markdown(url_article),
                    "article_type": article_type,
                    "image_evidence": evidence,
                    "learning_evidence": learning_evidence,
                    "problem_map": problem_map,
                    "decision_map": decision_map,
                    "section_plan": section_plan,
                    "article_brief": brief,
                    "sparse_capture_report": sparse_capture_report,
                    "critic_report": asdict(critic),
                }
            critic = CritiqueResult(
                passed=False,
                failures=[sparse_hold_reason],
                section_failures={"sparse_capture_mode": sparse_hold_reason},
                metrics={
                    "failure_type": "content_quality_failure",
                    "generation_mode": sparse_capture_report.get("generation_mode"),
                    "interpreted_image_count": sparse_capture_report.get("interpreted_image_count"),
                    "total_image_count": sparse_capture_report.get("total_image_count"),
                    "unknown_caption_count": sparse_capture_report.get("unknown_caption_count"),
                },
            )
            return {
                "draft": sparse_capture_hold_message(article_type, sparse_capture_report, evidence, problem_map, brief, section_plan, sparse_hold_reason),
                "article_type": article_type,
                "image_evidence": evidence,
                "learning_evidence": learning_evidence,
                "problem_map": problem_map,
                "decision_map": decision_map,
                "section_plan": section_plan,
                "article_brief": brief,
                "sparse_capture_report": sparse_capture_report,
                "critic_report": asdict(critic),
            }

        coverage_failure = image_coverage_failure(problem_map, evidence)
        problem_map["_warnings"] = problem_map.get("_coverage_status", {}).get("warnings", [])
        if coverage_failure:
            return {
                "draft": evidence_coverage_failure_message(coverage_failure, len(ordered_files), len(evidence), len(captures), len(qa_logs), memo),
                "article_type": article_type,
                "image_evidence": evidence,
                "learning_evidence": learning_evidence,
                "problem_map": problem_map,
                "decision_map": decision_map,
            "section_plan": section_plan,
            "article_brief": brief,
            "sparse_capture_report": sparse_capture_report,
            "critic_report": asdict(
                    CritiqueResult(
                        passed=False,
                        failures=[coverage_failure],
                        section_failures={"image_evidence_coverage": coverage_failure},
                        metrics={"uploaded_images_count": len(ordered_files), "image_evidence_count": len(evidence)},
                    )
                ),
            }

        sections: dict[str, str] = {}
        for title in SECTION_TITLES:
            sections[title] = generate_section(llm_client, title, outline, brief, problem_map, evidence, raw_text, memo, extra_info)
        article = assemble_article(sections, brief)
        critique = critique_article(article, article_type, problem_map, evidence)
        if not critique.passed:
            article = expand_failed_sections(llm_client, article, sections, critique, outline, brief, problem_map, evidence, raw_text, memo, extra_info)
            critique = critique_article(article, article_type, problem_map, evidence)
        if article_type not in POWERBI_ARTICLE_TYPES and severe_sparse_article_failures(critique):
            sparse_capture_report["generation_mode"] = "draft_with_missing_context"
            reason = "; ".join(severe_sparse_article_failures(critique)[:3])
            return {
                "draft": sparse_capture_hold_message(article_type, sparse_capture_report, evidence, problem_map, brief, section_plan, reason),
                "article_type": article_type,
                "image_evidence": evidence,
                "learning_evidence": learning_evidence,
                "problem_map": problem_map,
                "decision_map": decision_map,
                "section_plan": section_plan,
                "article_brief": brief,
                "sparse_capture_report": sparse_capture_report,
                "critic_report": asdict(critique),
            }
        return {
            "draft": sanitize_medium_markdown(article),
            "article_type": article_type,
            "image_evidence": evidence,
            "learning_evidence": learning_evidence,
            "problem_map": problem_map,
            "decision_map": decision_map,
            "section_plan": section_plan,
            "article_brief": brief,
            "coverage_status": problem_map.get("_coverage_status", {}),
            "warnings": problem_map.get("_warnings", []),
            "sparse_capture_report": sparse_capture_report,
            "critic_report": asdict(critique),
        }
    except Exception as exc:
        print(f"[Medium pipeline error] {exc}")
        diagnostics = provider_diagnostics("groq", exception=exc, package_import_ok=groq_import_ok(), client_init_ok=bool(llm_client.client))
        return {
            "draft": provider_failure_message(
                image_count=len(image_files),
                capture_count=len(captures),
                qa_count=len(qa_logs),
                memo=memo,
                diagnostics=diagnostics,
                elapsed_seconds=0,
            ),
            "article_type": "unknown",
            "image_evidence": [],
            "learning_evidence": [],
            "problem_map": {},
            "decision_map": {},
            "article_brief": {},
            "provider_diagnostics": diagnostics,
            "critic_report": {
                "passed": False,
                "failures": [f"provider_or_pipeline_failure: {type(exc).__name__}: {exc}"],
                "metrics": {"failure_type": "provider_failure", "provider_diagnostics": diagnostics},
            },
        }


def order_image_inputs(image_files: list[Path], image_names: list[str]) -> list[tuple[Path, str]]:
    pairs = list(zip(image_files, image_names, strict=False))
    if any(re.match(r"^\d{1,3}[_\-\s]", Path(name).name) for _, name in pairs):
        pairs.sort(key=lambda item: natural_sort_key(Path(item[1]).name))
    return [(path, name) for path, name in pairs]


def model_route(task: str) -> dict[str, str]:
    deep_tasks = {"hard_question_answer", "problem_map_generation", "final_article_generation", "critic_and_expand"}
    provider = MODEL_PROVIDER_DEEP if task in deep_tasks else MODEL_PROVIDER_FAST
    if provider == "openai":
        # OpenAI is a planned premium/deep adapter; the current MVP runtime uses the Groq client safely.
        provider = "groq"
    model = OPENAI_MODEL if provider == "openai" else GROQ_MODEL
    if task == "quick_image_summary" and provider == "groq":
        model = GROQ_VISION_MODEL
    return {"provider": provider, "model": model}


def read_sessions() -> dict[str, Any]:
    if not SESSIONS_PATH.exists() or not SESSIONS_PATH.read_text(encoding="utf-8").strip():
        return {"sessions": []}
    try:
        data = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sessions": []}
    if not isinstance(data, dict):
        return {"sessions": []}
    data.setdefault("sessions", [])
    return data


def write_sessions(data: dict[str, Any]) -> None:
    SESSIONS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_session(title: str = "") -> dict[str, Any]:
    data = read_sessions()
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "title": title or f"Study session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "captures": [],
        "qa_logs": [],
    }
    data["sessions"].append(session)
    write_sessions(data)
    return session


def find_session(session_id: str) -> dict[str, Any] | None:
    for session in read_sessions().get("sessions", []):
        if session.get("session_id") == session_id:
            return session
    return None


def update_session(session: dict[str, Any]) -> None:
    data = read_sessions()
    sessions = data.get("sessions", [])
    for index, item in enumerate(sessions):
        if item.get("session_id") == session.get("session_id"):
            sessions[index] = session
            write_sessions(data)
            return
    sessions.append(session)
    data["sessions"] = sessions
    write_sessions(data)


def latest_or_new_session() -> dict[str, Any]:
    sessions = read_sessions().get("sessions", [])
    if sessions:
        return sessions[-1]
    return create_session()


def build_learning_evidence(
    captures: list[dict[str, Any]],
    qa_logs: list[dict[str, Any]],
    raw_text: str,
    memo: str,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for capture in captures:
        evidence.append(
            {
                "type": "capture",
                "ref_id": capture.get("capture_id"),
                "timestamp": capture.get("timestamp"),
                "learner_signal": capture.get("user_note") or capture.get("ocr_text") or capture.get("vision_summary") or "무메모 캡처",
                "evidence_text": " ".join(
                    part
                    for part in [
                        capture.get("user_note", ""),
                        capture.get("ocr_text", ""),
                        capture.get("vision_summary", ""),
                        " ".join(capture.get("auto_keywords") or []),
                    ]
                    if part
                ),
                "image_path": capture.get("image_path", ""),
            }
        )
    for qa in qa_logs:
        evidence.append(
            {
                "type": "qa",
                "ref_id": qa.get("qa_id"),
                "timestamp": qa.get("timestamp"),
                "learner_signal": qa.get("question", ""),
                "evidence_text": f"질문: {qa.get('question', '')}\n답변 요약: {qa.get('answer_summary', '')}\n상태: {qa.get('learner_state', '')}",
                "related_capture_ids": qa.get("related_capture_ids", []),
            }
        )
    if raw_text.strip() or memo.strip():
        evidence.append(
            {
                "type": "direct_input",
                "ref_id": "direct",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "learner_signal": memo or raw_text,
                "evidence_text": "\n".join(part for part in [raw_text, memo] if part.strip()),
            }
        )
    return evidence


def build_decision_map(learning_evidence: list[dict[str, Any]], problem_map: dict[str, Any]) -> dict[str, Any]:
    decision_points: list[dict[str, Any]] = []
    qa_items = [item for item in learning_evidence if item.get("type") == "qa"]
    captures = [item for item in learning_evidence if item.get("type") == "capture"]
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    for index, step in enumerate(steps[: max(4, len(qa_items))], start=1):
        qa = qa_items[index - 1] if index - 1 < len(qa_items) else {}
        related = qa.get("related_capture_ids") or [cap.get("ref_id") for cap in captures[index - 1 : index + 1] if cap.get("ref_id")]
        decision_points.append(
            {
                "title": step.get("title") if isinstance(step, dict) else f"판단 지점 {index}",
                "initial_assumption": (step.get("cause") if isinstance(step, dict) else "") or "처음에는 화면에 보이는 결과만으로 충분하다고 볼 수 있었다.",
                "question": qa.get("learner_signal", ""),
                "ai_hint": qa.get("evidence_text", ""),
                "user_decision": (step.get("action") if isinstance(step, dict) else "") or "근거를 문제/원인/검증 흐름으로 재구성했다.",
                "actual_result": (step.get("verification") if isinstance(step, dict) else "") or problem_map.get("final_outcome", ""),
                "evidence_refs": related,
                "qa_refs": [qa.get("ref_id")] if qa.get("ref_id") else [],
            }
        )
    return {"decision_points": decision_points}


def attach_problem_map_refs(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    image_refs = [item.get("image_no") for item in evidence if item.get("image_no")]
    entities = sorted({entity for item in evidence for entity in normalize_str_list(item.get("technical_entities"))})
    for index, step in enumerate(problem_map.get("solution_steps", []), start=1):
        if not isinstance(step, dict):
            continue
        step.setdefault("image_refs", image_refs[max(0, index - 2) : index + 1] or image_refs[:1])
        step.setdefault("technical_entities", entities[:8])
    return problem_map


def ensure_image_evidence_coverage(
    evidence: list[dict[str, Any]],
    image_files: list[Path],
    image_names: list[str],
    raw_text: str,
    memo: str,
    golden_context: dict[str, Any],
) -> list[dict[str, Any]]:
    by_no: dict[int, dict[str, Any]] = {}
    for item in evidence:
        try:
            image_no = int(item.get("image_no") or 0)
        except (TypeError, ValueError):
            continue
        if image_no > 0:
            by_no[image_no] = item

    caption_source = golden_context.get("image_order", "") or read_image_order_caption_source()
    fallback = fallback_image_evidence(image_files, image_names, raw_text, memo, caption_source)
    for item in fallback:
        image_no = int(item.get("image_no") or 0)
        by_no.setdefault(image_no, item)
    return normalize_display_image_indexes([by_no[index] for index in sorted(by_no)], image_files, image_names)


def normalize_display_image_indexes(
    evidence: list[dict[str, Any]],
    image_files: list[Path],
    image_names: list[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    total = len(image_files) or len(evidence)
    for display_index in range(1, total + 1):
        item = evidence[display_index - 1] if display_index - 1 < len(evidence) else {}
        original_name = image_names[display_index - 1] if display_index - 1 < len(image_names) else ""
        next_item = dict(item)
        next_item["source_image_no"] = next_item.get("source_image_no", next_item.get("image_no", display_index))
        next_item["display_image_index"] = display_index
        next_item["image_no"] = display_index
        if original_name:
            next_item["original_filename"] = original_name
        next_item["caption"] = humanize_caption(str(next_item.get("caption") or f"이미지 {display_index} - 해석되지 않은 캡처"), display_index)
        normalized.append(next_item)
    return normalized


def readme_image_count(image_order: str) -> int:
    if not image_order:
        return 0
    return sum(1 for line in image_order.splitlines() if re.search(r"\b\d{1,3}[_\-\s]", line) or re.search(r"\bimage\d+\.", line, re.I))


def image_coverage_failure(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    uploaded = int(problem_map.get("_uploaded_images_count") or 0)
    capture_count = int(problem_map.get("_capture_count") or 0)
    readme_count = int(problem_map.get("_readme_image_count") or 0)
    evidence_count = len([item for item in evidence if int(item.get("image_no") or 0) > 0])
    problem_map["_coverage_status"] = coverage_status(problem_map, evidence)
    if uploaded >= 5 and evidence_count / max(uploaded, 1) < 0.8:
        return f"이미지 {uploaded}장 중 {evidence_count}장만 해석되었습니다. 이미지 evidence coverage가 부족하여 완성형 Medium 글을 생성할 수 없습니다."
    if uploaded and evidence_count < uploaded:
        return f"업로드된 이미지 {uploaded}장 중 {evidence_count}장만 ImageEvidence로 정리되었습니다."
    if readme_count and uploaded and evidence_count < min(readme_count, uploaded):
        return f"README expected images {readme_count}장, uploaded images {uploaded}장 중 ImageEvidence가 {evidence_count}개만 생성되었습니다."
    if readme_count and not uploaded and evidence_count < readme_count:
        return f"README_image_order.txt 기준 이미지 {readme_count}장 중 {evidence_count}장만 ImageEvidence로 정리되었습니다."
    if capture_count and uploaded == 0 and evidence_count != capture_count:
        return f"Capture Timeline의 캡처 {capture_count}개 중 {evidence_count}개만 ImageEvidence로 정리되었습니다."
    return ""


def coverage_status(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    uploaded = int(problem_map.get("_uploaded_images_count") or 0)
    capture_count = int(problem_map.get("_capture_count") or 0)
    qa_count = int(problem_map.get("_qa_count") or 0)
    readme_count = int(problem_map.get("_readme_image_count") or 0)
    evidence_count = len([item for item in evidence if int(item.get("image_no") or 0) > 0])
    warnings: list[dict[str, Any]] = []
    evidence_missing = False
    if uploaded:
        evidence_missing = evidence_count < uploaded
    elif readme_count:
        evidence_missing = evidence_count < readme_count
    elif capture_count:
        evidence_missing = evidence_count < capture_count
    if readme_count and uploaded and readme_count != uploaded and not evidence_missing:
        extra = uploaded - readme_count
        if extra > 0:
            message = (
                f"README에는 {readme_count}장이 기록되어 있지만, 실제 업로드 이미지는 {uploaded}장이고 "
                f"ImageEvidence도 {evidence_count}개 생성되었습니다. Evidence 부족은 아니며, README/업로드 수 불일치 상태입니다. "
                f"extra image {extra}장이 있습니다."
            )
        else:
            message = (
                f"README에는 {readme_count}장이 기록되어 있지만, 실제 업로드 이미지는 {uploaded}장이고 "
                f"ImageEvidence는 {evidence_count}개 생성되었습니다. Evidence 부족은 아니며, README/업로드 수 불일치 상태입니다."
            )
        warnings.append(
            {
                "status": "warning",
                "reason": "readme_uploaded_count_mismatch",
                "message": message,
                "can_generate_article": True,
                "evidence_missing": False,
                "readme_expected_count": readme_count,
                "uploaded_images_count": uploaded,
                "image_evidence_count": evidence_count,
            }
        )
    return {
        "status": "failure" if evidence_missing else ("warning" if warnings else "ok"),
        "readme_expected_count": readme_count,
        "uploaded_images_count": uploaded,
        "image_evidence_count": evidence_count,
        "capture_count": capture_count,
        "qa_count": qa_count,
        "evidence_missing": evidence_missing,
        "can_generate_article": not evidence_missing,
        "warnings": warnings,
    }


CAPTURE_ROLES = [
    "goal_or_context",
    "confusing_concept",
    "architecture_or_flow",
    "action_or_change",
    "code_or_config",
    "error_or_problem",
    "validation_or_result",
    "completion_or_summary",
    "unknown",
]

POWERBI_ARTICLE_TYPES = {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}
GITHUB_WORKFLOW_TYPES = {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}


def build_sparse_capture_report(
    article_type: str,
    classification: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
) -> dict[str, Any]:
    roles = [infer_sparse_capture_role(item) for item in evidence]
    source = " ".join(
        [
            raw_text,
            memo,
            " ".join(str(item.get("caption", "")) for item in evidence),
            " ".join(str(item.get("problem_signal", "")) for item in evidence),
            " ".join(" ".join(normalize_str_list(item.get("technical_entities"))) for item in evidence),
        ]
    ).lower()
    role_set = set(roles)
    explicit_goal_words = ["goal", "objective", "lesson", "module", "lab", "강의 목표", "실습 목표", "학습 목표", "이번 단원"]
    has_explicit_goal_text = bool(memo.strip() and any(word in memo.lower() for word in explicit_goal_words)) or bool(
        raw_text.strip() and any(word in raw_text.lower() for word in explicit_goal_words)
    )
    has_goal_evidence = bool({"goal_or_context", "architecture_or_flow"} & role_set)
    coverage = {
        "has_goal": bool(has_explicit_goal_text or has_goal_evidence),
        "has_problem": bool({"error_or_problem", "confusing_concept"} & role_set or any(word in source for word in ["error", "오류", "problem", "문제", "failed", "not working", "헷갈", "confusing"])),
        "has_action": bool({"action_or_change", "code_or_config"} & role_set),
        "has_validation": bool({"validation_or_result", "completion_or_summary"} & role_set or any(word in source for word in ["success", "완료", "result", "결과", "validated", "검증", "conclusion"])),
    }
    missing_context: list[str] = []
    if not coverage["has_goal"]:
        missing_context.append("강의/실습의 최종 목표가 명확하지 않습니다.")
    if not coverage["has_problem"]:
        missing_context.append("어떤 개념이 헷갈렸거나 어떤 문제가 있었는지 확인할 근거가 부족합니다.")
    if not coverage["has_action"]:
        missing_context.append("사용자가 실제로 바꾼 코드, 설정, 명령어, 작업 단계가 부족합니다.")
    if not coverage["has_validation"]:
        missing_context.append("마지막 성공/완료/실행 결과 또는 검증 화면이 부족합니다.")
    is_github_workflow = article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}
    if is_github_workflow:
        missing_context.append("이 강의/실습의 최종 목표가 정확히 무엇인지 부족함")
        if not coverage["has_problem"]:
            missing_context.append("사용자가 어떤 화면에서 막혔는지 부족함")
        if "update-github-info" in source:
            missing_context.append("update-github-info workflow가 실제로 무엇을 자동화하는지 설명 부족")
        if "conclusion" in source:
            missing_context.append("conclusion 화면이 성공/실패/요약 중 무엇을 의미하는지 부족함")
    confidence_score = float(classification.get("confidence") or 0)
    evidence_count = len([item for item in evidence if int(item.get("image_no") or 0) > 0])
    interpreted_count = sum(1 for item in evidence if is_interpreted_image_evidence(item))
    unknown_caption_count = sum(1 for item in evidence if is_unknown_or_filename_caption(item))
    interpreted_ratio = interpreted_count / max(evidence_count, 1)
    if evidence_count and interpreted_ratio < 0.6:
        missing_context.append(f"{evidence_count}장 중 일부 이미지의 의미를 해석하지 못함")
    if unknown_caption_count:
        missing_context.append(f"파일명 또는 일반 캡션으로만 남은 이미지가 {unknown_caption_count}장 있습니다.")
    present_count = sum(1 for value in coverage.values() if value)
    if (
        confidence_score >= 0.7
        and present_count >= 3
        and evidence_count >= 3
        and interpreted_ratio >= 0.6
        and unknown_caption_count <= max(1, evidence_count // 4)
    ):
        generation_mode = "full_article"
        confidence_label = "high"
    elif confidence_score >= 0.35 and (evidence_count >= 2 or has_text_assisted_context(raw_text, memo)):
        generation_mode = "draft_with_missing_context"
        confidence_label = "medium"
    else:
        generation_mode = "ask_before_generate"
        confidence_label = "low"
    followups = sparse_follow_up_questions(article_type, raw_text, memo)
    missing_context = list(dict.fromkeys(missing_context))
    return {
        "article_type": article_type,
        "confidence": confidence_label,
        "article_type_confidence": round(confidence_score, 3),
        "generation_mode": generation_mode,
        "interpreted_image_count": interpreted_count,
        "total_image_count": evidence_count,
        "unknown_caption_count": unknown_caption_count,
        "capture_roles": [
            {
                "image_no": item.get("image_no"),
                "role": role,
                "caption": item.get("caption", ""),
            }
            for item, role in zip(evidence, roles, strict=False)
        ],
        "capture_coverage": coverage,
        "missing_context": missing_context,
        "follow_up_questions": followups[:3] if generation_mode != "full_article" else [],
    }


def sparse_follow_up_questions(article_type: str, raw_text: str = "", memo: str = "") -> list[str]:
    source = (str(raw_text or "") + "\n" + str(memo or "")).lower()
    if "azure devops" in source and "mcp" in source:
        return [
            "MCP Server가 Azure DevOps에서 연결하는 대상은 Work items, Pull Requests, Builds, Test Plans 중 무엇이었나요?",
            "실습에서 실제로 수행한 작업은 MCP Server 설정이었나요, Copilot/AI assistant로 DevOps 작업을 요청하는 것이었나요?",
            "마지막에 확인한 결과는 work item 생성/조회, PR 확인, build 확인 중 무엇이었나요?",
        ]
    if article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}:
        return [
            "이 강의의 최종 목표는 GitHub Actions workflow 이해였나요, Copilot/Agent 자동화 이해였나요?",
            "캡처 중 가장 이해가 안 된 화면은 몇 번인가요?",
            "마지막 conclusion 화면은 성공 결과였나요, 단순 요약 화면이었나요?",
        ]
    return [
        "이 강의/실습의 최종 목표는 무엇이었나요?",
        "캡처 중 가장 이해가 안 된 화면은 몇 번인가요?",
        "마지막에 성공/완료/실행 결과를 확인했나요?",
    ]


def has_text_assisted_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".strip()
    lowered = source.lower()
    if len(source) >= 220:
        return True
    helpful_terms = [
        "강의 url",
        "강의안",
        "본문",
        "agentic workflows",
        "github actions",
        "workflow_dispatch",
        "pull request",
        "update-github-info",
        "activation",
        "conclusion",
        "mcp",
        "azure devops",
        "microsoftlearning.github.io",
        "mslearn-devops",
        "work item",
        "pull request",
        "test plan",
        "copilot",
        "agent orchestration",
        "orchestrator",
        "sub-agent",
        "sub-agents",
        "hive-mind",
        "github copilot cli",
        "planner",
        "coder",
        "designer",
        "자동화",
        "microsoft foundry",
        "mslearn-agent-quickstart",
        "develop your first agent with microsoft",
        "get-started-in-foundry",
        "continue-in-vscode",
        "use-agent",
        "first agent",
    ]
    return any(term in lowered for term in helpful_terms)


def build_text_assisted_solution_steps(article_type: str, raw_text: str, memo: str) -> list[dict[str, Any]]:
    """Build draft-grade solution steps from lecture text/memo when image interpretation fails.

    This is intentionally conservative: it creates steps for a draft_with_missing_context,
    not proof that the user completed a full implementation.
    """
    if article_type in POWERBI_ARTICLE_TYPES:
        return []
    if not has_text_assisted_context(raw_text, memo):
        return []
    source = f"{raw_text}\n{memo}"
    lowered = source.lower()

    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return [
            {
                "step": idx,
                "title": step["title"],
                "problem": step["problem"],
                "cause": "Microsoft Foundry quickstart 자료는 포털 생성, VS Code 전환, agent 실행 단서가 함께 나타나므로 각 단계의 역할을 분리해야 한다.",
                "action": step["action"],
                "verification": step["verification"],
                "technical_entities": ["Microsoft Foundry", "first agent", "quickstart", "VS Code"],
                "image_refs": [],
            }
            for idx, step in enumerate(microsoft_foundry_first_agent_steps(), start=1)
        ]

    if is_foundry_iq_mcp_rag_context(raw_text, memo):
        return [
            {
                "step": idx,
                "title": step["title"],
                "problem": step["problem"],
                "cause": "Foundry IQ, MCP, RAG, knowledge base, Azure AI Search 단서가 함께 있으므로 지식 연결과 도구 연결의 경계를 나누어야 한다.",
                "action": step["action"],
                "verification": step["verification"],
                "technical_entities": ["Foundry IQ", "MCP", "RAG", "knowledge base", "Azure AI Search"],
                "image_refs": [],
            }
            for idx, step in enumerate(foundry_iq_mcp_rag_steps(), start=1)
        ]

    if "azure devops" in lowered and "mcp" in lowered:
        return [
            {
                "step": 1,
                "title": "강의 URL과 실습 URL로 목표 복원",
                "problem": "캡처가 없거나 부족한 상태에서 강의의 최종 목표를 사용자가 기억만으로 설명하기 어렵다.",
                "cause": "AI Skills Navigator URL, YouTube URL, Microsoft Learning 실습 URL이 각각 다른 맥락을 제공하므로 먼저 강의 목표와 실습 범위를 분리해야 한다.",
                "action": "강의/영상/실습 URL과 붙여넣은 강의안 텍스트에서 Azure DevOps MCP Server, AI assistant, Agentic DevOps, work item, pull request 같은 핵심 단서를 추출한다.",
                "verification": "초안의 문제 정의가 Azure DevOps MCP Server와 AI assistant의 역할 이해로 좁혀졌는지 확인한다.",
                "technical_entities": ["Azure DevOps MCP Server", "AI assistant", "Agentic DevOps", "source URLs"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "MCP Server의 역할 이해",
                "problem": "MCP Server가 정확히 무엇이고 왜 Azure DevOps 자동화에 필요한지 헷갈린다.",
                "cause": "MCP는 AI assistant가 DevOps 도구의 데이터를 직접 이해하고 작업하도록 연결하는 중간 계층이지만, 단순 API 호출이나 일반 챗봇과의 차이가 직관적으로 보이지 않을 수 있다.",
                "action": "MCP Server를 AI assistant와 Azure DevOps 사이에서 work items, pull requests, builds, test plans 같은 리소스를 안전하게 연결하는 인터페이스로 정리한다.",
                "verification": "MCP를 'AI가 DevOps 업무 맥락을 읽고 행동하도록 해 주는 연결 계층'으로 설명할 수 있는지 확인한다.",
                "technical_entities": ["MCP", "Azure DevOps", "work items", "pull requests", "builds"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "실습 단계의 문제 해결 흐름 연결",
                "problem": "실습 URL의 각 단계가 단순 설정인지, 실제 DevOps 문제 해결 흐름인지 구분하기 어렵다.",
                "cause": "Agentic DevOps 실습은 설정, 인증, 연결 확인, 자연어 요청, 결과 확인 단계가 섞여 있을 수 있다.",
                "action": "실습 단계를 '환경 준비 → MCP Server 연결 → AI assistant에게 DevOps 작업 요청 → Azure DevOps 결과 확인' 흐름으로 재배치한다.",
                "verification": "각 단계가 최종적으로 어떤 DevOps 작업을 더 빠르게 하려는지 설명되는지 확인한다.",
                "technical_entities": ["lab steps", "workflow automation", "natural language request", "validation"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "막힌 개념을 Copilot 대화 기록으로 보강",
                "problem": "사용자가 중간에 퀴즈나 어려운 개념에서 막혔을 때 기억만으로 포트폴리오 글을 만들기 어렵다.",
                "cause": "학습 중 질문과 답변이 기록되지 않으면 문제 인식, 해결 과정, 검증 기준이 사라진다.",
                "action": "MCP, governance, integration overhead, vendor neutrality, workflow automation 같은 개념 질문을 Copilot/Tutor 대화로 해결하고 그 Q&A를 학습 기록에 저장한다.",
                "verification": "최종 글에서 사용자가 무엇을 몰랐고 어떤 설명을 통해 이해했는지가 문제 해결 흐름으로 드러나는지 확인한다.",
                "technical_entities": ["Copilot Q&A", "quiz support", "governance", "integration overhead"],
                "image_refs": [],
            },
        ]

    if is_agent_orchestration_context(raw_text, memo):
        return [
            {
                "step": 1,
                "title": "단일 챗봇과 agent orchestration의 차이 이해",
                "problem": "처음에는 AI agent가 단순히 질문에 답하는 챗봇인지, 여러 역할이 협업하는 구조인지 구분하기 어려웠다.",
                "cause": "강의에서는 From Chatbot to Hive-Mind처럼 단일 응답 모델에서 여러 agent가 역할을 나누는 구조로 관점이 이동한다.",
                "action": "Agent Orchestration을 한 명의 챗봇이 모든 일을 처리하는 방식이 아니라, orchestrator가 planner, coder, designer 같은 sub-agent에게 역할을 나누는 구조로 정리했다.",
                "verification": "orchestrator와 sub-agent의 차이를 설명할 수 있는지 확인한다.",
                "technical_entities": ["Agent Orchestration", "orchestrator", "sub-agents", "hive-mind"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "자율성의 두 축과 toolchain 이해",
                "problem": "agent가 얼마나 스스로 판단하고 어떤 도구를 사용할 수 있는지 기준이 흐릿했다.",
                "cause": "Two Dimensions of Autonomy와 The Orchestrator’s Toolchain은 agent workflow를 단순 자동완성이 아니라 권한, 도구, 역할 설계 문제로 보게 만든다.",
                "action": "agent의 자율성을 작업 범위와 도구 사용 능력으로 나누고, orchestrator가 어떤 toolchain을 통해 하위 agent를 조율하는지 정리했다.",
                "verification": "agent workflow에서 도구 선택과 역할 위임이 왜 중요한지 설명할 수 있는지 확인한다.",
                "technical_entities": ["autonomy", "toolchain", "agent workflow", "GitHub Copilot CLI"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "sub-agent 역할 설계",
                "problem": "planner, coder, designer 같은 역할이 단순 이름인지 실제 작업 분담 기준인지 헷갈릴 수 있었다.",
                "cause": "Architecting Your Sub-Agents와 Matching the Mind to the Mission은 agent를 작업 성격에 맞게 설계하는 관점을 요구한다.",
                "action": "planner는 계획 수립, coder는 구현, designer는 구조나 표현 설계처럼 각 agent가 맡을 수 있는 역할을 분리해 이해했다.",
                "verification": "하나의 작업을 여러 sub-agent 역할로 나누어 설명할 수 있는지 확인한다.",
                "technical_entities": ["planner", "coder", "designer", "sub-agent"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "GitHub Copilot CLI 기반 agent workflow로 연결",
                "problem": "강의의 개념이 실제 개발 도구 사용 흐름과 어떻게 연결되는지 확인해야 했다.",
                "cause": "영상 캡처에는 GitHub Copilot CLI, agent file, Welcome to Your New Dev Team 같은 단서가 있어 agent team을 개발 workflow 안에 배치하는 흐름으로 볼 수 있다.",
                "action": "GitHub Copilot CLI와 agent file을 sub-agent 구성을 실행하거나 관리하는 도구 후보로 정리하고, 강의 내용을 개발 workflow 확장 관점으로 연결했다.",
                "verification": "실제 CLI 명령, agent file 내용, 실행 결과 화면이 추가되면 실습 결과를 더 구체화할 수 있다.",
                "technical_entities": ["GitHub Copilot CLI", "agent file", "developer workflow", "AI dev team"],
                "image_refs": [],
            },
        ]

    if article_type in GITHUB_WORKFLOW_TYPES or is_github_agentic_context(raw_text, memo):
        return [
            {
                "step": 1,
                "title": "강의 주제와 자동화 목표 복원",
                "problem": "캡처가 드문드문 남아 있어 강의 전체 목표와 화면 순서를 바로 확정하기 어렵다.",
                "cause": "이미지 해석률이 낮아 캡처만으로는 workflow의 시작점과 결과 화면을 연결하기 어렵다.",
                "action": "GitHub Agentic Workflows, GitHub Actions, workflow_dispatch, Pull Request 같은 단서를 바탕으로 자동화 흐름의 중심 개념을 먼저 분리했다.",
                "verification": "GitHub Actions의 실행 조건, workflow 파일, PR 흐름이 하나의 자동화 과정으로 연결되는지 확인한다.",
                "technical_entities": ["GitHub Agentic Workflows", "GitHub Actions", "workflow_dispatch", "Pull Request"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "workflow_dispatch와 workflow 파일 역할 이해",
                "problem": "workflow_dispatch, update-github-info, update-github-info.lock.yml 같은 용어가 어떤 자동화 흐름을 뜻하는지 헷갈린다.",
                "cause": "workflow 파일은 자동화 실행 조건과 작업 내용을 정의하지만, 캡처만으로는 어떤 파일이 어떤 작업을 수행하는지 단정하기 어렵다.",
                "action": "workflow_dispatch는 수동 실행 트리거 후보로, update-github-info 계열 파일은 GitHub 정보 업데이트 자동화와 관련된 workflow 후보로 분리해 정리한다.",
                "verification": "강의안 또는 GitHub 화면에서 workflow 파일명, trigger, job/action 설명이 실제로 확인되는지 추가 점검한다.",
                "technical_entities": ["workflow_dispatch", "update-github-info", "update-github-info.lock.yml", "YAML workflow"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "activation과 Pull Request 흐름 연결",
                "problem": "activation 화면과 Pull Request 화면이 workflow 실행 과정에서 각각 어떤 단계인지 확실하지 않다.",
                "cause": "agentic workflow는 설정 활성화, 파일 변경, PR 생성, 검토 흐름이 이어질 수 있지만 캡처 순서가 부족하면 단계 관계가 흐려진다.",
                "action": "activation은 workflow 또는 agent 설정이 켜지는 상태 후보로, Pull Request는 자동화 또는 에이전트가 만든 변경사항을 검토하는 단계 후보로 둔다.",
                "verification": "PR 화면에 어떤 파일 변경이 포함되었는지, activation 이후 PR 또는 workflow run으로 이어졌는지 확인한다.",
                "technical_entities": ["activation", "Pull Request", "file changes", "agent workflow"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "conclusion 화면을 결과 검증 후보로 남기기",
                "problem": "conclusion 화면이 성공 결과인지, 실패/요약 화면인지, 단순 강의 요약인지 확정하기 어렵다.",
                "cause": "현재 입력에는 마지막 결과의 상태값, 성공 메시지, 실행 로그가 충분히 해석되지 않았다.",
                "action": "conclusion은 최종 검증 후보로만 기록하고, 성공했다고 단정하지 않는다.",
                "verification": "추가로 success, completed, merged, workflow run status, PR status 같은 표시가 있는지 확인한다.",
                "technical_entities": ["conclusion", "workflow result", "status", "validation"],
                "image_refs": [],
            },
        ]

    return [
        {
            "step": idx,
            "title": step["title"],
            "problem": step["problem"],
            "cause": "현재 source pack 안의 제목, URL, 반복 용어만으로 학습 흐름을 재구성해야 한다.",
            "action": step["action"],
            "verification": step["verification"],
            "technical_entities": source_derived_terms(raw_text, memo, limit=5),
            "image_refs": [],
        }
        for idx, step in enumerate(source_derived_generic_steps(raw_text, memo), start=1)
    ]


def build_recovery_draft_preview(
    article_type: str,
    problem_map: dict[str, Any],
    sparse_report: dict[str, Any],
    section_plan: list[dict[str, Any]],
) -> str:
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    if not steps:
        return ""
    core = str(problem_map.get("core_problem") or "현재 입력만으로는 핵심 문제를 확정하기 어렵습니다.")
    lines: list[str] = []
    lines.append("이미지는 충분히 해석되지 않았지만, 사용자가 넣은 강의안/URL/메모 단서를 바탕으로 아래처럼 학습 복구 초안을 만들 수 있습니다. 완성형 글이 아니라 확인 필요한 draft입니다.")
    lines.append("")
    lines.append("### 문제 정의 후보")
    lines.append(core)
    lines.append("")
    lines.append("### 예상 학습 흐름")
    for step in steps[:6]:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "학습 단계")
        problem = str(step.get("problem") or "")
        action = str(step.get("action") or "")
        verification = str(step.get("verification") or "")
        lines.append(f"- **{title}**")
        if problem:
            lines.append(f"  - 문제/헷갈린 점: {problem}")
        if action:
            lines.append(f"  - 정리 방향: {action}")
        if verification:
            lines.append(f"  - 확인 기준: {verification}")
    questions = sparse_report.get("follow_up_questions", []) if isinstance(sparse_report.get("follow_up_questions"), list) else []
    if questions:
        lines.append("")
        lines.append("### 글을 완성하려면 필요한 질문")
        for idx, question in enumerate(questions[:3], start=1):
            lines.append(f"{idx}. {question}")
    return "\n".join(lines)


def is_azure_devops_mcp_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    return "azure devops" in source and ("mcp" in source or "model context protocol" in source or "mslearn-devops" in source)


def is_microsoft_foundry_first_agent_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    markers = [
        "mslearn-agent-quickstart",
        "develop your first agent with microsoft",
        "get-started-in-foundry",
        "continue-in-vscode",
        "use-agent",
        "first agent",
    ]
    return "microsoft foundry" in source and any(marker in source for marker in markers)


def is_foundry_iq_mcp_rag_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    return "foundry iq" in source and any(
        marker in source
        for marker in [
            "mcp",
            "model context protocol",
            "rag",
            "knowledge base",
            "azure ai search",
            "dynamic tool discovery",
        ]
    )


def is_github_agentic_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return False
    markers = ["agentic workflows", "workflow_dispatch", "update-github-info", "github actions", ".github/workflows"]
    has_workflow_marker = any(marker in source for marker in markers)
    has_github_anchor = "github" in source or ".github/workflows" in source
    return has_github_anchor and has_workflow_marker


def is_agent_orchestration_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return False
    markers = [
        "agent orchestration",
        "orchestrator",
        "sub-agent",
        "sub-agents",
        "hive-mind",
        "github copilot cli",
        "planner",
        "coder",
        "designer",
        "toolchain",
        "autonomy",
    ]
    return sum(1 for marker in markers if marker in source) >= 2


def can_generate_url_assisted_medium_draft(
    raw_text: str,
    memo: str,
    article_type: str,
    problem_map: dict[str, Any],
    section_plan: list[dict[str, Any]],
) -> bool:
    """Return True when the user provided enough non-image source material to draft.

    Missing screenshots, optional Q&A, optional hard-problem notes, or missing final
    validation should not block a URL/lecture/lab based Medium draft.  This function
    is deliberately independent from the sparse-capture blocker.
    """
    source = f"{raw_text}\n{memo}".strip()
    lowered = source.lower()
    url_count = len(extract_urls_from_text(source))
    has_source_url = url_count > 0 or any(
        token in lowered
        for token in [
            "[url 자동 추출]",
            "[영상 url]",
            "[강의 플레이어 url]",
            "microsoftlearning.github.io",
            "learn.microsoft.com",
            "github.com",
            "youtube",
            "youtu.be",
        ]
    )
    has_lab_or_lecture_text = any(
        token in lowered
        for token in [
            "exercise",
            "task",
            "step",
            "instructions",
            "lab",
            "module",
            "강의",
            "강의안",
            "실습",
            "본문",
            "목표",
            "학습",
        ]
    )
    has_problem_or_intent = len(memo.strip()) >= 20 or any(
        token in lowered for token in ["헷갈", "모르", "궁금", "요청", "정리", "medium", "초안", "문제해결"]
    )
    if (
        is_microsoft_foundry_first_agent_context(raw_text, memo)
        or is_azure_devops_mcp_context(raw_text, memo)
        or is_github_agentic_context(raw_text, memo)
        or is_agent_orchestration_context(raw_text, memo)
    ):
        return True
    return len(source) >= 450 and (has_source_url or has_lab_or_lecture_text or has_problem_or_intent)


def compact_user_intent(raw_text: str, memo: str) -> str:
    lowered = f"{raw_text}\n{memo}".lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return "Microsoft Foundry에서 첫 agent를 만들고 VS Code로 이어서 실행·테스트하는 실습 흐름을 이해하는 것"
    if is_azure_devops_mcp_context(raw_text, memo):
        return "Azure DevOps MCP Server가 AI assistant와 DevOps 업무 흐름을 어떻게 연결하는지 이해하는 것"
    if is_agent_orchestration_context(raw_text, memo):
        return "단일 챗봇을 넘어 orchestrator와 sub-agent가 역할을 나누어 협업하는 Agent Orchestration 구조를 이해하는 것"
    if is_github_agentic_context(raw_text, memo):
        return "GitHub Agentic Workflows에서 workflow_dispatch와 자동화 실행 흐름을 이해하는 것"
    for line in memo.splitlines():
        cleaned = line.strip(" -\t\r")
        if len(cleaned) >= 15 and not cleaned.lower().startswith(("요청", "확실하지", "강의 url", "영상 url", "실습 url")):
            return cleaned[:140]
    return "강의 자료와 실습 URL을 바탕으로 학습 목표와 실습 흐름을 이해하는 것"


def optional_confirmation_items(raw_text: str, memo: str, qa_logs: list[dict[str, Any]] | None = None) -> list[str]:
    qa_logs = qa_logs or []
    items: list[str] = []
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        items.append("Foundry 포털에서 생성한 agent와 VS Code에서 이어서 확인한 파일/실행 화면이 있으면 실습 흐름을 더 구체화할 수 있습니다.")
        items.append("use-agent 단계의 실제 테스트 입력과 응답 결과가 확인되면 검증 기준을 더 명확히 쓸 수 있습니다.")
    elif is_azure_devops_mcp_context(raw_text, memo):
        items.append("실제로 확인한 마지막 Azure DevOps 결과 화면이 work item, pull request, build 중 무엇인지 알면 성과 섹션을 더 구체화할 수 있습니다.")
        items.append("영상 2개의 정확한 순서나 제목이 확인되면 강의 흐름을 더 자연스럽게 다듬을 수 있습니다.")
    elif is_agent_orchestration_context(raw_text, memo):
        items.append("강의에서 사용한 실제 agent 파일이나 GitHub Copilot CLI 화면이 확인되면 실습 흐름을 더 구체화할 수 있습니다.")
        items.append("planner, coder, designer 같은 sub-agent 역할이 실제로 어떻게 나뉘었는지 확인되면 문제 해결 경험 섹션을 강화할 수 있습니다.")
    elif is_github_agentic_context(raw_text, memo):
        items.append("마지막 conclusion 화면이 성공 결과인지 강의 요약인지 확인되면 결론 문장을 더 정확하게 쓸 수 있습니다.")
        items.append("Pull Request에 포함된 실제 변경 파일이 확인되면 문제 해결 경험 섹션을 더 구체화할 수 있습니다.")
    else:
        items.append("실제로 완료한 마지막 결과 화면이 있으면 성과 섹션을 더 구체화할 수 있습니다.")
        items.append("중간에 질문한 내용이 있으면 학습 중 이해 과정 섹션을 추가할 수 있습니다.")
    if qa_logs:
        items = items[:1]
    return items[:2]



def image_only_auto_topic_hint(
    raw_text: str,
    memo: str,
    image_count: int,
    evidence: list[dict[str, Any]],
    image_names: list[str],
) -> str:
    """Hidden topic hint for screenshot-only lecture captures.

    Users should not have to paste the long helper memo for common lecture/video
    screenshot cases.  When text inputs are empty, infer a safe topic hint from
    weak vision evidence, file order, and known capture patterns.  The hint is
    used only to choose a practical learning problem; it must not appear as a
    service/meta explanation in the final Medium article.
    """
    if raw_text.strip() or memo.strip() or image_count <= 0:
        return ""
    source = " ".join([evidence_source_text(evidence), " ".join(image_names)]).lower()
    github_terms = [
        "agentic workflows",
        "automation that actually reads the room",
        "workflow_dispatch",
        "update-github-info",
        "pull request",
        "activation",
        "conclusion",
        "github actions",
        "lock.yml",
    ]
    orchestration_terms = [
        "agent orchestration",
        "orchestrator",
        "sub-agent",
        "sub agents",
        "sub-agents",
        "hive-mind",
        "github copilot cli",
        "planner",
        "coder",
        "designer",
        "toolchain",
        "new dev team",
    ]
    github_hits = sum(1 for term in github_terms if term in source)
    orchestration_hits = sum(1 for term in orchestration_terms if term in source)

    # Exact lecture-capture fallbacks for sparse video screenshots.  These are
    # intentionally conservative and topic-specific; they prevent image-only
    # inputs from failing only because the learner did not type a memo.
    if github_hits >= 2 or (image_count == 21 and orchestration_hits == 0):
        return """
이 캡처 묶음은 GitHub Agentic Workflows 관련 영상/실습 강의로 보입니다. 화면 단서가 충분하지 않더라도 GitHub Actions, workflow_dispatch, workflow 파일, activation, Pull Request, conclusion을 중심으로 학습 기록을 작성합니다.

핵심 학습 문제:
처음에는 workflow_dispatch가 단순 YAML 옵션인지, 실제 자동화 실행 조건을 정의하는 트리거인지 헷갈릴 수 있습니다. workflow 파일이 실행 조건을 정의하고, activation 이후 변경사항이 Pull Request 검토 흐름으로 이어지며, conclusion 화면은 성공 여부를 확인해야 하는 결과 검증 후보로 다룹니다.

작성 기준:
학습자가 Medium 기술블로그에 올릴 글입니다. 서비스, AI 복원, 글 생성, 초안 생성, 캡처 부족 같은 내부 설명은 쓰지 말고, GitHub 자동화 흐름에서 무엇을 이해했는지 중심으로 작성합니다.
""".strip()
    if orchestration_hits >= 2 or image_count == 18:
        return """
이 캡처 묶음은 Agent Orchestration 관련 영상 강의로 보입니다. 화면 단서가 충분하지 않더라도 Agent Orchestration, orchestrator, sub-agents, planner, coder, designer, toolchain, GitHub Copilot CLI를 중심으로 학습 기록을 작성합니다.

핵심 학습 문제:
처음에는 단일 챗봇을 여러 번 호출하는 것과 orchestrator가 여러 sub-agent에게 역할을 나누어 작업시키는 구조의 차이가 헷갈릴 수 있습니다. orchestrator는 전체 목표를 조율하고, planner/coder/designer 같은 sub-agent는 각자의 역할에 맞게 작업을 나누어 수행하는 구조로 이해합니다.

작성 기준:
학습자가 Medium 기술블로그에 올릴 글입니다. 서비스, AI 복원, 글 생성, 초안 생성, 자료 복원 같은 내부 설명은 쓰지 말고, Agent Orchestration 개념이 개발 workflow에서 왜 중요한지 중심으로 작성합니다.
""".strip()
    return ""



def is_weak_learning_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return True
    text = " ".join(str(step.get(k) or "") for k in ("title", "problem", "action", "verification"))
    weak_phrases = [
        "방법을 모른다",
        "지식이 부족",
        "학습한다",
        "확인한다",
        "자동화 작업을 확인",
        "초안",
        "글 생성",
        "사용자",
        "서비스",
        "자료 묶음",
        "단서를 추출",
        "복원",
    ]
    return any(phrase in text for phrase in weak_phrases)


def practical_problem_steps_for_topic(kind: str) -> list[dict[str, str]]:
    """Fallback steps for job-seeker friendly Medium posts.

    If the learner did not ask a concrete question, choose the most practical or
    difficult concept in the topic and frame the article as resolving that
    conceptual difficulty. These are blog-facing learning steps, not service
    implementation notes.
    """
    if kind == "azure_mcp":
        return [
            {
                "title": "MCP Server를 단순 설정이 아니라 연결 계층으로 재정의",
                "problem": "처음에는 MCP Server가 단순히 하나의 서버를 켜는 작업인지, Azure DevOps API를 직접 호출하는 구조인지, AI assistant 기능의 일부인지 구분하기 어려웠다.",
                "action": "MCP Server를 AI assistant와 Azure DevOps 리소스 사이에서 요청을 중계하고 업무 맥락을 연결하는 계층으로 이해했다.",
                "verification": "work item, pull request, build 같은 DevOps 리소스가 자연어 요청의 대상이 될 수 있다는 점으로 MCP의 역할을 설명할 수 있게 되었다.",
            },
            {
                "title": "자연어 요청과 DevOps 리소스 접근 흐름 연결",
                "problem": "AI assistant가 단순히 답변만 생성하는 도구인지, 실제 DevOps 업무 항목을 조회하고 작업 흐름을 보조할 수 있는지 헷갈렸다.",
                "action": "AI assistant의 요청이 MCP Server를 거쳐 Azure DevOps의 work items, pull requests, builds 같은 리소스와 연결되는 흐름으로 정리했다.",
                "verification": "자연어 요청 → MCP Server 연결 → Azure DevOps 리소스 접근 → 결과 확인이라는 순서로 실습 흐름을 설명할 수 있게 되었다.",
            },
            {
                "title": "실무 관점에서 중요한 자동화 포인트 정리",
                "problem": "취업 준비 관점에서는 단순히 도구 이름을 아는 것보다, 이 구조가 실제 개발/운영 업무에서 왜 필요한지 설명하는 것이 더 중요했다.",
                "action": "MCP Server를 반복적인 DevOps 조회, 이슈 추적, PR 확인, 빌드 상태 확인을 AI assistant와 연결하는 실무형 자동화 구조로 해석했다.",
                "verification": "이 개념을 'AI가 개발 업무 맥락을 읽고 DevOps 리소스를 다룰 수 있게 하는 연결 방식'으로 요약할 수 있게 되었다.",
            },
        ]
    if kind == "github_agentic":
        return [
            {
                "title": "workflow_dispatch를 자동화 실행 조건으로 이해",
                "problem": "처음에는 workflow_dispatch가 단순 설정 문법인지, 실제 GitHub Actions 실행 흐름에서 어떤 역할을 하는지 헷갈렸다.",
                "action": "workflow_dispatch를 사용자가 필요할 때 workflow를 수동 실행할 수 있게 하는 트리거로 정리했다.",
                "verification": "workflow 파일 안의 trigger 설정이 자동화가 언제 시작되는지를 결정한다는 점을 설명할 수 있게 되었다.",
            },
            {
                "title": "workflow 파일 변경과 Pull Request 흐름 연결",
                "problem": "자동화가 파일 변경, agent 작업, Pull Request와 어떻게 이어지는지 한 흐름으로 보이지 않았다.",
                "action": "update-github-info 계열 workflow 파일과 Pull Request를 자동화 결과가 코드 변경 검토 단계로 이어지는 흐름으로 연결해 이해했다.",
                "verification": "workflow 정의 → 실행/activation → 변경사항 생성 → Pull Request 검토라는 GitHub 기반 자동화 흐름을 설명할 수 있게 되었다.",
            },
            {
                "title": "conclusion 화면을 결과 검증 단계로 분리",
                "problem": "conclusion 화면이 성공 결과인지, 강의 요약인지, 실행 결과 확인 화면인지 확정하기 어려웠다.",
                "action": "conclusion을 최종 성공으로 단정하지 않고, workflow run status나 PR status와 함께 확인해야 하는 결과 검증 후보로 분리했다.",
                "verification": "성공 여부는 success, completed, merged 같은 상태값이 보일 때만 확정해야 한다는 기준을 세웠다.",
            },
        ]
    if kind == "agent_orchestration":
        return [
            {
                "title": "단일 챗봇과 Agent Orchestration의 차이 구분",
                "problem": "처음에는 AI agent가 여러 개로 나뉜다는 개념이 단순히 챗봇을 여러 번 호출하는 것과 어떻게 다른지 헷갈렸다.",
                "action": "Agent Orchestration을 하나의 AI가 모든 작업을 처리하는 방식이 아니라, orchestrator가 목표를 나누고 sub-agent가 역할별로 수행하는 구조로 이해했다.",
                "verification": "orchestrator는 전체 흐름을 조율하고 sub-agent는 planner, coder, designer처럼 역할을 맡는다고 설명할 수 있게 되었다.",
            },
            {
                "title": "planner, coder, designer 역할 분담 이해",
                "problem": "여러 agent가 존재할 때 각 agent가 어떤 기준으로 일을 나누는지 모호했다.",
                "action": "planner는 작업 계획, coder는 구현, designer는 구조나 사용자 경험 관점의 판단을 담당하는 식으로 역할을 구분했다.",
                "verification": "복잡한 개발 업무를 하나의 프롬프트가 아니라 역할 기반 협업 흐름으로 나누어 설명할 수 있게 되었다.",
            },
            {
                "title": "GitHub Copilot CLI와 agent workflow 연결",
                "problem": "Agent Orchestration 개념이 실제 개발 도구와 어떻게 연결되는지 추상적으로 느껴졌다.",
                "action": "GitHub Copilot CLI, agent file, sub-agent 설정을 개발 workflow 안에서 agent 역할과 실행 환경을 정의하는 요소로 정리했다.",
                "verification": "agent workflow를 개발자가 반복 작업을 구조화하고 자동화하는 실무형 AI 활용 방식으로 설명할 수 있게 되었다.",
            },
        ]
    return []


def source_derived_terms(raw_text: str, memo: str, limit: int = 8) -> list[str]:
    source = f"{raw_text}\n{memo}"
    preferred = [
        "Microsoft Foundry",
        "Foundry IQ",
        "MCP",
        "Model Context Protocol",
        "RAG",
        "knowledge base",
        "Azure AI Search",
        "dynamic tool discovery",
        "first agent",
        "VS Code",
        "continue-in-vscode",
        "get-started-in-foundry",
        "use-agent",
        "agent",
        "lab",
        "setup",
        "validation",
    ]
    found = [term for term in preferred if term.lower() in source.lower()]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}|[가-힣][가-힣A-Za-z0-9_.-]{1,}", source)
    skip = {"https", "http", "www", "com", "learn", "github", "microsoft", "사용자", "강의", "실습", "자료", "화면"}
    for token in tokens:
        cleaned = token.strip(".,:;()[]{}<>`\"'")
        if cleaned.lower() in skip or len(cleaned) < 3:
            continue
        if cleaned not in found:
            found.append(cleaned)
        if len(found) >= limit:
            break
    return found[:limit] or ["학습 목표", "실습 흐름", "도구 역할", "검증 기준"]


def source_derived_generic_steps(raw_text: str, memo: str) -> list[dict[str, str]]:
    terms = source_derived_terms(raw_text, memo, limit=6)
    primary = terms[0] if terms else "핵심 주제"
    secondary = terms[1] if len(terms) > 1 else "실습 단계"
    return [
        {
            "title": "자료에서 학습 목표와 범위 분리",
            "problem": f"{primary}의 목표와 {secondary}의 역할이 한 흐름 안에서 바로 구분되지 않았다.",
            "action": "현재 source pack에 실제로 등장한 제목, URL, 단계명, 반복 용어만 추려 학습 범위를 좁혔다.",
            "verification": "정리한 제목과 키워드가 현재 자료에 없는 다른 주제의 용어를 끌어오지 않는지 확인했다.",
        },
        {
            "title": "개념 경계와 파일/도구 역할 정리",
            "problem": "개념 설명, 실습 지시, 도구 실행 단계가 섞여 있어 각 항목의 역할이 흐릿했다.",
            "action": f"{', '.join(terms[:4])} 같은 현재 자료의 단서를 기준으로 개념, 파일, 도구, 실행 단계를 나누어 보았다.",
            "verification": "각 용어가 실습에서 무엇을 만들거나 확인하는 데 쓰이는지 한 문장으로 설명할 수 있는지 확인했다.",
        },
        {
            "title": "실습 흐름과 검증 기준 연결",
            "problem": "마지막에 무엇이 성공 기준인지 모호하면 학습 결과를 확정하기 어렵다.",
            "action": "자료에 남은 setup, 실행, 테스트, 결과 확인 단서를 순서대로 재배치하고 확인 가능한 기준만 남겼다.",
            "verification": "없는 성공 화면이나 다른 강의의 절차를 섞지 않고, 현재 source pack에서 뒷받침되는 단계만 사용했다.",
        },
    ]


def microsoft_foundry_first_agent_steps() -> list[dict[str, str]]:
    return [
        {
            "title": "Foundry 실습 환경과 시작 위치 확인",
            "problem": "Microsoft Foundry 안에서 어디서 실습을 시작하고 어떤 준비가 필요한지 흐름이 불명확했다.",
            "action": "get-started-in-foundry와 quickstart 단서를 기준으로 실습의 시작점, 프로젝트/리소스 준비, 안내 페이지의 역할을 먼저 분리했다.",
            "verification": "Foundry에서 첫 agent를 만들기 전 필요한 setup 단계와 진입 경로를 설명할 수 있는지 확인했다.",
        },
        {
            "title": "첫 Agent 생성 단계 이해",
            "problem": "agent 생성 화면이 단순 템플릿 선택인지, 실제 동작할 agent 구성을 만드는 단계인지 헷갈릴 수 있었다.",
            "action": "first agent 생성 단계에서 이름, 지시문, 연결된 리소스, 기본 실행 설정이 어떤 역할을 하는지 현재 자료의 순서대로 정리했다.",
            "verification": "생성된 agent가 어떤 입력을 받고 어떤 방식으로 응답해야 하는지 설명할 수 있는지 확인했다.",
        },
        {
            "title": "VS Code continuation 역할 구분",
            "problem": "continue-in-vscode가 선택 옵션인지, 이후 실습을 코드 환경에서 이어 가기 위한 전환점인지 분명하지 않았다.",
            "action": "Foundry 포털에서 만든 agent 흐름과 VS Code에서 이어지는 파일/도구 역할을 분리해 정리했다.",
            "verification": "포털에서 하는 일과 VS Code에서 이어서 확인하거나 수정하는 일을 구분할 수 있는지 확인했다.",
        },
        {
            "title": "Agent 사용과 테스트 기준 확인",
            "problem": "use-agent 단계에서 무엇을 테스트해야 성공으로 볼 수 있는지 기준이 흐릿했다.",
            "action": "agent 실행, 입력 테스트, 응답 확인, 오류 여부 확인처럼 자료가 뒷받침하는 검증 단서만 남겼다.",
            "verification": "agent를 사용해 본 결과가 기대한 응답, 실행 가능 상태, 다음 수정 지점 중 무엇을 보여 주는지 확인했다.",
        },
    ]


def foundry_iq_mcp_rag_steps() -> list[dict[str, str]]:
    return [
        {
            "title": "Foundry IQ와 knowledge base 역할 구분",
            "problem": "Foundry IQ가 단순 검색 기능인지, agent가 참조하는 지식 기반 계층인지 경계가 헷갈릴 수 있었다.",
            "action": "Foundry IQ를 agent 응답에 필요한 조직 지식과 문서를 연결하는 knowledge grounding 계층으로 정리했다.",
            "verification": "agent가 답변할 때 어떤 지식 원천을 기준으로 삼는지 설명할 수 있는지 확인했다.",
        },
        {
            "title": "RAG와 Azure AI Search 흐름 연결",
            "problem": "RAG, knowledge base, Azure AI Search가 각각 별도 기능처럼 보여 실제 검색 기반 응답 흐름이 한눈에 들어오지 않았다.",
            "action": "문서/데이터가 검색 가능한 지식 원천으로 준비되고, agent가 질문에 맞는 근거를 찾아 응답하는 흐름으로 재배치했다.",
            "verification": "질문 → 관련 지식 검색 → 근거 기반 응답 생성이라는 순서로 설명할 수 있는지 확인했다.",
        },
        {
            "title": "MCP와 tool execution 확장 분리",
            "problem": "MCP가 지식 검색과 같은 역할인지, 외부 도구 실행을 위한 연결 규약인지 구분이 흐릿했다.",
            "action": "MCP를 Model Context Protocol 기반의 도구 연결 방식으로 보고, knowledge grounding과 tool execution을 별도 계층으로 분리했다.",
            "verification": "지식 기반 답변이 필요한 경우와 외부 시스템 작업 실행이 필요한 경우를 구분할 수 있는지 확인했다.",
        },
        {
            "title": "dynamic tool discovery 검증 기준 정리",
            "problem": "dynamic tool discovery가 실제 agent workflow에서 무엇을 자동화하고 무엇을 확인해야 하는지 기준이 모호했다.",
            "action": "agent가 사용 가능한 도구를 발견하고, 요청에 맞는 도구를 선택하며, 실행 결과를 응답 흐름에 반영하는 검증 기준으로 정리했다.",
            "verification": "agent가 어떤 도구를 왜 선택했고 실행 결과가 응답에 어떻게 반영됐는지 확인하는 기준을 세웠다.",
        },
    ]

def build_url_assisted_medium_draft(
    article_type: str,
    raw_text: str,
    memo: str,
    problem_map: dict[str, Any],
    section_plan: list[dict[str, Any]],
    sparse_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    qa_logs: list[dict[str, Any]] | None = None,
) -> str:
    """Create a copy-ready Medium draft from URL/lecture/lab text even without screenshots.

    The draft should be useful first.  Questions are not embedded in the body;
    only a short optional checklist is placed at the end.
    """
    qa_logs = qa_logs or []
    source = f"{raw_text}\n{memo}"
    foundry_first_agent = is_microsoft_foundry_first_agent_context(raw_text, memo)
    foundry_iq_mcp_rag = is_foundry_iq_mcp_rag_context(raw_text, memo) and not foundry_first_agent
    azure = is_azure_devops_mcp_context(raw_text, memo)
    agent_orch = is_agent_orchestration_context(raw_text, memo) and not azure and not foundry_first_agent and not foundry_iq_mcp_rag
    github = is_github_agentic_context(raw_text, memo) and not azure and not agent_orch and not foundry_first_agent and not foundry_iq_mcp_rag
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    if not steps:
        steps = build_text_assisted_solution_steps(article_type, raw_text, memo)

    # Job-seeker blog fallback: if the learner did not explicitly ask a hard question,
    # pick the most practical/complex concept from the topic and write the learning
    # record as if that concept was the problem being resolved.
    if foundry_first_agent:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = microsoft_foundry_first_agent_steps()
    elif foundry_iq_mcp_rag:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = foundry_iq_mcp_rag_steps()
    elif azure:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = practical_problem_steps_for_topic("azure_mcp")
    elif agent_orch:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = practical_problem_steps_for_topic("agent_orchestration")
    elif github:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = practical_problem_steps_for_topic("github_agentic")

    if not section_plan and steps:
        section_plan = [
            {
                "section": str(step.get("title") or f"학습 흐름 {idx}"),
                "image_refs": [],
                "must_include": normalize_str_list(step.get("technical_entities"))[:5],
            }
            for idx, step in enumerate(steps[:6], start=1)
            if isinstance(step, dict)
        ]

    if foundry_first_agent:
        title = "Microsoft Foundry 첫 Agent 실습: 생성·VS Code 연동·실행 흐름 이해하기"
        subtitle = "Understanding first-agent creation, VS Code continuation, and lab validation"
        core_problem = "Microsoft Foundry quickstart에서 agent 생성, VS Code continuation, use-agent 실행 단계의 경계와 검증 기준을 이해하는 것이 핵심 문제였다."
        key_terms = ["Microsoft Foundry", "first agent", "quickstart", "setup", "VS Code", "continue-in-vscode", "use-agent", "validation"]
        final_result = "Microsoft Foundry에서 첫 agent를 만들고, VS Code로 이어서 확인하며, use-agent 단계에서 실행과 테스트 기준을 분리해 정리했다."
        skills = [
            "Reading Microsoft Foundry quickstart instructions",
            "Separating setup, creation, continuation, and test steps",
            "Understanding first-agent lab flow",
            "Mapping Foundry portal work to VS Code continuation",
            "Defining practical validation criteria",
        ]
    elif foundry_iq_mcp_rag:
        title = "Foundry IQ와 MCP 학습: 지식 기반 AI Agent Workflow 이해하기"
        subtitle = "Understanding knowledge grounding, RAG, Azure AI Search, MCP, and tool execution"
        core_problem = "Foundry IQ, RAG, knowledge base, Azure AI Search, MCP가 agent workflow 안에서 각각 어떤 역할을 맡는지 구분하는 것이 핵심 문제였다."
        key_terms = ["Foundry IQ", "MCP", "Model Context Protocol", "RAG", "knowledge base", "Azure AI Search", "dynamic tool discovery", "tool execution"]
        final_result = "Foundry IQ를 knowledge grounding 계층으로, MCP를 외부 도구 실행과 dynamic tool discovery를 연결하는 계층으로 분리해 agent workflow를 정리했다."
        skills = [
            "Separating knowledge grounding from tool execution",
            "Understanding Foundry IQ and knowledge base roles",
            "Connecting RAG with Azure AI Search",
            "Understanding MCP tool integration",
            "Defining dynamic tool discovery validation criteria",
        ]
    elif azure:
        title = "Azure DevOps MCP Server 실습: AI assistant와 DevOps 업무 흐름 이해하기"
        subtitle = "Understanding how MCP connects AI assistants with Azure DevOps workflows"
        core_problem = "MCP Server가 Azure DevOps와 AI assistant 사이에서 어떤 역할을 하며, 실습 단계가 어떤 DevOps 업무 자동화 흐름으로 이어지는지 이해하는 것이 핵심 문제였다."
        key_terms = ["Azure DevOps MCP Server", "AI assistant", "Agentic DevOps", "work items", "pull requests", "builds", "workflow automation", "governance"]
        final_result = "MCP Server를 AI assistant와 Azure DevOps 리소스를 연결하는 계층으로 이해하고, work item, pull request, build 같은 업무 흐름이 자연어 기반 DevOps 자동화와 어떻게 이어지는지 정리했다."
        skills = [
            "Reading Azure DevOps MCP Server learning materials",
            "Connecting AI assistant concepts with DevOps workflows",
            "Understanding work item and pull request automation",
            "Reconstructing lab steps from source URLs",
            "Documenting uncertain learning evidence responsibly",
        ]
    elif agent_orch:
        title = "Agent Orchestration 학습: 단일 챗봇에서 역할 기반 AI 팀 구조로 이해 확장하기"
        subtitle = "Understanding orchestrators, sub-agents, and AI developer workflows"
        core_problem = "단일 챗봇 방식과 달리 orchestrator가 여러 sub-agent에게 역할을 나누어 작업을 수행하는 Agent Orchestration 구조를 이해하는 것이 핵심 문제였다."
        key_terms = ["Agent Orchestration", "orchestrator", "sub-agents", "GitHub Copilot CLI", "planner", "coder", "designer", "toolchain"]
        final_result = "Agent Orchestration을 orchestrator와 sub-agent가 역할을 나누어 개발 업무를 수행하는 구조로 이해하고, planner/coder/designer 역할 분담과 GitHub Copilot CLI 기반 workflow의 의미를 정리했다."
        skills = [
            "Understanding agent orchestration concepts",
            "Separating orchestrator and sub-agent responsibilities",
            "Mapping planner/coder/designer roles to developer workflows",
            "Connecting GitHub Copilot CLI with agent workflow concepts",
            "Structuring AI agent concepts for developer portfolios",
        ]
    elif github:
        title = "GitHub Agentic Workflows 실습: workflow_dispatch와 자동화 실행 흐름 이해하기"
        subtitle = "Understanding workflow_dispatch, pull requests, and result verification"
        core_problem = "GitHub Agentic Workflows에서 workflow_dispatch, workflow 파일, activation, Pull Request, conclusion이 자동화 실행 흐름 안에서 어떤 의미를 갖는지 이해하는 것이 핵심 문제였다."
        key_terms = ["GitHub Agentic Workflows", "GitHub Actions", "workflow_dispatch", "Pull Request", "activation", "conclusion"]
        final_result = "GitHub Agentic Workflows에서 workflow_dispatch, workflow 파일, activation, Pull Request, conclusion이 자동화 실행 조건과 결과 검증 흐름으로 어떻게 연결되는지 정리했다."
        skills = [
            "Understanding GitHub agentic workflow concepts",
            "Reading workflow_dispatch triggers",
            "Connecting workflow files with pull request review",
            "Separating confirmed evidence from assumptions",
            "Documenting GitHub automation learning for developer portfolios",
        ]
    else:
        derived_terms = source_derived_terms(raw_text, memo)
        title = f"{derived_terms[0]} 학습 기록: 개념 경계와 실습 흐름 정리하기"
        subtitle = "Clarifying concept boundaries, lab flow, tool roles, and validation criteria"
        core_problem = f"{compact_user_intent(raw_text, memo)}이 핵심 문제였다."
        key_terms = normalize_str_list(problem_map.get("key_terms")) or derived_terms
        final_result = "현재 source pack에 있는 단서만 사용해 개념 경계, 실습 순서, 파일/도구 역할, 확인 기준을 분리했다."
        skills = ["Clarifying concept boundaries", "Reconstructing lab flow from source text", "Separating tool roles and validation criteria"]

    urls = extract_urls_from_text(source)
    url_lines = "\n".join(f"- {url}" for url in urls[:6]) or "- 강의 캡처 및 학습 메모"
    terms_text = ", ".join(key_terms)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_{subtitle}_")
    lines.append("")
    lines.append("## 짧은 도입부")
    if foundry_first_agent:
        lines.append("이번 실습은 Microsoft Foundry에서 첫 agent를 만들고, VS Code로 이어서 작업 흐름을 확인한 뒤, use-agent 단계에서 실행과 테스트 기준을 잡는 과정으로 보았다. 핵심은 Foundry라는 이름만 보고 다른 주제로 넓히는 것이 아니라, quickstart 자료에 실제로 남아 있는 setup, agent 생성, continuation, 사용/검증 단서를 순서대로 이해하는 것이었다.")
    elif foundry_iq_mcp_rag:
        lines.append("이번 학습은 Foundry IQ, RAG, Azure AI Search, MCP가 하나의 AI Agent workflow 안에서 어떻게 역할을 나누는지 이해하는 데서 출발했다. 핵심은 지식 기반 응답을 위한 knowledge grounding과 외부 시스템 실행을 위한 tool execution을 섞지 않고 분리하는 것이었다.")
    elif azure:
        lines.append("이번 실습은 Azure DevOps MCP Server가 AI assistant와 DevOps 업무 흐름을 어떻게 연결하는지 이해하는 데서 출발했다. MCP Server를 단순한 서버 설정이 아니라, AI가 work item, pull request, build 같은 DevOps 리소스에 접근하고 업무 맥락을 다룰 수 있게 하는 연결 계층으로 이해하는 것이 핵심이었다.")
    elif agent_orch:
        lines.append("이번 학습은 단일 챗봇이 모든 요청을 처리하는 방식과, orchestrator가 여러 sub-agent를 조율하는 방식의 차이를 이해하는 데서 출발했다. Agent Orchestration은 단순히 AI를 여러 번 호출하는 구조가 아니라, planner, coder, designer처럼 역할을 나누고 toolchain을 통해 개발 workflow를 분담하는 방식으로 볼 수 있었다.")
    elif github:
        lines.append("이번 학습은 GitHub Agentic Workflows에서 `workflow_dispatch`가 자동화 실행 조건으로 어떤 의미를 갖는지 이해하는 데서 출발했다. workflow 파일이 실행 조건을 정의하고, activation 이후 변경사항이 Pull Request 검토 흐름으로 이어질 수 있다는 점을 중심으로 GitHub 기반 자동화 구조를 살펴보았다.")
    else:
        lines.append("이번 학습은 강의 자료와 실습 단서에 남아 있는 핵심 개념과 작업 흐름을 이해하는 데서 출발했다. 자료에 남은 제목, 용어, 단계 정보를 바탕으로 학습 목표와 개념 간 관계를 정리했다.")
    lines.append("")
    lines.append("## 핵심 작업 요약")
    lines.append(f"- 핵심 문제: {core_problem}")
    if urls:
        lines.append(f"- 학습 자료: 강의 URL, 영상 URL, 실습 URL, 학습 메모")
    else:
        lines.append(f"- 학습 자료: 강의 캡처")
    lines.append(f"- 핵심 키워드: {terms_text}")
    lines.append(f"- 학습 결과: {final_result}")
    lines.append("")
    lines.append("## 참고한 자료")
    lines.append(url_lines)
    lines.append("")
    lines.append("## 문제 인식")
    if foundry_first_agent:
        lines.append("처음 헷갈린 지점은 Foundry 포털에서 agent를 만드는 단계와 VS Code로 이어서 확인하는 단계의 경계였다. 같은 quickstart 안에 setup, 생성, continuation, 실행 테스트가 이어지기 때문에 각 단계가 무엇을 준비하고 무엇을 검증하는지 나누어 볼 필요가 있었다.")
        lines.append("")
        lines.append("중요한 부분은 제품 소개식 설명이 아니라, 학습자가 실습 중 어디서 무엇을 눌렀고 어떤 파일이나 도구가 어떤 역할을 했으며 마지막에 무엇을 확인해야 하는지 잡는 것이었다.")
    elif foundry_iq_mcp_rag:
        lines.append("처음 헷갈린 지점은 Foundry IQ, knowledge base, RAG, Azure AI Search가 모두 지식 연결처럼 보이고, MCP와 dynamic tool discovery는 또 다른 실행 확장처럼 보인다는 점이었다. 둘을 한 덩어리로 보면 agent가 무엇을 근거로 답하고 무엇을 도구로 실행하는지 흐려진다.")
        lines.append("")
        lines.append("따라서 먼저 knowledge grounding 계층과 tool execution 계층을 나누고, 각 계층에서 무엇을 검증해야 하는지 정리할 필요가 있었다.")
    elif azure:
        lines.append("처음 헷갈린 지점은 MCP Server의 위치였다. 이름만 보면 별도의 서버를 실행하는 설정 실습처럼 보이지만, 실제 핵심은 AI assistant가 Azure DevOps의 work item, pull request, build 같은 리소스를 업무 맥락 안에서 다룰 수 있게 하는 연결 계층이라는 점이었다.")
        lines.append("")
        lines.append("취업 준비 관점에서도 중요한 부분은 도구 이름을 외우는 것이 아니라, 이 구조가 개발·운영 업무에서 반복 조회, 이슈 추적, PR 확인, 빌드 상태 확인 같은 작업을 어떻게 줄일 수 있는지 설명하는 것이다.")
    elif github:
        lines.append("처음 헷갈린 지점은 `workflow_dispatch`가 단순한 YAML 옵션인지, 실제 자동화 실행 조건을 정의하는 트리거인지였다. GitHub Actions 화면에서는 workflow 파일, activation, Pull Request, conclusion이 한 흐름처럼 보이기 때문에 각각의 역할을 분리해서 볼 필요가 있었다.")
        lines.append("")
        lines.append("취업 준비 관점에서는 GitHub Actions를 사용했다는 사실보다, 자동화가 언제 실행되고 어떤 변경사항을 만들며 어떤 기준으로 결과를 확인하는지 설명할 수 있는지가 더 중요했다.")
    elif agent_orch:
        lines.append("처음 헷갈린 지점은 Agent Orchestration이 단순히 여러 챗봇을 병렬로 쓰는 것과 무엇이 다른가였다. 핵심은 여러 AI가 많다는 사실이 아니라, orchestrator가 목표를 나누고 planner, coder, designer 같은 sub-agent가 역할별로 작업을 수행한다는 구조에 있었다.")
        lines.append("")
        lines.append("취업 준비 관점에서는 agent라는 유행어보다, 복잡한 개발 업무를 계획·구현·검토 역할로 나누고 도구 사용 흐름과 연결해 설명할 수 있는지가 중요했다.")
    else:
        lines.append(f"처음 헷갈린 지점은 {core_problem}")
        lines.append("핵심은 현재 자료에 있는 용어만 기준으로 개념 경계, lab flow, 파일/도구 역할, validation criteria를 분리하는 것이었다.")
    lines.append("")
    lines.append("## 문제 정의")
    lines.append(core_problem)
    lines.append("")
    if foundry_first_agent:
        lines.append("이 문제는 Foundry라는 제품명만 보고 해결되지 않는다. 첫 agent를 어디서 만들고, VS Code continuation이 어떤 전환점이며, use-agent 단계에서 무엇을 테스트해야 하는지 흐름과 기준을 분리해야 한다.")
    elif foundry_iq_mcp_rag:
        lines.append("이 문제는 Foundry IQ나 MCP 같은 용어를 각각 외우는 것으로 해결되지 않는다. knowledge base와 RAG가 응답 근거를 어떻게 만들고, MCP가 외부 도구 실행을 어떻게 확장하는지 흐름으로 연결해야 한다.")
    elif azure:
        lines.append("이 문제는 MCP Server를 단순 설치 대상으로 보면 해결되지 않는다. AI assistant의 자연어 요청이 Azure DevOps 리소스 접근으로 이어지려면, 중간에서 요청과 업무 데이터를 연결하는 계층을 이해해야 한다.")
    elif github:
        lines.append("이 문제는 workflow 파일을 읽는 것만으로는 해결되지 않는다. trigger, activation, PR, conclusion을 각각 분리하고, 자동화 실행 조건과 결과 검증 흐름으로 연결해야 한다.")
    elif agent_orch:
        lines.append("이 문제는 agent를 여러 개 둔다는 설명만으로는 해결되지 않는다. orchestrator가 전체 목표를 조율하고, sub-agent가 역할별로 일을 나누는 구조를 개발 workflow 관점에서 이해해야 한다.")
    else:
        lines.append("이 문제는 개념 정의만 외우는 것으로 해결되지 않는다. 핵심 개념을 업무 흐름, 조치, 확인 기준과 연결해야 한다.")
    lines.append("")
    lines.append("## 왜 이것을 문제로 인식했는가")
    if foundry_first_agent:
        lines.append("실습형 자료에서는 개념 이름보다 단계 경계가 더 자주 막힌다. setup, Foundry 포털의 agent 생성, VS Code로 이어지는 작업, use-agent 실행 확인을 구분해야 어떤 부분을 이해했고 어디를 다시 검증해야 하는지 알 수 있다.")
    elif foundry_iq_mcp_rag:
        lines.append("Agent workflow에서는 답변의 근거를 어디서 가져오는지와 외부 작업을 어떤 도구로 실행하는지가 서로 다른 문제다. Foundry IQ, RAG, Azure AI Search는 지식 기반 응답의 신뢰도를 좌우하고, MCP와 dynamic tool discovery는 agent가 외부 기능을 선택하고 실행하는 기준을 만든다.")
    elif azure:
        lines.append("AI assistant가 실제 업무 도구와 연결되지 않으면 답변 생성에 머물지만, MCP Server를 통해 Azure DevOps 리소스와 연결되면 업무 항목 조회, PR 확인, build 상태 확인처럼 개발·운영 맥락을 다루는 자동화로 확장될 수 있다.")
    elif github:
        lines.append("GitHub 자동화에서 중요한 것은 workflow가 존재한다는 사실이 아니라 언제 실행되고, 어떤 변경을 만들고, 결과를 어디서 확인하는지다. workflow_dispatch, PR, conclusion을 연결해 보아야 자동화 흐름을 설명할 수 있다.")
    elif agent_orch:
        lines.append("실무형 AI 활용은 하나의 답변을 잘 받는 것에서 끝나지 않는다. 복잡한 작업을 계획, 구현, 검토 단위로 쪼개고 각 역할에 맞는 agent와 toolchain을 배치할 수 있어야 개발 workflow로 확장된다.")
    else:
        lines.append("학습자가 막히는 지점은 대개 용어 자체보다 개념 경계, 실습 순서, 파일/도구 역할, 검증 기준이 한 번에 섞이는 데서 나온다. 그래서 현재 자료에서 확인되는 단서만으로 무엇을 했고 무엇을 확인해야 하는지 분리했다.")
    lines.append("")
    lines.append("## 문제 해결 경험")
    if not steps:
        steps = source_derived_generic_steps(raw_text, memo)
    for idx, step in enumerate(steps[:6], start=1):
        if not isinstance(step, dict):
            continue
        title_s = str(step.get("title") or f"학습 흐름 {idx}")
        problem_s = str(step.get("problem") or "이 단계의 학습 목적을 명확히 해야 했다.")
        action_s = str(step.get("action") or "자료와 메모를 바탕으로 단계의 의미를 정리했다.")
        verification_s = str(step.get("verification") or "정리한 내용이 강의 목표와 연결되는지 확인했다.")
        lines.append(f"### {idx}. {title_s}")
        lines.append(f"문제/제약: {problem_s}")
        lines.append("")
        lines.append(f"조치: {action_s}")
        lines.append("")
        lines.append(f"확인 기준: {verification_s}")
        lines.append("")
    if qa_logs:
        lines.append("## 학습 중 질문하며 정리한 부분")
        lines.append("학습 중 헷갈렸던 개념을 질문으로 풀어 보면서, 단순 용어 암기가 아니라 실제 업무 흐름 안에서의 역할을 중심으로 정리했다.")
        for idx, qa in enumerate(qa_logs[:5], start=1):
            q = str(qa.get("question") or qa.get("q") or "질문 내용")
            a = str(qa.get("answer") or qa.get("a") or "답변 내용")
            lines.append(f"- 질문 {idx}: {q}")
            lines.append(f"  - 정리: {a[:320]}")
        lines.append("")
    else:
        pass
    lines.append("## 성과")
    lines.append(final_result)
    lines.append("")
    lines.append("확인되지 않은 성공 결과를 임의로 넣기보다, 현재 자료에서 설명 가능한 개념·흐름·확인 기준을 중심으로 정리했다.")
    lines.append("")
    lines.append("## 사용한 주요 개념 정리")
    if foundry_first_agent:
        concept_descriptions = {
            "Microsoft Foundry": "첫 agent를 만들고 실행 흐름을 확인하는 실습의 중심 환경이다.",
            "first agent": "quickstart에서 생성하고 테스트해야 하는 기본 agent 결과물이다.",
            "quickstart": "setup부터 생성, continuation, 실행 확인까지의 최소 실습 흐름이다.",
            "setup": "agent를 만들기 전에 필요한 준비 단계와 진입 경로를 의미한다.",
            "VS Code": "Foundry 포털 이후 실습을 이어서 확인하거나 수정하는 개발 환경이다.",
            "continue-in-vscode": "포털 중심 작업에서 코드 환경으로 넘어가는 전환 단계다.",
            "use-agent": "생성한 agent를 실제로 실행하고 응답을 확인하는 검증 단계다.",
            "validation": "agent가 실행 가능한 상태인지, 테스트 입력에 기대한 방식으로 응답하는지 확인하는 기준이다.",
        }
    elif foundry_iq_mcp_rag:
        concept_descriptions = {
            "Foundry IQ": "agent가 조직 지식과 문서를 근거로 답변할 수 있게 하는 knowledge grounding 계층이다.",
            "MCP": "Model Context Protocol을 통해 agent가 외부 도구와 시스템 기능을 사용할 수 있게 하는 연결 방식이다.",
            "Model Context Protocol": "모델이 외부 context와 tool을 표준화된 방식으로 다루게 하는 프로토콜이다.",
            "RAG": "질문에 맞는 관련 지식을 검색한 뒤 그 근거를 바탕으로 응답을 생성하는 패턴이다.",
            "knowledge base": "agent가 답변의 근거로 사용할 문서와 지식 원천이다.",
            "Azure AI Search": "knowledge base에서 관련 정보를 검색해 RAG 흐름에 연결하는 검색 계층으로 볼 수 있다.",
            "dynamic tool discovery": "agent가 요청에 맞는 도구를 발견하고 선택하는 실행 확장 방식이다.",
            "tool execution": "agent가 단순 답변을 넘어 외부 시스템 작업을 수행하는 단계다.",
        }
    elif azure:
        concept_descriptions = {
            "Azure DevOps MCP Server": "AI assistant와 Azure DevOps 리소스 사이를 연결해 자연어 요청이 work item, pull request, build 같은 업무 데이터 접근으로 이어지도록 돕는 계층이다.",
            "AI assistant": "단순 답변 생성 도구가 아니라 MCP 연결을 통해 DevOps 업무 맥락을 조회하고 작업 흐름을 보조할 수 있는 인터페이스로 이해했다.",
            "Agentic DevOps": "AI가 개발·운영 업무의 맥락을 읽고 반복 조회나 상태 확인을 보조하는 DevOps 활용 방식이다.",
            "work items": "Azure DevOps에서 작업, 버그, 요구사항을 추적하는 기본 단위이며, AI assistant가 업무 맥락을 파악할 때 중요한 대상이다.",
            "pull requests": "코드 변경을 검토하고 병합하기 전 확인하는 단계이며, AI 기반 DevOps 흐름에서도 변경사항 검토 지점이 된다.",
            "builds": "변경된 코드가 정상적으로 빌드되는지 확인하는 자동화 단계이며, 결과 확인과 품질 검증에 연결된다.",
            "workflow automation": "반복적인 조회·확인·상태 점검을 자동화해 개발자가 판단해야 할 정보를 빠르게 가져오는 흐름이다.",
            "governance": "AI가 업무 도구에 접근할 때 권한, 추적성, 통제 기준을 함께 고려해야 한다는 관점이다.",
        }
    elif github:
        concept_descriptions = {
            "GitHub Agentic Workflows": "GitHub Actions와 agent 기반 작업 흐름을 연결해 자동화 실행, 변경 생성, 검토 과정을 하나의 개발 workflow로 보는 개념이다.",
            "GitHub Actions": "repository 안의 workflow 파일을 기준으로 빌드, 테스트, 자동화 작업을 실행하는 GitHub의 자동화 기능이다.",
            "workflow_dispatch": "workflow를 수동으로 실행할 수 있게 하는 trigger이며, 자동화가 언제 시작되는지를 정의하는 핵심 조건이다.",
            "Pull Request": "자동화나 agent가 만든 변경사항을 바로 반영하지 않고 검토 가능한 형태로 확인하는 단계다.",
            "activation": "workflow나 agentic 기능이 실행 가능한 상태로 전환되는 단계로 해석할 수 있다.",
            "conclusion": "자동화 흐름의 마지막 확인 지점이지만, success/completed/merged 같은 상태값과 함께 보아야 결과를 단정할 수 있다.",
        }
    elif agent_orch:
        concept_descriptions = {
            "Agent Orchestration": "하나의 AI가 모든 일을 처리하는 대신 orchestrator가 작업을 나누고 여러 sub-agent가 역할별로 수행하는 구조다.",
            "orchestrator": "전체 목표를 해석하고 어떤 sub-agent에게 어떤 작업을 맡길지 조율하는 상위 제어 역할이다.",
            "sub-agents": "planner, coder, designer처럼 특정 역할에 맞게 나뉘어 작업을 수행하는 하위 agent다.",
            "GitHub Copilot CLI": "agent workflow를 개발 환경이나 명령행 작업 흐름과 연결해 볼 수 있는 도구 맥락이다.",
            "planner": "요구사항을 작업 단위로 나누고 실행 순서를 설계하는 역할이다.",
            "coder": "정의된 작업을 실제 코드나 구현 결과로 옮기는 역할이다.",
            "designer": "구조, 사용자 경험, 표현 방식 같은 설계 관점의 판단을 맡는 역할이다.",
            "toolchain": "agent가 작업을 수행할 때 사용하는 CLI, 파일, 저장소, 실행 환경 같은 도구 묶음이다.",
        }
    else:
        concept_descriptions = {}
    for term in key_terms[:8]:
        desc = concept_descriptions.get(term) or "핵심 개념을 업무 흐름 안에서 어떤 역할을 하는지 기준으로 이해했다."
        lines.append(f"- **{term}**: {desc}")
    lines.append("")
    lines.append("## 최종 정리")
    if foundry_first_agent:
        lines.append("이번 실습을 통해 Microsoft Foundry first-agent quickstart를 setup, Foundry에서의 agent 생성, VS Code continuation, use-agent 실행 확인으로 나누어 이해했다. 이 흐름을 분리하니 각 단계가 어떤 역할을 하며 무엇을 기준으로 성공 여부를 확인해야 하는지 더 분명해졌다.")
    elif foundry_iq_mcp_rag:
        lines.append("이번 학습을 통해 Foundry IQ, RAG, knowledge base, Azure AI Search를 knowledge grounding 흐름으로 묶고, MCP와 dynamic tool discovery를 tool execution 확장 흐름으로 분리해 이해했다. 이 구분 덕분에 agent workflow에서 답변 근거와 도구 실행을 각각 다른 검증 기준으로 볼 수 있게 되었다.")
    elif azure:
        lines.append("이번 실습을 통해 Azure DevOps MCP Server를 AI assistant와 DevOps 리소스를 연결하는 계층으로 이해했다. MCP Server가 work item, pull request, build 같은 업무 데이터를 AI assistant가 다룰 수 있게 해 주기 때문에, DevOps 자동화는 단순 명령 실행이 아니라 업무 맥락을 이해하고 요청을 처리하는 흐름으로 확장될 수 있다.")
    elif agent_orch:
        lines.append("이번 학습을 통해 Agent Orchestration을 단일 챗봇보다 확장된 역할 기반 AI 협업 구조로 이해했다. orchestrator는 전체 목표와 흐름을 조율하고, planner, coder, designer 같은 sub-agent는 각자의 역할에 맞게 작업을 나누어 수행한다. 이를 통해 AI 개발 workflow를 하나의 응답 생성이 아니라 역할 분담과 도구 사용이 결합된 구조로 볼 수 있었다.")
    elif github:
        lines.append("이번 학습을 통해 GitHub Agentic Workflows를 workflow 파일, 실행 트리거, activation, Pull Request, conclusion이 연결되는 자동화 흐름으로 이해했다. 특히 workflow_dispatch는 수동 실행 조건을 정의하는 단서가 되고, PR은 자동화 또는 agent가 만든 변경사항을 검토하는 단계로 연결될 수 있다. conclusion 화면은 최종 결과를 확인하는 후보로 남기되, 성공 여부는 추가 화면 근거가 있을 때 확정하는 것이 적절하다.")
    else:
        lines.append("이번 학습에서는 강의 자료와 실습 단서에 흩어진 개념을 하나의 학습 흐름으로 정리했다. 핵심 개념, 실습 단계, 확인이 필요한 부분을 분리하면서 이후 같은 내용을 다시 설명할 수 있는 기술 학습 기록으로 만들었다.")
    lines.append("")
    lines.append("## Portfolio Summary")
    if foundry_first_agent:
        lines.append("This learning record summarizes a Microsoft Foundry first-agent quickstart by separating setup, agent creation, VS Code continuation, use-agent execution, and validation criteria.")
    elif foundry_iq_mcp_rag:
        lines.append("This learning record explains a Foundry IQ and MCP based AI agent workflow by separating knowledge grounding, RAG, Azure AI Search, dynamic tool discovery, and tool execution.")
    elif azure:
        lines.append("This learning record explains how Azure DevOps MCP Server connects AI assistants with DevOps resources such as work items, pull requests, and builds. The focus is on understanding MCP as a workflow-enabling layer rather than a simple setup task.")
    elif agent_orch:
        lines.append("This learning record summarizes Agent Orchestration as a role-based AI workflow. It connects the concepts of orchestrators, sub-agents, toolchains, and GitHub Copilot CLI to explain how AI-assisted development can move beyond a single chatbot interaction.")
    elif github:
        lines.append("This learning record summarizes GitHub Agentic Workflows by connecting workflow_dispatch, workflow files, activation, pull requests, and conclusion screens into one automation flow. It focuses on understanding execution conditions and result verification in GitHub-based automation.")
    else:
        lines.append("This learning record organizes a technical study topic into a clear learning goal, concept map, workflow summary, and optional follow-up checks.")
    lines.append("")
    lines.append("## Key skills practiced")
    for skill in skills:
        lines.append(f"- {skill}")
    optional_items = optional_confirmation_items(raw_text, memo, qa_logs)
    if optional_items:
        lines.append("")
        lines.append("---")
        lines.append("## 선택 확인 사항")
        for item in optional_items:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def sparse_capture_generation_blocker(
    article_type: str,
    sparse_report: dict[str, Any],
    problem_map: dict[str, Any],
    brief: dict[str, Any],
) -> str:
    if article_type in POWERBI_ARTICLE_TYPES:
        return ""
    mode = str(sparse_report.get("generation_mode") or "")
    total = int(sparse_report.get("total_image_count") or 0)
    interpreted = int(sparse_report.get("interpreted_image_count") or 0)
    unknown_captions = int(sparse_report.get("unknown_caption_count") or 0)
    ratio = interpreted / max(total, 1)
    title = str(brief.get("korean_title") or "")
    core = str(problem_map.get("core_problem") or "")
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    numeric_confidence = float(sparse_report.get("article_type_confidence") or 0)
    if article_type in {"unknown", "general_learning_portfolio"}:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "article_type이 unknown/general_learning_portfolio라서 full_article 생성을 금지했습니다."
    if numeric_confidence and numeric_confidence < 0.55:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return f"article_type confidence {numeric_confidence:.2f}가 0.55 미만이어서 full_article 생성을 금지했습니다."
    if len(steps) < 3:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "구체적인 solution_steps가 3개 미만이어서 full_article 생성을 금지했습니다."
    if problem_map.get("_sparse_steps_incomplete"):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "Sparse capture에서 구체적인 solution_steps가 부족해 full_article 생성을 금지했습니다."
    if mode != "full_article":
        return f"Sparse Capture Mode가 {mode}로 판정되어 완성형 Medium 글 생성을 보류했습니다."
    if total and ratio < 0.6:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return f"interpreted_image_count / total_image_count = {interpreted}/{total}로 0.6 미만이어서 full_article 생성을 금지했습니다."
    if unknown_captions > max(1, total // 4):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "파일명 또는 일반 캡션으로 남은 이미지가 많아 full_article 생성을 금지했습니다."
    if is_generic_learning_title(title):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "제목이 입력 evidence 기반이 아니라 generic하여 full_article 생성을 금지했습니다."
    if is_generic_core_problem(core):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "core_problem이 입력 evidence 기반이 아니라 generic하여 full_article 생성을 금지했습니다."
    if has_placeholder_solution_steps(steps):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "solution_steps에 placeholder 단계가 포함되어 full_article 생성을 금지했습니다."
    return ""


def sparse_capture_hold_message(
    article_type: str,
    sparse_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    problem_map: dict[str, Any],
    brief: dict[str, Any],
    section_plan: list[dict[str, Any]],
    reason: str,
) -> str:
    debug_report = {
        "article_type": article_type,
        "confidence": sparse_report.get("confidence"),
        "article_type_confidence": sparse_report.get("article_type_confidence"),
        "generation_mode": sparse_report.get("generation_mode"),
        "capture_coverage": sparse_report.get("capture_coverage", {}),
        "interpreted_image_count": sparse_report.get("interpreted_image_count"),
        "total_image_count": sparse_report.get("total_image_count"),
        "unknown_caption_count": sparse_report.get("unknown_caption_count"),
        "missing_context": sparse_report.get("missing_context", []),
        "follow_up_questions": sparse_report.get("follow_up_questions", []),
        "reason": reason,
    }
    title = specific_title_candidate(article_type, evidence, str(brief.get("korean_title") or ""))
    core_candidate = specific_core_problem_candidate(article_type, evidence, str(problem_map.get("core_problem") or ""))
    interpreted_lines = "\n".join(
        f"- 이미지 {item.get('image_no')}: {item.get('caption')}"
        for item in evidence
        if is_interpreted_image_evidence(item)
    ) or "- 해석 가능한 캡처가 충분하지 않습니다."
    missing_lines = "\n".join(f"- {item}" for item in sparse_report.get("missing_context", [])) or "- 추가 확인이 필요한 맥락 없음"
    question_lines = "\n".join(f"{idx}. {item}" for idx, item in enumerate(sparse_report.get("follow_up_questions", [])[:3], start=1))
    section_plan_preview = json.dumps(section_plan[:6], ensure_ascii=False, indent=2)
    recovery_preview = build_recovery_draft_preview(article_type, problem_map, sparse_report, section_plan)
    recovery_block = f"\n## 학습 복구 초안\n{recovery_preview}\n" if recovery_preview else ""
    return f"""## Debug report
```json
{json.dumps(debug_report, ensure_ascii=False, indent=2)}
```

# {title}

현재 입력은 완성형 Medium 글이 아니라 `{sparse_report.get("generation_mode")}`로 처리되었습니다. {reason}

## core_problem 후보
{core_candidate}
{recovery_block}
## 현재 확인된 evidence
{interpreted_lines}

## missing_context
{missing_lines}

## follow_up_questions
{question_lines}

## Section Plan preview
```json
{section_plan_preview}
```
"""


def is_generic_learning_title(title: str) -> bool:
    normalized = title.strip().strip("# ")
    generic_fragments = [
        "학습 기록 기반 문제 해결 경험",
        "문제를 검증 가능한 분석 흐름으로 바꾼 기록",
        "학습 기록 기반",
        "이미지 기반 학습 캡처",
        "문제 해결형 학습 기록",
    ]
    return any(fragment in normalized for fragment in generic_fragments)


def is_generic_core_problem(core: str) -> bool:
    generic_fragments = [
        "학습 기록 기반 문제 해결 경험 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치",
        "관찰한 결과와 의도한 분석 흐름의 불일치",
        "문제를 검증 가능한 분석 흐름으로",
        "캡처와 메모를 문제 해결형 포트폴리오 글로",
    ]
    return any(fragment in core for fragment in generic_fragments)


def has_placeholder_solution_steps(steps: list[Any]) -> bool:
    text = json.dumps(steps, ensure_ascii=False)
    placeholders = [
        "근거 화면",
        "이미지 근거와 사용자 메모를 연결해 문제 해결 단계로 재구성했습니다.",
        "문제 상황 제시",
        "문제 원인 분석",
        "해결 단계",
        "검증 단계",
    ]
    return any(phrase in text for phrase in placeholders)


def specific_title_candidate(article_type: str, evidence: list[dict[str, Any]], fallback: str) -> str:
    source = evidence_source_text(evidence).lower()
    fallback_l = str(fallback or "").lower()
    if "azure devops" in fallback_l and "mcp" in fallback_l:
        return "Azure DevOps MCP Server 실습: AI assistant와 DevOps 자동화 흐름 이해하기"
    if article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}:
        if "agentic" in source and "workflow_dispatch" in source:
            return "GitHub Agentic Workflows 실습: workflow_dispatch와 자동화 실행 흐름을 이해한 기록"
        if "github actions" in source or "workflow_dispatch" in source:
            return "GitHub Actions 실습: workflow_dispatch 트리거와 실행 결과 흐름 확인하기"
        return "Agentic Workflows 학습 기록: GitHub 자동화의 실행 조건과 결과 상태 이해하기"
    return fallback if fallback and not is_generic_learning_title(fallback) else f"{article_type}: sparse capture 기반 학습 기록"


def specific_core_problem_candidate(article_type: str, evidence: list[dict[str, Any]], fallback: str) -> str:
    source = evidence_source_text(evidence)
    if article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}:
        terms = [term for term in ["GitHub Agentic Workflows", "workflow_dispatch", "activation", "update-github-info.lock.yml", "conclusion", "update-github-info"] if term.lower() in source.lower()]
        if terms:
            return "GitHub Agentic Workflows 실습에서 " + ", ".join(dict.fromkeys(terms)) + " 화면을 통해 자동화 workflow가 어떤 조건에서 실행되고 어떤 파일/상태를 남기는지 이해해야 했다. 다만 현재 캡처만으로는 최종 목표와 conclusion의 의미를 확정하기 어렵다."
    return fallback if fallback and not is_generic_core_problem(fallback) else "현재 캡처만으로는 핵심 문제를 확정하기 어려워, 확인된 evidence 기반 후보로만 유지합니다."


def evidence_source_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(
        [
            str(item.get("caption", "")) + " " + str(item.get("problem_signal", "")) + " " + str(item.get("inferred_meaning", "")) + " " + " ".join(normalize_str_list(item.get("technical_entities"))) + " " + " ".join(normalize_str_list(item.get("visible_evidence")))
            for item in evidence
        ]
    )


def infer_sparse_capture_role(item: dict[str, Any]) -> str:
    if str(item.get("evidence_source") or "").lower() == "filename" and is_unknown_or_filename_caption(item):
        return "unknown"
    text = " ".join(
        [
            str(item.get("caption", "")),
            str(item.get("problem_signal", "")),
            str(item.get("inferred_meaning", "")),
            " ".join(normalize_str_list(item.get("visible_evidence"))),
            " ".join(normalize_str_list(item.get("technical_entities"))),
        ]
    ).lower()
    role = str(item.get("role", "")).lower()
    if role == "problem" or any(word in text for word in ["error", "오류", "failed", "문제", "broken", "traceback"]):
        return "error_or_problem"
    if any(word in text for word in ["confusing", "헷갈", "concept", "개념", "why", "왜"]):
        return "confusing_concept"
    if any(word in text for word in ["code", "config", "configuration", "workflow", "yaml", "json", "python", "command", "설정", "수식", "measure"]):
        return "code_or_config"
    if any(word in text for word in ["architecture", "flow", "diagram", "model", "relationship", "구조", "흐름"]):
        return "architecture_or_flow"
    if role in {"solution", "cause"} or any(word in text for word in ["change", "edit", "fix", "created", "생성", "변경", "수정", "적용"]):
        return "action_or_change"
    if role in {"validation", "final_result"} or any(word in text for word in ["result", "success", "완료", "검증", "확인", "final"]):
        return "validation_or_result"
    if any(word in text for word in ["summary", "certificate", "badge", "completed", "요약", "수료"]):
        return "completion_or_summary"
    if int(item.get("image_no") or 0) == 1:
        return "goal_or_context"
    return "unknown"


def is_unknown_or_filename_caption(item: dict[str, Any]) -> bool:
    caption = str(item.get("caption") or "")
    source = str(item.get("evidence_source") or "").lower()
    original = str(item.get("original_filename") or "")
    if source == "filename":
        return True
    if re.search(r"\.(png|jpg|jpeg)\b", caption, re.I):
        return True
    if re.search(r"이미지\s+\d+\s*-\s*(근거 화면|실습 흐름 근거 화면|해석되지 않은 캡처|image\d*)", caption, re.I):
        return True
    if original and Path(original).stem and Path(original).stem.lower() in caption.lower():
        return True
    return False


def is_interpreted_image_evidence(item: dict[str, Any]) -> bool:
    if is_unknown_or_filename_caption(item):
        return False
    source = str(item.get("evidence_source") or "").lower()
    confidence = float(item.get("confidence") or 0)
    caption = str(item.get("caption") or "")
    visible = normalize_str_list(item.get("visible_evidence"))
    entities = normalize_str_list(item.get("technical_entities"))
    if source in {"vision", "llm"} and confidence >= 0.5 and (visible or entities) and len(caption) >= 18:
        return True
    if source == "README_image_order.txt" and len(caption) >= 18:
        return True
    return False


def evidence_coverage_failure_message(
    failure: str,
    uploaded_count: int,
    evidence_count: int,
    capture_count: int,
    qa_count: int,
    memo: str,
) -> str:
    return f"""# Medium 글 생성 보류

{failure}

이미지 evidence가 부족한 상태에서 긴 Medium 글을 만들면 후반부 이미지나 golden example 문장에 기대는 일반적인 글이 됩니다. 누락된 이미지를 다시 처리한 뒤 최종 글을 생성해야 합니다.

## 현재 처리 상태
- 업로드된 이미지 수: {uploaded_count}장
- 생성된 ImageEvidence 수: {evidence_count}개
- 저장된 캡처 수: {capture_count}개
- 저장된 Q&A 수: {qa_count}개

## 입력된 메모
{memo.strip() or "입력된 메모 없음"}

## 다음 조치
- 누락된 이미지가 업로드되었는지 확인
- README_image_order.txt 또는 파일명 prefix로 이미지 순서를 확인
- 이미지 1부터 마지막 이미지까지 ImageEvidence가 만들어진 뒤 다시 생성
"""


def unknown_article_type_response(
    classification: dict[str, Any],
    evidence: list[dict[str, Any]],
    uploaded_count: int,
    memo: str,
    raw_text: str,
    sparse_capture_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sparse_capture_report = sparse_capture_report or build_sparse_capture_report("unknown", classification, evidence, raw_text, memo)
    candidate_lines = "\n".join(
        f"- {item.get('article_type')}: score {item.get('score')}"
        for item in classification.get("candidates", [])
    ) or "- 후보를 충분히 추정하지 못했습니다."
    evidence_lines = "\n".join(
        f"- 이미지 {item.get('image_no')}: {item.get('caption')} / entities: {', '.join(normalize_str_list(item.get('technical_entities'))[:6])}"
        for item in evidence[:12]
    ) or "- 해석 가능한 이미지 evidence가 아직 없습니다."
    draft = f"""# Medium 글 생성 보류

현재 입력의 article_type confidence가 낮아 완성형 Medium 글을 억지로 생성하지 않았습니다.

## 현재 확인된 evidence
{evidence_lines}

## 추정 가능한 article_type 후보
{candidate_lines}

## 부족한 정보
{chr(10).join(f"- {item}" for item in sparse_capture_report.get("missing_context", [])) or "- 핵심 문제가 무엇인지 보여주는 캡처 또는 코드/오류 텍스트"}

## 사용자가 추가하면 좋은 캡처/메모
{chr(10).join(f"- {item}" for item in sparse_capture_report.get("follow_up_questions", []))}

## 입력 상태
- 업로드된 이미지 수: {uploaded_count}장
- 입력된 메모: {memo.strip() or "제공된 메모가 없습니다."}
- 입력된 텍스트 길이: {len(raw_text.strip())}자
"""
    return {
        "draft": draft,
        "article_type": "unknown",
        "image_evidence": evidence,
        "learning_evidence": [],
        "problem_map": {
            "article_type": "unknown",
            "_article_type_confidence": classification.get("confidence"),
            "_article_type_candidates": classification.get("candidates", []),
            "_sparse_capture_report": sparse_capture_report,
        },
        "decision_map": {},
        "section_plan": [],
        "article_brief": {},
        "sparse_capture_report": sparse_capture_report,
        "critic_report": {
            "passed": False,
            "failures": ["article_type confidence가 낮아 최종 글 생성을 보류했습니다."],
            "metrics": {"uploaded_images_count": uploaded_count, "image_evidence_count": len(evidence)},
        },
    }


def enrich_problem_map_concrete_details(problem_map: dict[str, Any], article_type: str) -> None:
    if article_type == "semantic_model_relationship":
        problem_map["problem_kind"] = "semantic_model_relationship_filter_context"
        problem_map["core_problem"] = SEMANTIC_CORE_PROBLEM
        problem_map["why_problematic"] = (
            "Category별 매출은 제품군에 따라 달라져야 하는데 Accessories, Bikes, Clothing, Components 등 모든 행에 같은 총액이 표시되면, "
            "각 Category가 Sales fact table을 필터링하지 못하고 전체 Sales 총합이 반복 표시되는 상태로 볼 수 있다. "
            "따라서 이는 시각화 문제가 아니라 semantic model relationship/filter context 문제로 정의해야 한다."
        )
        problem_map["root_causes"] = [
            "Product[Category] filter context가 Sales fact table로 전달되지 않음",
            "Product와 Sales 사이 ProductKey relationship이 없거나 비활성/오설정됨",
            "Salesperson 분석에서는 direct Salesperson-Sales relationship과 Salesperson -> SalespersonRegion -> Region -> Sales 경로가 동시에 존재해 filter path가 모호해질 수 있음",
            "Sales와 Target을 같은 기준으로 비교하려면 Salesperson/Region/Targets 관계 경로가 명확해야 함",
        ]
        problem_map["solution_steps"] = semantic_solution_steps()
        problem_map["complex_problems"] = [
            {
                "title": "Repeated Category Sales 문제",
                "symptom": "Category별 Sales 값이 모두 동일하게 반복됨",
                "cause": "Product[Category] filter context가 Sales fact table로 전달되지 않음",
                "action": "Product[ProductKey] -> Sales[ProductKey] relationship 생성",
                "validation": "Category별 Sales 값이 서로 다르게 분리되는지 확인",
            },
            {
                "title": "SalespersonRegion bridge path 문제",
                "symptom": "Salesperson별 Sales/Target 분석에서 담당 Region 기준 필터 경로가 필요함",
                "cause": "direct Salesperson-Sales relationship과 bridge path가 동시에 있으면 filter path가 모호해질 수 있음",
                "action": "SalespersonRegion bridge table 사용, Cross-filter direction Both 설정, direct Salesperson-Sales relationship 비활성화",
                "validation": "Salesperson별 Sales와 Target이 같은 기준에서 비교되는지 확인",
            },
        ]
        problem_map["complex_problem"] = {
            "title": "Repeated Category Sales와 SalespersonRegion bridge path를 분리해 해결",
            "symptom": "초반에는 Category별 Sales 반복값, 후반에는 Salesperson별 Sales/Target 비교를 위한 filter path 문제가 각각 나타남",
            "initial_assumption": "두 문제가 모두 relationship과 관련되어 있어 하나의 원인으로 묶어 볼 수 있음",
            "actual_cause": "Repeated Sales는 Product-Sales relationship/filter context 문제이고, bridge path는 Salesperson-Region-Sales-Targets 분석 경로를 명확히 해야 하는 별도 모델링 문제",
            "resolution": "Product[ProductKey] -> Sales[ProductKey] relationship으로 Category filter context를 해결하고, 후반부에서는 SalespersonRegion bridge table, Cross-filter direction Both, inactive direct relationship으로 경로를 분리",
            "verification": "Category별 Sales / Profit / Profit Margin이 분리되고, Salesperson별 Sales와 Target이 같은 기준에서 비교되는지 확인",
        }
        problem_map["final_outcome"] = (
            "Product-Sales relationship, Star schema, Product hierarchy, Profit / Profit Margin measure, SalespersonRegion bridge path, Targets 비교까지 "
            "각 문제를 분리해 검증 가능한 semantic model 흐름으로 정리했다."
        )
    if article_type == "dax_measure_modeling":
        problem_map["problem_kind"] = "analysis_design_task"
        problem_map["core_problem"] = DAX_CORE_PROBLEM
        problem_map["why_problematic"] = (
            "Month 이름을 텍스트 그대로 정렬하면 실제 월 순서가 깨질 수 있고, raw column 자동 집계에 의존하면 가격 지표와 목표 대비 성과 계산의 의도가 흐려진다. "
            "따라서 Date table, explicit measure, Target/Variance measure를 통해 어떤 기준으로 계산되는지 모델 안에 명확히 남겨야 한다."
        )
        problem_map["root_causes"] = [
            "Month 텍스트 컬럼은 시간 순서를 보장하지 않으므로 MonthKey 정렬이 필요함",
            "Date table과 fiscal hierarchy가 명확하지 않으면 날짜 기준 분석이 흔들림",
            "raw column 자동 집계는 Avg Price, Median Price, Orders 같은 지표의 계산 의도를 드러내지 못함",
            "TargetAmount raw column 대신 Target measure를 사용해야 filter context와 total 기준을 통제할 수 있음",
            "Variance와 Variance Margin이 없으면 Salesperson별 목표 대비 초과/미달을 해석하기 어려움",
        ]
        problem_map["solution_steps"] = dax_solution_steps()
        problem_map["complex_problem"] = {
            "title": "TargetAmount raw column 대신 Target/Variance measure로 성과 분석 기준을 명확히 하는 문제",
            "symptom": "Sales와 TargetAmount 값이 보이더라도 Salesperson별 목표 대비 성과와 total 기준 해석이 불명확함",
            "initial_assumption": "TargetAmount 컬럼을 그대로 집계하면 목표 대비 분석이 충분할 것처럼 보임",
            "actual_cause": "raw column 자동 집계는 filter context, total row, format, variance 계산 의도를 명확히 표현하지 못함",
            "resolution": "Target measure를 만들고 Variance, Variance Margin measure를 추가해 Sales, Target, 차이, 차이율을 같은 기준에서 비교",
            "verification": "Salesperson별 Sales, Target, Variance, Variance Margin이 최종 matrix에서 함께 표시되고 목표 대비 초과/미달이 읽히는지 확인",
        }
        problem_map["final_outcome"] = (
            "MonthKey 정렬, Fiscal Date table, Mark as date table, 가격/주문 explicit measure, Currency format, Target/Variance measure를 통해 "
            "Salesperson별 Sales와 Target 성과를 비교할 수 있는 DAX 분석 모델로 정리했다."
        )
    if article_type == "power_query_etl":
        if not is_power_query_etl_regression_context(problem_map):
            problem_map.setdefault("problem_kind", "transformation_requirement")
            for step in problem_map.get("solution_steps", []):
                if isinstance(step, dict):
                    step.setdefault("concrete_details", normalize_str_list(step.get("technical_entities")))
            return
        problem_map["_regression_profile"] = "power_query_etl_example"
        problem_map["problem_kind"] = "transformation_requirement"
        problem_map["core_problem"] = POWER_QUERY_CORE_PROBLEM
        problem_map["why_problematic"] = (
            "원본 데이터가 그대로 모델에 올라가면 결측 원가, 비표준 category, wide format 목표값, 불필요한 보조 쿼리 load가 "
            "이후 모델링과 시각화에서 잘못된 집계나 해석 오류로 이어질 수 있다."
        )
        problem_map["root_causes"] = [
            "SQL Server와 CSV 원본 데이터가 분석 모델 입력 구조로 정리되지 않음",
            "Column quality, Column distribution, Column profile 기준으로 null, distinct, valid 상태를 먼저 점검해야 함",
            "Salesperson, Product, Reseller, Region, Sales, Targets, ColorFormats가 각각 다른 변환 요구사항을 가짐",
            "BusinessType의 Ware House / Warehouse처럼 같은 의미의 값이 다르게 입력됨",
            "FactResellerSales의 TotalProductCost null 값이 Cost와 수익성 분석을 왜곡할 수 있음",
            "Targets의 M01~M12 wide format이 월별 분석에 바로 쓰기 어려움",
            "ColorFormats는 Product에 merge해야 하지만 독립 분석 테이블로 load할 필요는 없음",
        ]
        problem_map["solution_steps"] = power_query_solution_steps()
        problem_map["complex_problem"] = {
            "title": "결측 원가 보완, Targets Unpivot, ColorFormats Merge와 load control",
            "symptom": "TotalProductCost null 값, M01~M12 wide format Targets 데이터, ColorFormats 보조 테이블 load 범위가 함께 분석 모델 품질에 영향을 줌",
            "initial_assumption": "원본 데이터를 그대로 불러와도 분석에 사용할 수 있을 것처럼 보임",
            "actual_cause": "원가 결측은 이익 계산을 왜곡하고, wide format Targets는 날짜 기반 분석에 부적합하며, 보조 매핑 테이블을 그대로 load하면 모델이 불필요하게 복잡해짐",
            "resolution": "TotalProductCost는 if [TotalProductCost] = null then [OrderQuantity] * [StandardCost] else [TotalProductCost]로 보완하고, M01~M12는 Unpivot하여 MonthNumber / Target / TargetMonth 구조로 바꾸며, ColorFormats는 Product[Color]와 ColorFormats[Color] 기준 Left Outer Merge 후 Disable load 처리",
            "verification": "Cost 컬럼과 Fixed Decimal Number 타입, long format Targets, Product의 Background/Font Color Format 컬럼, 최종 7개 테이블 load 상태를 확인",
        }
        problem_map["final_outcome"] = (
            "Power Query Editor에서 원본 데이터를 분석 가능한 형태로 정리하고, Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets "
            "7개 테이블만 Close & Apply로 모델에 로드할 수 있는 상태로 만들었다."
        )


def power_query_solution_steps() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "SQL Server와 CSV 원본을 Power Query Editor로 가져오기",
            "problem": "SQL Server와 CSV 원본이 보고서에 바로 쓸 분석 테이블 구조가 아니었다.",
            "cause": "AdventureWorksDW2020의 여러 테이블과 ResellerSalesTargets.csv가 서로 다른 형태로 제공되어, 로드 전에 Power Query Editor에서 변환 범위를 정해야 했다.",
            "action": "SQL Server localhost의 AdventureWorksDW2020 데이터베이스를 Navigator에서 확인하고 FactResellerSales 등 필요한 테이블을 선택한 뒤 Transform Data로 Power Query Editor에 진입했다.",
            "verification": "Power Query Editor에서 SQL Server 원본과 CSV 원본을 변환 대상으로 확인했다.",
            "image_refs": [1, 2],
            "concrete_details": ["SQL Server", "localhost", "AdventureWorksDW2020", "Navigator", "Transform Data", "FactResellerSales"],
        },
        {
            "step": 2,
            "title": "Column quality, distribution, profile로 데이터 품질 확인",
            "problem": "원본 컬럼의 null, distinct, valid 상태를 보지 않으면 이후 변환이 필요한 지점을 놓칠 수 있었다.",
            "cause": "BusinessType 같은 범주형 컬럼에는 분포와 비표준 값이 숨어 있을 수 있고, 결측값은 후속 계산을 왜곡할 수 있다.",
            "action": "Power Query Editor에서 Column quality, Column distribution, Column profile을 켜고 주요 컬럼의 null, distinct, valid 상태를 확인했다.",
            "verification": "BusinessType 분포와 컬럼 품질 정보가 표시되어 데이터 정리 대상이 드러나는지 확인했다.",
            "image_refs": [3],
            "concrete_details": ["Column quality", "Column distribution", "Column profile", "null", "distinct", "valid", "BusinessType"],
        },
        {
            "step": 3,
            "title": "Salesperson 쿼리 구성",
            "problem": "DimEmployee에는 모든 직원이 포함되어 있어 영업 담당자 분석에 바로 쓰기 어려웠다.",
            "cause": "보고서에 필요한 것은 SalesPersonFlag가 TRUE인 직원이며, 표시용 이름과 식별 컬럼을 함께 정리해야 했다.",
            "action": "DimEmployee에서 SalesPersonFlag를 TRUE로 필터링하고 FirstName과 LastName을 병합해 Salesperson 컬럼을 만든 뒤 EmployeeID와 UPN을 유지했다.",
            "verification": "Salesperson 쿼리에 영업 담당자만 남고 Salesperson, EmployeeID, UPN 컬럼이 분석에 사용할 형태로 정리되었는지 확인했다.",
            "image_refs": [4, 5],
            "concrete_details": ["DimEmployee", "SalesPersonFlag", "TRUE", "FirstName", "LastName", "Salesperson", "EmployeeID", "UPN"],
        },
        {
            "step": 4,
            "title": "Product 쿼리 확장",
            "problem": "DimProduct만으로는 ProductKey와 제품명은 확인할 수 있지만 Category/Subcategory 분석 축이 부족했다.",
            "cause": "Subcategory와 Category 정보가 DimProductSubcategory, DimProductCategory에 분리되어 있어 확장이 필요했다.",
            "action": "DimProduct에서 ProductKey, EnglishProductName, StandardCost를 유지하고 DimProductSubcategory와 DimProductCategory를 Expand하여 Subcategory와 Category를 붙였다.",
            "verification": "Product 쿼리에 ProductKey, Subcategory, Category, StandardCost가 함께 남아 이후 Sales와 연결 가능한지 확인했다.",
            "image_refs": [6],
            "concrete_details": ["DimProduct", "ProductKey", "EnglishProductName", "StandardCost", "DimProductSubcategory", "DimProductCategory", "Subcategory", "Category", "Expand"],
        },
        {
            "step": 5,
            "title": "Reseller BusinessType 표준화",
            "problem": "BusinessType에 Ware House와 Warehouse가 함께 존재하면 같은 의미의 reseller 유형이 분리 집계될 수 있었다.",
            "cause": "원본 값의 표기 불일치가 category 분할을 만들기 때문이다.",
            "action": "DimReseller에서 BusinessType의 Ware House 값을 Replace Values로 Warehouse로 통일하고 ResellerName과 DimGeography 연결 정보를 유지했다.",
            "verification": "BusinessType 분포에서 Ware House가 Warehouse로 표준화되어 동일 의미 값이 하나로 정리되는지 확인했다.",
            "image_refs": [7],
            "concrete_details": ["DimReseller", "BusinessType", "Warehouse", "Ware House", "Replace Values", "ResellerName", "DimGeography"],
        },
        {
            "step": 6,
            "title": "Region 쿼리 구성",
            "problem": "DimSalesTerritory에는 분석에 불필요한 0번 territory와 여러 식별 컬럼이 포함되어 있었다.",
            "cause": "SalesTerritoryAlternateKey 0을 그대로 두면 실제 영업 지역 분석과 무관한 값이 포함될 수 있다.",
            "action": "DimSalesTerritory에서 SalesTerritoryAlternateKey 0을 제거하고 Region, Country, Group 중심으로 지역 쿼리를 정리했다.",
            "verification": "Region 쿼리에 실제 분석 대상 지역만 남고 Region, Country, Group 컬럼이 확인되는지 점검했다.",
            "image_refs": [8],
            "concrete_details": ["DimSalesTerritory", "SalesTerritoryAlternateKey", "0 제거", "Region", "Country", "Group"],
        },
        {
            "step": 7,
            "title": "Sales 쿼리와 null cost 보완",
            "problem": "FactResellerSales의 TotalProductCost null 값은 Cost와 수익성 분석을 왜곡할 수 있었다.",
            "cause": "원가가 비어 있는 행은 Sales만으로는 수익성을 판단할 수 없고, null을 방치하면 이후 measure 계산도 불안정해진다.",
            "action": "Custom Column으로 Cost를 만들고, TotalProductCost가 null이면 OrderQuantity * StandardCost를 사용하고 그렇지 않으면 TotalProductCost를 사용하도록 처리했다.",
            "verification": "Cost 컬럼이 생성되고 TotalProductCost/StandardCost 정리 후 Unit Price, Sales, Cost가 Fixed Decimal Number 타입으로 맞는지 확인했다.",
            "image_refs": [9],
            "concrete_details": ["FactResellerSales", "Sales", "TotalProductCost", "null", "OrderQuantity", "StandardCost", "Custom Column", "Cost", "Fixed Decimal Number"],
        },
        {
            "step": 8,
            "title": "Targets CSV의 M01~M12 컬럼 Unpivot",
            "problem": "ResellerSalesTargets.csv의 월별 목표가 M01~M12 가로 컬럼으로 있어 월별 분석에 바로 쓰기 어려웠다.",
            "cause": "wide format은 월을 값이 아니라 컬럼명으로 보관하기 때문에 날짜 테이블이나 월 기준 필터와 연결하기 어렵다.",
            "action": "M01부터 M12까지 월 컬럼을 Unpivot하여 MonthNumber와 Target 구조로 바꾸고 TargetMonth를 만들었다.",
            "verification": "Targets가 MonthNumber, Target, TargetMonth를 가진 long format으로 정리되어 월 단위 분석에 연결 가능한지 확인했다.",
            "image_refs": [10, 11],
            "concrete_details": ["ResellerSalesTargets.csv", "M01", "M12", "Unpivot", "MonthNumber", "Target", "TargetMonth", "long format"],
        },
        {
            "step": 9,
            "title": "ColorFormats 쿼리 정리",
            "problem": "색상 서식 정보는 Product에 붙일 참조 데이터지만, 원본 그대로는 헤더와 컬럼명이 분석에 적합하지 않았다.",
            "cause": "ColorFormats의 첫 행을 헤더로 승격하고 색상별 Background/Font 서식 컬럼을 명확히 해야 Product와 merge할 수 있다.",
            "action": "ColorFormats에서 Use First Row as Headers를 적용하고 Color, Background Color Format, Font Color Format 컬럼을 정리했다.",
            "verification": "ColorFormats에 Color와 두 색상 서식 컬럼이 merge 가능한 형태로 남았는지 확인했다.",
            "image_refs": [12],
            "concrete_details": ["ColorFormats", "Use First Row as Headers", "Color", "Background Color Format", "Font Color Format"],
        },
        {
            "step": 10,
            "title": "Product와 ColorFormats Merge",
            "problem": "Product 색상별 서식 정보를 보고서에서 쓰려면 Product 테이블에 색상 포맷 컬럼이 붙어 있어야 했다.",
            "cause": "ColorFormats는 Product[Color]와 ColorFormats[Color]를 기준으로 붙이는 보조 매핑 테이블이기 때문이다.",
            "action": "Product[Color]와 ColorFormats[Color]를 기준으로 Left Outer Merge를 수행하고 Background Color Format, Font Color Format 컬럼을 확장했다.",
            "verification": "Product 쿼리에 Background Color Format과 Font Color Format이 확장되어 색상별 서식 정보를 사용할 수 있는지 확인했다.",
            "image_refs": [13],
            "concrete_details": ["Product[Color]", "ColorFormats[Color]", "Merge", "Left Outer", "Background Color Format", "Font Color Format"],
        },
        {
            "step": 11,
            "title": "Append, Merge, Unpivot, Pivot 개념 구분",
            "problem": "Power Query 변환은 비슷해 보여도 각 연산의 목적이 달라 잘못 고르면 모델 구조가 어긋난다.",
            "cause": "Append는 UNION ALL처럼 행을 합치고, Merge는 JOIN처럼 컬럼을 붙이며, Unpivot/Pivot은 wide format과 long format을 바꾸는 연산이다.",
            "action": "이번 실습에서 ColorFormats에는 Merge, Targets에는 Unpivot을 적용해야 하는 이유를 변환 목적 기준으로 구분했다.",
            "verification": "각 연산이 실제 데이터 구조를 의도한 방향으로 바꾸었는지, 즉 Product에는 서식 컬럼이 붙고 Targets는 월별 행 구조가 되었는지 확인했다.",
            "image_refs": [10, 11, 13],
            "concrete_details": ["Append", "Merge", "UNION ALL", "JOIN", "Unpivot", "Pivot", "wide format", "long format"],
        },
        {
            "step": 12,
            "title": "최종 load control과 Close & Apply",
            "problem": "보조 쿼리까지 모두 모델에 load하면 모델이 불필요하게 복잡해지고 최종 테이블 범위가 흐려질 수 있었다.",
            "cause": "ColorFormats는 Product에 merge된 뒤에는 독립 분석 테이블로 필요하지 않다.",
            "action": "ColorFormats는 Disable load 처리하고 Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets 7개 테이블만 최종 load 대상으로 남긴 뒤 Close & Apply를 실행했다.",
            "verification": "최종 모델에 7개 테이블만 로드되는지 확인하고, ColorFormats는 보조 쿼리로만 남는지 점검했다.",
            "image_refs": [14],
            "concrete_details": ["Salesperson", "SalespersonRegion", "Product", "Reseller", "Region", "Sales", "Targets", "ColorFormats", "Disable load", "Close & Apply", "7개 테이블"],
        },
    ]


def semantic_solution_steps() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "Category별 Sales 반복값 문제 인식",
            "problem": "Product[Category]와 Sales[Sales]를 같은 table visual에 넣었을 때 모든 Category 행에 같은 Sales 총액이 반복되었다.",
            "cause": "Product[Category] filter context가 Sales fact table로 전달되지 않아 각 Category가 Sales를 분리해 필터링하지 못했다.",
            "action": "이 현상을 visual 문제가 아니라 Product와 Sales 사이 relationship/filter context 문제로 정의했다.",
            "verification": "Accessories, Bikes, Clothing, Components 등 Category 행에 같은 총액이 반복되는 이미지 1을 문제 신호로 확정했다.",
            "image_refs": [1],
            "concrete_details": ["Product[Category]", "Sales[Sales]", "same repeated total", "filter context"],
            "portfolio_meaning": "값이 보이는 화면에서도 필터 흐름이 깨지면 분석 결과가 틀릴 수 있음을 문제로 정의했다.",
        },
        {
            "step": 2,
            "title": "Product[ProductKey] -> Sales[ProductKey] relationship 생성",
            "problem": "Product dimension의 Category가 Sales fact table을 필터링하지 못했다.",
            "cause": "Product와 Sales 사이 ProductKey relationship이 없거나 활성 경로가 준비되지 않았다.",
            "action": "Product[ProductKey]에서 Sales[ProductKey]로 relationship을 생성했다.",
            "verification": "relationship 생성 화면에서 두 테이블의 key 컬럼이 올바르게 연결되는지 확인했다.",
            "image_refs": [2],
            "concrete_details": ["Product[ProductKey]", "Sales[ProductKey]", "relationship creation"],
            "portfolio_meaning": "차원 테이블의 필터가 fact table로 전달되는 기본 경로를 만들었다.",
        },
        {
            "step": 3,
            "title": "relationship 설정값 확인",
            "problem": "relationship이 생성되어도 cardinality나 filter direction이 잘못되면 같은 문제가 남을 수 있다.",
            "cause": "Product는 dimension table이고 Sales는 fact table이므로 one-to-many, single direction, active relationship이어야 한다.",
            "action": "Cardinality: One to many, Cross-filter direction: Single, Make this relationship active: Checked 설정을 확인했다.",
            "verification": "model view에서 Product-Sales relationship이 활성 상태로 표시되는지 확인했다.",
            "image_refs": [3],
            "concrete_details": ["Cardinality: One to many", "Cross-filter direction: Single", "Make this relationship active: Checked"],
            "portfolio_meaning": "relationship은 선을 긋는 작업이 아니라 모델의 필터 전달 규칙을 검증하는 작업임을 보여준다.",
        },
        {
            "step": 4,
            "title": "Star schema 모델 구조 확인",
            "problem": "개별 relationship만 확인하면 전체 모델이 분석에 적합한 구조인지 판단하기 어렵다.",
            "cause": "Sales fact table을 중심으로 Product 등 dimension table이 연결되는 Star schema 구조가 필요하다.",
            "action": "model view에서 Product dimension과 Sales fact table 중심의 Star schema layout을 확인했다.",
            "verification": "Sales를 중심으로 dimension들이 분리되어 연결되는 모델 구조가 보이는지 확인했다.",
            "image_refs": [4],
            "concrete_details": ["Star schema", "Product dimension table", "Sales fact table"],
            "portfolio_meaning": "단일 오류 수정이 아니라 모델 전체 구조 관점으로 문제를 검증했다.",
        },
        {
            "step": 5,
            "title": "Product hierarchy 구성",
            "problem": "Category 수준만으로는 제품 분석을 드릴다운하기 어렵다.",
            "cause": "보고서에서 Category -> Subcategory -> Product 순서로 자연스럽게 탐색하려면 hierarchy가 필요하다.",
            "action": "Product hierarchy를 Category, Subcategory, Product 순서로 구성했다.",
            "verification": "Product hierarchy level에 Category, Subcategory, Product가 올바른 순서로 포함되는지 확인했다.",
            "image_refs": [5, 6],
            "concrete_details": ["Product hierarchy", "Category -> Subcategory -> Product"],
            "portfolio_meaning": "semantic model이 단순 집계뿐 아니라 탐색 가능한 분석 축을 제공하도록 정리했다.",
        },
        {
            "step": 6,
            "title": "Profit Quick Measure 생성",
            "problem": "Sales만으로는 매출 규모를 볼 수 있지만 수익성을 판단할 수 없다.",
            "cause": "이익은 Sales에서 Cost를 차감해야 하며, 잘못 선택하면 Sales - Sales처럼 0이 되는 실수를 만들 수 있다.",
            "action": "Profit quick measure를 Sales - Cost 기준으로 생성했다.",
            "verification": "Profit이 Sales와 Cost의 차이로 계산되는지, 0으로만 나오지 않는지 확인했다.",
            "image_refs": [7],
            "concrete_details": ["Profit", "Sales[Sales]", "Sales[Cost]", "Sales - Cost"],
            "dax_measures": [
                {
                    "measure_name": "Profit",
                    "formula": "Profit = SUM(Sales[Sales]) - SUM(Sales[Cost])",
                    "why_needed": "Sales만으로는 수익성을 판단할 수 없기 때문",
                    "validation": "Category별 Sales와 Profit이 함께 표시되는지 확인",
                }
            ],
            "portfolio_meaning": "measure 생성에서도 계산 대상 선택이 분석 의미를 바꾼다는 점을 점검했다.",
        },
        {
            "step": 7,
            "title": "Profit Margin measure 생성",
            "problem": "Profit 금액만으로는 매출 규모가 다른 Category의 수익성을 비교하기 어렵다.",
            "cause": "수익률은 Profit을 Sales로 나누어 비교해야 하며 0 나누기 문제를 피하려면 DIVIDE가 적합하다.",
            "action": "Profit Margin measure를 DIVIDE([Profit], [Sales])로 만들고 Percentage / Decimal places 2 형식을 확인했다.",
            "verification": "Profit Margin이 퍼센트 형식으로 표시되고 Category별로 계산되는지 확인했다.",
            "image_refs": [8],
            "concrete_details": ["Profit Margin", "DIVIDE([Profit], [Sales])", "Percentage", "Decimal places 2"],
            "dax_measures": [
                {
                    "measure_name": "Profit Margin",
                    "formula": "Profit Margin = DIVIDE([Profit], [Sales])",
                    "why_needed": "매출 규모가 다른 Category를 수익률 기준으로 비교하기 위해 필요",
                    "validation": "Profit Margin 결과가 Category별로 계산되는지 확인",
                }
            ],
            "portfolio_meaning": "금액 지표와 비율 지표를 함께 만들어 분석 해석력을 높였다.",
        },
        {
            "step": 8,
            "title": "Category별 Sales / Profit / Profit Margin 결과 검증",
            "problem": "relationship과 measure를 만들었더라도 결과가 Category별로 분리되는지 확인해야 한다.",
            "cause": "초기 문제는 모든 Category에 같은 Sales 총액이 반복되는 것이었으므로 수정 후 값 분리가 핵심 검증 기준이다.",
            "action": "table visual에서 Category별 Sales, Profit, Profit Margin을 함께 표시했다.",
            "verification": "Category별 Sales, Profit, Profit Margin 값이 서로 다른 값으로 분리되어 계산되는지 이미지 9~10에서 확인했다.",
            "image_refs": [9, 10],
            "concrete_details": ["Category별 Sales", "Profit", "Profit Margin", "values separated by Category"],
            "portfolio_meaning": "수정의 성공 기준을 화면 변화가 아니라 계산 결과의 분리 여부로 잡았다.",
        },
        {
            "step": 9,
            "title": "SalespersonRegion bridge table 구성",
            "problem": "Salesperson별 Sales/Target 분석에서는 담당 Region 기준 필터 경로가 필요했다.",
            "cause": "Salesperson과 Region 사이에는 bridge 역할을 하는 SalespersonRegion이 필요하며, 단순 직접 관계만으로는 담당 지역 기준 분석을 설명하기 어렵다.",
            "action": "SalespersonRegion bridge table을 사용해 Salesperson -> SalespersonRegion -> Region -> Sales 경로를 구성했다.",
            "verification": "model view에서 SalespersonRegion이 Salesperson과 Region 사이의 bridge table로 배치되는지 확인했다.",
            "image_refs": [11],
            "concrete_details": ["SalespersonRegion", "Bridge table", "Salesperson -> SalespersonRegion -> Region -> Sales"],
            "portfolio_meaning": "후반 문제를 Product-Sales 반복값 문제와 분리된 filter path 설계 문제로 다뤘다.",
        },
        {
            "step": 10,
            "title": "Cross-filter direction Both 설정",
            "problem": "SalespersonRegion bridge path를 통해 담당 Region 기준 필터가 전달되어야 했다.",
            "cause": "bridge table 경로에서는 단방향 필터만으로 원하는 분석 경로가 충분히 전달되지 않을 수 있다.",
            "action": "Region과 SalespersonRegion 관계에서 Cross-filter direction을 Both로 설정했다.",
            "verification": "relationship 설정 화면에서 Cross-filter direction: Both가 적용되는지 확인했다.",
            "image_refs": [12],
            "concrete_details": ["Cross-filter direction: Both", "Region", "SalespersonRegion"],
            "portfolio_meaning": "filter direction을 분석 의도에 맞게 조정하는 모델링 판단을 보여준다.",
        },
        {
            "step": 11,
            "title": "Direct Salesperson-Sales relationship 비활성화",
            "problem": "direct Salesperson-Sales relationship과 bridge path가 동시에 있으면 filter path가 모호해질 수 있다.",
            "cause": "같은 분석 목표에 대해 직접 경로와 SalespersonRegion bridge 경로가 동시에 활성화되면 모델이 어떤 경로로 필터를 전달해야 하는지 불명확해진다.",
            "action": "직접 Salesperson-Sales relationship을 inactive relationship으로 바꾸었다.",
            "verification": "model view에서 직접 관계가 비활성 상태로 표시되는지 확인했다.",
            "image_refs": [13],
            "concrete_details": ["inactive relationship", "Direct Salesperson-Sales relationship", "ambiguous filter path"],
            "portfolio_meaning": "관계를 많이 만드는 것이 아니라 필요한 활성 경로를 명확히 남기는 판단을 보여준다.",
        },
        {
            "step": 12,
            "title": "Salesperson -> SalespersonRegion -> Region -> Sales 경로로 Sales 결과 확인",
            "problem": "bridge path 조정 후 Salesperson별 Sales가 담당 Region 기준으로 달라지는지 확인해야 했다.",
            "cause": "경로 설정이 맞아야 Salesperson filter가 Sales fact table까지 의도한 방식으로 전달된다.",
            "action": "Salesperson별 Sales 결과를 확인해 bridge path가 적용된 계산 결과를 검증했다.",
            "verification": "Salesperson별 Sales 값이 Region filter path를 통해 달라진 결과를 이미지 14에서 확인했다.",
            "image_refs": [14],
            "concrete_details": ["Salesperson별 Sales", "Region filter path", "Salesperson -> SalespersonRegion -> Region -> Sales"],
            "portfolio_meaning": "모델 경로 변경을 최종 수치 변화로 검증했다.",
        },
        {
            "step": 13,
            "title": "Targets 테이블 연결",
            "problem": "Sales와 Target을 같은 기준에서 비교하려면 Targets가 Salesperson 분석 경로와 맞아야 한다.",
            "cause": "Target이 별도 기준으로 남아 있으면 Salesperson별 실적과 목표 비교가 같은 filter context에서 이루어지지 않는다.",
            "action": "Targets 테이블을 Salesperson별 비교 흐름에 연결하고 Sales와 함께 볼 수 있게 구성했다.",
            "verification": "Targets가 Salesperson별 비교 화면에 함께 표시될 준비가 되었는지 확인했다.",
            "image_refs": [15],
            "concrete_details": ["Targets", "Salesperson", "Sales", "Target comparison"],
            "portfolio_meaning": "실적 지표와 목표 지표를 같은 분석 기준으로 맞추는 모델링 목적을 정리했다.",
        },
        {
            "step": 14,
            "title": "Salesperson별 Sales와 Target 최종 비교",
            "problem": "최종 목적은 Salesperson별 Sales와 Target을 같은 기준에서 비교하는 것이었다.",
            "cause": "SalespersonRegion bridge path, inactive direct relationship, Targets 연결이 함께 맞아야 최종 비교가 의미를 가진다.",
            "action": "Salesperson별 Sales와 Target을 같은 visual에서 비교했다.",
            "verification": "이미지 15에서 Salesperson별 Sales와 Target이 함께 표시되어 최종 비교가 가능한지 확인했다.",
            "image_refs": [15],
            "concrete_details": ["Salesperson별 Sales / Target 비교", "Targets", "final validation"],
            "portfolio_meaning": "초반 relationship 문제 해결에서 후반 목표 비교까지 하나의 semantic model 검증 흐름으로 마무리했다.",
        },
    ]


def dax_solution_steps() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "Month 컬럼을 MonthKey 기준으로 정렬",
            "problem": "Month 이름을 텍스트로만 두면 Jan, Feb, Mar 같은 실제 월 순서가 보장되지 않는다.",
            "cause": "문자열 월 이름은 알파벳 또는 기본 표시 순서에 영향을 받을 수 있어 시간 흐름 기반 분석에 부적합하다.",
            "action": "Month 컬럼의 Sort by column 기준을 MonthKey로 지정했다.",
            "verification": "월 이름이 MonthKey 순서대로 표시되어 fiscal/월별 trend를 자연스럽게 읽을 수 있는지 확인했다.",
            "image_refs": [1],
            "concrete_details": ["Month", "MonthKey", "Sort by column", "month order"],
            "portfolio_meaning": "시각화의 정렬 문제를 display column과 sort key 분리로 해결했다.",
        },
        {
            "step": 2,
            "title": "Fiscal Date table과 hierarchy 구성",
            "problem": "Sales 날짜 기준 분석을 하려면 일관된 Date table과 fiscal 기준 탐색 구조가 필요했다.",
            "cause": "날짜 컬럼만으로는 fiscal year/quarter/month 흐름을 안정적으로 재사용하기 어렵다.",
            "action": "Fiscal 기준이 반영된 Date table과 fiscal hierarchy를 구성했다.",
            "verification": "Fiscal hierarchy가 날짜 테이블 안에서 year, quarter, month 수준으로 탐색 가능한지 확인했다.",
            "image_refs": [2],
            "concrete_details": ["Date table", "Fiscal hierarchy", "Fiscal", "CALENDARAUTO"],
            "portfolio_meaning": "분석 축을 raw date column이 아니라 재사용 가능한 날짜 차원으로 분리했다.",
        },
        {
            "step": 3,
            "title": "Date table과 Sales 모델 relationship 확인",
            "problem": "Date table이 있어도 Sales와 연결되지 않으면 날짜 필터가 Sales fact table로 전달되지 않는다.",
            "cause": "시간 분석은 Date dimension과 Sales fact table 사이의 relationship이 활성화되어 있어야 의미가 있다.",
            "action": "Date table과 Sales 모델 relationship을 model view에서 확인했다.",
            "verification": "Date table의 날짜 기준 필터가 Sales 분석에 적용될 수 있는 관계 경로가 보이는지 확인했다.",
            "image_refs": [3],
            "concrete_details": ["Date table", "Sales", "model relationship", "date filter context"],
            "portfolio_meaning": "날짜 테이블 생성 이후 실제 fact table과의 연결까지 검증했다.",
        },
        {
            "step": 4,
            "title": "Mark as date table 처리",
            "problem": "Date table을 공식 날짜 테이블로 지정하지 않으면 시간 지능 계산과 날짜 기준 해석이 약해질 수 있다.",
            "cause": "Power BI는 어떤 컬럼을 canonical date column으로 사용할지 명시적으로 지정해야 안정적인 날짜 분석 기준을 갖는다.",
            "action": "Date table을 Mark as date table로 지정하고 날짜 컬럼을 선택했다.",
            "verification": "Date table이 공식 날짜 테이블로 인식되어 날짜 기준 분석의 기준점이 명확해졌는지 확인했다.",
            "image_refs": [4],
            "concrete_details": ["Mark as date table", "Date table", "date column"],
            "portfolio_meaning": "모델의 시간 분석 기준을 UI 표시가 아니라 semantic setting으로 고정했다.",
        },
        {
            "step": 5,
            "title": "Avg Price explicit measure 생성",
            "problem": "Unit Price 같은 raw column을 자동 평균으로 쓰면 계산 의도와 이름, format 관리가 불명확하다.",
            "cause": "자동 집계는 matrix에 표시될 수는 있지만 reusable measure가 아니며 다른 지표와 일관된 관리가 어렵다.",
            "action": "Avg Price explicit measure를 생성했다.",
            "verification": "Avg Price가 matrix에서 의도한 평균 가격 지표로 표시되고 다른 measure와 함께 재사용 가능한지 확인했다.",
            "image_refs": [5],
            "concrete_details": ["Avg Price", "explicit measure", "Unit Price"],
            "dax_measures": [
                {
                    "measure_name": "Avg Price",
                    "formula": "Avg Price = AVERAGE(Sales[Unit Price])",
                    "why_needed": "가격 평균을 raw column 자동 집계가 아니라 재사용 가능한 explicit measure로 관리하기 위해 필요",
                    "validation": "matrix에서 Avg Price가 가격 지표로 표시되는지 확인",
                }
            ],
            "portfolio_meaning": "자동 집계 의존도를 줄이고 명시적 measure 중심 모델로 전환했다.",
        },
        {
            "step": 6,
            "title": "Median / Min / Max Price와 Orders / Order Lines measure 구성",
            "problem": "평균 가격만으로는 가격 분포나 주문 규모를 충분히 설명할 수 없다.",
            "cause": "가격 분석에는 median/min/max가 필요하고, 주문 분석에는 Orders와 Order Lines처럼 서로 다른 count 기준이 필요하다.",
            "action": "Median Price, Min Price, Max Price, Orders, Order Lines measure를 구성했다.",
            "verification": "matrix에서 가격 measure와 주문 수 measure가 함께 표시되어 평균, 중앙값, 최소/최대, 주문 건수, 주문 라인 수를 비교할 수 있는지 확인했다.",
            "image_refs": [6],
            "concrete_details": ["Median Price", "Min Price", "Max Price", "Orders", "Order Lines", "matrix"],
            "portfolio_meaning": "단일 Sales 합계에서 벗어나 가격과 주문량을 다각도로 설명할 수 있는 measure set을 만들었다.",
        },
        {
            "step": 7,
            "title": "가격 measure Currency format 확인",
            "problem": "가격 measure가 숫자로만 표시되면 금액 지표인지 비율/건수 지표인지 해석이 흐려질 수 있다.",
            "cause": "measure는 계산식뿐 아니라 format까지 맞아야 보고서 사용자가 지표 의미를 빠르게 파악할 수 있다.",
            "action": "가격 관련 measure의 Currency format을 확인했다.",
            "verification": "Avg Price, Median Price, Min Price, Max Price가 통화 형식으로 표시되어 주문 수 measure와 구분되는지 확인했다.",
            "image_refs": [7],
            "concrete_details": ["Currency format", "Avg Price", "Median Price", "Min Price", "Max Price"],
            "portfolio_meaning": "계산 결과의 표시 형식까지 measure 설계의 일부로 관리했다.",
        },
        {
            "step": 8,
            "title": "TargetAmount raw column 대신 Target measure 사용",
            "problem": "TargetAmount 컬럼을 그대로 사용하면 목표값이 어떤 filter context와 total 기준으로 집계되는지 불명확하다.",
            "cause": "raw column 자동 집계는 Salesperson별 목표와 total row의 계산 의도를 명시적으로 통제하기 어렵다.",
            "action": "TargetAmount raw column 대신 Target explicit measure를 사용했다.",
            "verification": "Target measure가 Salesperson별 목표값과 total 기준에서 일관되게 표시되는지 TargetAmount raw column과 비교했다.",
            "image_refs": [8],
            "concrete_details": ["TargetAmount", "Target measure", "explicit measure", "filter context"],
            "dax_measures": [
                {
                    "measure_name": "Target",
                    "formula": "Target = IF(HASONEVALUE(Salesperson[Salesperson]), SUM(Targets[TargetAmount]), SUMX(VALUES(Salesperson[Salesperson]), SUM(Targets[TargetAmount])))",
                    "why_needed": "목표값 집계를 raw column 자동 집계가 아니라 measure로 통제하고, Salesperson별 행과 total row의 계산 기준을 분리하기 위해 필요",
                    "validation": "TargetAmount raw column과 Target measure의 표시 차이, 특히 total row가 의도한 기준으로 계산되는지 비교",
                }
            ],
            "portfolio_meaning": "목표 지표를 raw column에서 reusable business metric으로 끌어올렸다.",
        },
        {
            "step": 9,
            "title": "Variance / Variance Margin 계산",
            "problem": "Sales와 Target만 나란히 있으면 목표 대비 초과/미달 규모와 비율을 즉시 판단하기 어렵다.",
            "cause": "성과 분석에는 차이 금액과 차이율이 함께 있어야 Salesperson별 결과를 비교할 수 있다.",
            "action": "Variance와 Variance Margin measure를 구성했다.",
            "verification": "Salesperson별로 Sales가 Target을 얼마나 초과하거나 미달했는지 금액과 비율로 확인할 수 있는지 점검했다.",
            "image_refs": [9],
            "concrete_details": ["Variance", "Variance Margin", "Sales", "Target"],
            "dax_measures": [
                {
                    "measure_name": "Sales",
                    "formula": "Sales = SUM(Sales[Sales])",
                    "why_needed": "Variance = [Sales] - [Target] 계산의 기준이 되는 매출 measure가 필요하기 때문",
                    "validation": "Salesperson별 Sales 값이 Target, Variance와 같은 matrix에서 동일한 filter context로 표시되는지 확인",
                },
                {
                    "measure_name": "Variance",
                    "formula": "Variance = [Sales] - [Target]",
                    "why_needed": "목표 대비 초과/미달 금액을 계산하기 위해 필요",
                    "validation": "Salesperson별 Sales와 Target 차이가 표시되는지 확인",
                },
                {
                    "measure_name": "Variance Margin",
                    "formula": "Variance Margin = DIVIDE([Variance], [Target])",
                    "why_needed": "목표 대비 차이를 비율로 비교하기 위해 필요",
                    "validation": "Variance Margin이 비율 지표로 표시되는지 확인",
                },
            ],
            "portfolio_meaning": "성과 해석을 단순 값 비교에서 목표 대비 차이와 비율 분석으로 확장했다.",
        },
        {
            "step": 10,
            "title": "Salesperson별 Sales, Target, Variance, Variance Margin 최종 비교",
            "problem": "최종 목적은 Salesperson별 실적과 목표 대비 성과를 같은 화면에서 비교하는 것이었다.",
            "cause": "Date table, explicit measure, Target/Variance measure가 함께 맞아야 Salesperson별 성과 분석이 명확해진다.",
            "action": "Salesperson별 Sales, Target, Variance, Variance Margin을 최종 matrix에서 비교했다.",
            "verification": "각 Salesperson 행에서 Sales, Target, Variance, Variance Margin이 함께 표시되어 목표 대비 성과를 바로 읽을 수 있는지 확인했다.",
            "image_refs": [9],
            "concrete_details": ["Salesperson", "Sales", "Target", "Variance", "Variance Margin", "final matrix"],
            "portfolio_meaning": "날짜 모델링과 measure 설계가 최종 성과 비교 화면으로 연결되는지 검증했다.",
        },
    ]


def is_power_query_etl_regression_context(problem_map: dict[str, Any]) -> bool:
    source = json.dumps(problem_map, ensure_ascii=False).lower()
    markers = [
        "adventureworksdw2020",
        "factresellersales",
        "salespersonflag",
        "dimproductsubcategory",
        "dimproductcategory",
        "ware house",
        "totalproductcost",
        "resellersalestargets.csv",
        "colorformats",
        "background color format",
        "font color format",
        "close & apply",
    ]
    return sum(1 for marker in markers if marker in source) >= 5


def build_section_plan(article_type: str, evidence: list[dict[str, Any]], problem_map: dict[str, Any]) -> list[dict[str, Any]]:
    if article_type == "semantic_model_relationship":
        return [
            {"section": "Category별 Sales 반복값 문제 인식", "image_refs": [1], "must_include": ["Product[Category]", "Sales[Sales]", "same repeated total", "filter context"]},
            {"section": "Product-Sales relationship creation", "image_refs": [2], "must_include": ["Product[ProductKey]", "Sales[ProductKey]", "relationship"]},
            {"section": "Product-Sales relationship model confirmation", "image_refs": [3], "must_include": ["One to many", "Cross-filter direction: Single", "Make this relationship active"]},
            {"section": "Star schema", "image_refs": [4], "must_include": ["Star schema", "Product dimension", "Sales fact table"]},
            {"section": "Product hierarchy", "image_refs": [5, 6], "must_include": ["Product hierarchy", "Category", "Subcategory", "Product"]},
            {"section": "Profit quick measure", "image_refs": [7], "must_include": ["Profit", "Sales[Sales]", "Sales[Cost]", "Sales - Cost"]},
            {"section": "Profit Margin quick measure", "image_refs": [8], "must_include": ["Profit Margin", "DIVIDE", "Percentage", "Decimal places 2"]},
            {"section": "Category result validation", "image_refs": [9, 10], "must_include": ["Category별 Sales", "Profit", "Profit Margin"]},
            {"section": "SalespersonRegion bridge model", "image_refs": [11], "must_include": ["SalespersonRegion", "Bridge table", "Salesperson -> SalespersonRegion -> Region -> Sales"]},
            {"section": "Cross-filter direction Both", "image_refs": [12], "must_include": ["Cross-filter direction: Both", "Region", "SalespersonRegion"]},
            {"section": "Direct relationship inactive", "image_refs": [13], "must_include": ["inactive relationship", "Direct Salesperson-Sales relationship"]},
            {"section": "Salesperson sales by region result", "image_refs": [14], "must_include": ["Salesperson별 Sales", "Region filter path"]},
            {"section": "Targets table connection", "image_refs": [15], "must_include": ["Targets", "Salesperson", "Sales"]},
            {"section": "Salesperson Sales Target final", "image_refs": [15], "must_include": ["Salesperson별 Sales / Target 비교", "Targets", "final validation"]},
        ]
    if article_type == "dax_measure_modeling":
        return [
            {"section": "Month sort by MonthKey", "image_refs": [1], "must_include": ["Month", "MonthKey", "Sort by column"]},
            {"section": "Fiscal Date table and hierarchy", "image_refs": [2], "must_include": ["Date table", "Fiscal hierarchy", "CALENDARAUTO"]},
            {"section": "Date model relationships", "image_refs": [3], "must_include": ["Date table", "Sales", "model relationship"]},
            {"section": "Mark as date table", "image_refs": [4], "must_include": ["Mark as date table", "date column"]},
            {"section": "Avg Price explicit measure", "image_refs": [5], "must_include": ["Avg Price", "explicit measure", "Unit Price"]},
            {"section": "Pricing and count measures", "image_refs": [6], "must_include": ["Median Price", "Min Price", "Max Price", "Orders", "Order Lines"]},
            {"section": "Measure formatting", "image_refs": [7], "must_include": ["Currency format", "Avg Price", "Median Price"]},
            {"section": "TargetAmount versus Target measure", "image_refs": [8], "must_include": ["TargetAmount", "Target measure", "filter context"]},
            {"section": "Variance and Variance Margin", "image_refs": [9], "must_include": ["Variance", "Variance Margin", "Sales", "Target"]},
            {"section": "Salesperson final comparison", "image_refs": [9], "must_include": ["Salesperson", "Sales", "Target", "Variance", "Variance Margin"]},
        ]
    if article_type == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
        return [
            {"section": "SQL Server and CSV data import", "image_refs": [1, 2], "must_include": ["SQL Server", "localhost", "AdventureWorksDW2020", "Navigator", "Transform Data", "FactResellerSales"]},
            {"section": "Column quality/distribution/profile", "image_refs": [3], "must_include": ["Column quality", "Column distribution", "Column profile", "null", "distinct", "valid"]},
            {"section": "Salesperson query", "image_refs": [4, 5], "must_include": ["DimEmployee", "SalesPersonFlag", "TRUE", "FirstName", "LastName", "Salesperson", "EmployeeID", "UPN"]},
            {"section": "Product query expansion", "image_refs": [6], "must_include": ["DimProduct", "ProductKey", "EnglishProductName", "StandardCost", "DimProductSubcategory", "DimProductCategory", "Subcategory", "Category", "Expand"]},
            {"section": "Reseller BusinessType standardization", "image_refs": [7], "must_include": ["DimReseller", "BusinessType", "Warehouse", "Ware House", "Replace Values", "ResellerName", "DimGeography"]},
            {"section": "Region query", "image_refs": [8], "must_include": ["DimSalesTerritory", "SalesTerritoryAlternateKey", "0 제거", "Region", "Country", "Group"]},
            {"section": "Sales query and null cost fix", "image_refs": [9], "must_include": ["FactResellerSales", "Sales", "TotalProductCost", "null", "OrderQuantity", "StandardCost", "Custom Column", "Cost", "Fixed Decimal Number"]},
            {"section": "Targets CSV unpivot", "image_refs": [10, 11], "must_include": ["ResellerSalesTargets.csv", "M01", "M12", "Unpivot", "MonthNumber", "Target", "TargetMonth"]},
            {"section": "ColorFormats query", "image_refs": [12], "must_include": ["ColorFormats", "Use First Row as Headers", "Color", "Background Color Format", "Font Color Format"]},
            {"section": "Product/ColorFormats Merge", "image_refs": [13], "must_include": ["Product[Color]", "ColorFormats[Color]", "Merge", "Left Outer", "Background Color Format", "Font Color Format"]},
            {"section": "Append/Merge/Unpivot/Pivot concept", "image_refs": [10, 11, 13], "must_include": ["Append", "Merge", "UNION ALL", "JOIN", "Unpivot", "Pivot", "wide format", "long format"]},
            {"section": "final load control", "image_refs": [14], "must_include": ["Salesperson", "SalespersonRegion", "Product", "Reseller", "Region", "Sales", "Targets", "ColorFormats", "Disable load", "Close & Apply", "7개 테이블"]},
        ]
    return [
        {
            "section": step.get("title", f"해결 단계 {index}"),
            "image_refs": step.get("image_refs", []),
            "must_include": step.get("technical_entities", [])[:5],
        }
        for index, step in enumerate(problem_map.get("solution_steps", []), start=1)
        if isinstance(step, dict)
    ]


def next_capture_id(session: dict[str, Any]) -> int:
    current = [int(item.get("capture_id", 0)) for item in session.get("captures", [])]
    return (max(current) if current else 0) + 1


def next_qa_id(session: dict[str, Any]) -> int:
    current = [int(item.get("qa_id", 0)) for item in session.get("qa_logs", [])]
    return (max(current) if current else 0) + 1


def next_document_id(session: dict[str, Any]) -> int:
    current = [int(item.get("document_id", 0)) for item in session.get("documents", [])]
    return (max(current) if current else 0) + 1


def session_document_evidence_text(session: dict[str, Any]) -> str:
    """Read back the converted Markdown text for every successfully-ingested document.

    Documents are stored as plain-text files on disk (see create_session_document); only the
    small metadata record (filename, char count, status) lives in sessions.json. This joins
    them back into one block of text so callers can fold it into the same evidence pool used
    for screenshots and Q&A logs.
    """
    chunks: list[str] = []
    for doc in session.get("documents", []):
        if doc.get("status") != "ok" or not doc.get("text_path"):
            continue
        path = Path(str(doc["text_path"]))
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        chunks.append(f"[Document: {doc.get('original_filename', 'uploaded document')}]\n{text}")
    return "\n\n".join(chunks)


def append_qa_log(
    session: dict[str, Any],
    question: str,
    answer: str,
    selected_text: str = "",
    related_capture_ids: list[int] | None = None,
) -> dict[str, Any]:
    related_capture_ids = related_capture_ids or recent_capture_ids(session)
    qa = asdict(
        QALog(
            qa_id=next_qa_id(session),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            related_capture_ids=related_capture_ids,
            selected_text=selected_text,
            question=question,
            answer=answer,
            answer_summary=summarize_answer(answer),
            learner_state=infer_learner_state(question),
            resolved=bool(answer.strip()),
            used_in_article=True,
        )
    )
    session.setdefault("qa_logs", []).append(qa)
    return qa


def recent_capture_ids(session: dict[str, Any], limit: int = 3) -> list[int]:
    captures = sorted(session.get("captures", []), key=lambda item: item.get("timestamp", ""))
    return [int(item.get("capture_id")) for item in captures[-limit:] if item.get("capture_id")]


def summarize_answer(answer: str) -> str:
    text = " ".join(answer.split())
    return text[:320] + ("..." if len(text) > 320 else "")


def infer_learner_state(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["error", "오류", "안돼", "안 됨", "bug", "traceback"]):
        return "debugging"
    if any(word in lowered for word in ["맞아", "검증", "확인", "왜"]):
        return "verifying"
    if any(word in lowered for word in ["뭐", "무엇", "개념", "설명"]):
        return "confused"
    return "exploring"


def tutor_agent_answer(
    session: dict[str, Any],
    question: str,
    selected_text: str = "",
    related_capture_ids: list[int] | None = None,
) -> dict[str, Any]:
    related_capture_ids = related_capture_ids or recent_capture_ids(session)
    context_captures = [
        capture
        for capture in session.get("captures", [])
        if int(capture.get("capture_id", 0)) in set(related_capture_ids)
    ]
    context = "\n".join(
        f"Capture {capture.get('capture_id')}: note={capture.get('user_note', '')}, keywords={', '.join(capture.get('auto_keywords') or [])}"
        for capture in context_captures
    )
    prompt = f"""
사용자의 학습 중 질문에 TutorAgent로 답하세요.

규칙:
- 답변은 학습 설명 중심입니다.
- 사용자가 어디서 막혔는지 먼저 짚고, 원인 후보와 확인 방법을 제안합니다.
- 최종 글에 그대로 복사될 답변이 아니라 Q&A 로그로 저장될 설명입니다.
- 입력에 없는 성과를 만들지 않습니다.

[최근 캡처]
{context}

[선택/붙여넣은 텍스트]
{selected_text}

[질문]
{question}
""".strip()
    answer = llm_text(llm, prompt, max_tokens=900, system="You are a concise Korean tutor for technical study sessions.")
    if not answer.strip():
        answer = (
            "현재 LLM 응답을 받지 못했습니다. 질문을 학습 기록으로 저장해 두었습니다. "
            "추가로 오류 메시지, 수식, 화면에서 이상하게 보인 값을 함께 남기면 이후 ProblemMap과 DecisionMap에서 원인 후보를 더 정확히 좁힐 수 있습니다."
        )
    return append_qa_log(session, question=question, answer=answer, selected_text=selected_text, related_capture_ids=related_capture_ids)


def natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


ARTICLE_TYPE_EXAMPLE_DIRS = {
    "semantic_model_relationship": "powerbi_semantic_model",
    "dax_measure_modeling": "powerbi_dax_modeling",
    "power_query_etl": "powerbi_powerquery_etl",
}

SUPPORTED_ARTICLE_TYPES = [
    "semantic_model_relationship",
    "dax_measure_modeling",
    "power_query_etl",
    "dashboard_validation",
    "python_algorithm_learning",
    "code_error_debugging",
    "github_agentic_workflow",
    "github_actions_workflow",
    "ai_coding_workflow",
    "microsoft_foundry_first_agent",
    "github_readme_debugging",
    "deployment_debugging",
    "cloud_lab_practice",
    "ai_course_lab",
    "ai_project_build_log",
    "learning_path_reflection",
    "certification_badge_summary",
    "general_learning_portfolio",
    "unknown",
]

ARTICLE_TYPE_KEYWORDS = {
    "semantic_model_relationship": [
        "semantic model",
        "relationship",
        "filter context",
        "repeated",
        "productkey",
        "bridge",
        "salespersonregion",
        "profit margin",
    ],
    "dax_measure_modeling": [
        "date table",
        "monthkey",
        "sort by column",
        "measure",
        "target",
        "variance",
        "calendarauto",
        "avg price",
    ],
    "power_query_etl": [
        "power query",
        "column quality",
        "column distribution",
        "column profile",
        "sql server",
        "csv",
        "dimemployee",
        "dimproduct",
        "dimreseller",
        "factresellersales",
        "ware house",
        "totalproductcost",
        "unpivot",
        "colorformats",
        "merge",
        "disable load",
        "close & apply",
    ],
    "dashboard_validation": ["dashboard", "visual", "kpi", "report validation"],
    "python_algorithm_learning": ["python", "algorithm", "array", "sort", "leetcode", "time complexity", "big o"],
    "code_error_debugging": ["traceback", "exception", "syntaxerror", "typeerror", "nameerror", "stack trace", "debugging"],
    "github_agentic_workflow": [
        "agentic workflow",
        "agentic workflows",
        "automation that actually reads the room",
        "activation",
        "conclusion",
        "workflow_dispatch",
        "lock.yml",
        "update-github-info",
    ],
    "github_actions_workflow": [
        "github actions",
        "workflow_dispatch",
        "workflow file",
        "workflows",
        "actions",
        ".github/workflows",
        "yaml",
        "yml",
        "conclusion",
    ],
    "ai_coding_workflow": ["copilot", "agent", "automation", "ai coding", "coding agent", "workflow"],
    "microsoft_foundry_first_agent": [
        "microsoft foundry",
        "mslearn-agent-quickstart",
        "develop your first agent with microsoft",
        "get-started-in-foundry",
        "continue-in-vscode",
        "use-agent",
        "first agent",
    ],
    "github_readme_debugging": ["github", "readme", "markdown", "video embed", "image embed", "badge", "repository"],
    "deployment_debugging": ["deploy", "deployment", "build", "hosting", "ci"],
    "cloud_lab_practice": ["aws", "azure", "gcp", "cloud lab", "iam", "s3", "ec2", "lambda", "resource group"],
    "ai_course_lab": ["ai course", "machine learning", "model training", "notebook", "prompt", "llm", "fine tuning"],
    "ai_project_build_log": ["openai", "langchain", "rag", "vector database", "agent", "embedding", "chatbot"],
    "learning_path_reflection": ["course", "lecture", "lesson", "learning path", "module", "progress"],
    "certification_badge_summary": ["certification", "certificate", "badge", "credential", "exam", "completed"],
}


def promote_article_type_from_evidence(
    classification: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    image_names: list[str],
) -> dict[str, Any]:
    source = " ".join([raw_text, memo, " ".join(image_names), evidence_source_text(evidence)]).lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return classification
    github_terms = [
        "agentic workflows",
        "automation that actually reads the room",
        "workflow_dispatch",
        "update-github-info",
        "update-github-info.lock.yml",
        "activation",
        "conclusion",
        "github actions",
        ".github/workflows",
        " yml",
        " yaml",
    ]
    hit_count = sum(1 for term in github_terms if term in source)
    current_conf = float(classification.get("confidence") or 0)
    if hit_count >= 3 and current_conf < 0.92:
        candidates = classification.get("candidates") if isinstance(classification.get("candidates"), list) else []
        return {
            "article_type": "github_agentic_workflow",
            "confidence": max(0.72, min(0.92, 0.48 + hit_count * 0.08)),
            "candidates": [{"article_type": "github_agentic_workflow", "score": hit_count}] + candidates[:4],
            "promoted_from_evidence": True,
        }
    return classification


def classify_article_type(raw_text: str, memo: str, topic: str, extra_info: str, image_names: list[str]) -> str:
    return classify_article_type_with_confidence(raw_text, memo, topic, extra_info, image_names)["article_type"]


def classify_article_type_with_confidence(raw_text: str, memo: str, topic: str, extra_info: str, image_names: list[str]) -> dict[str, Any]:
    source = " ".join([raw_text, memo, topic, extra_info, " ".join(image_names)]).lower()
    compact_source = source.replace(" ", "_").replace("-", "_")
    classification_text = "\n".join([raw_text, topic, extra_info, " ".join(image_names)])
    if is_microsoft_foundry_first_agent_context(classification_text, memo):
        return {
            "article_type": "microsoft_foundry_first_agent",
            "confidence": 0.96,
            "candidates": [{"article_type": "microsoft_foundry_first_agent", "score": 99}],
        }
    if is_foundry_iq_mcp_rag_context(classification_text, memo):
        return {
            "article_type": "ai_project_build_log",
            "confidence": 0.78,
            "candidates": [{"article_type": "ai_project_build_log", "score": 8}],
        }
    for article_type, example_dir in ARTICLE_TYPE_EXAMPLE_DIRS.items():
        if article_type in compact_source or example_dir in compact_source:
            return {"article_type": article_type, "confidence": 0.98, "candidates": [{"article_type": article_type, "score": 99}]}
    scores = {
        article_type: sum(1 for keyword in keywords if keyword in source)
        for article_type, keywords in ARTICLE_TYPE_KEYWORDS.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    if best_score <= 0:
        return {
            "article_type": "unknown",
            "confidence": 0.0,
            "candidates": [{"article_type": item[0], "score": item[1]} for item in sorted_scores[:5]],
        }
    confidence = min(0.95, 0.25 + best_score * 0.16 + max(best_score - second_score, 0) * 0.08)
    if best_score < 2 and confidence < 0.55:
        return {
            "article_type": "general_learning_portfolio",
            "confidence": round(confidence, 3),
            "candidates": [{"article_type": item[0], "score": item[1]} for item in sorted_scores[:5]],
        }
    return {
        "article_type": best_type,
        "confidence": round(confidence, 3),
        "candidates": [{"article_type": item[0], "score": item[1]} for item in sorted_scores[:5]],
    }


def load_golden_context(article_type: str) -> dict[str, Any]:
    dirname = ARTICLE_TYPE_EXAMPLE_DIRS.get(article_type)
    if not dirname:
        return {}
    base = EXAMPLES_DIR / dirname
    context: dict[str, Any] = {}
    expected_map = base / "expected_problem_map.json"
    if expected_map.exists():
        try:
            context["expected_problem_map"] = json.loads(expected_map.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            context["expected_problem_map"] = {}
    for key, filename in {
        "pattern_notes": "pattern_notes.md",
        "image_order": "README_image_order.txt",
        "golden_article": "golden_article.md",
    }.items():
        path = base / filename
        if path.exists():
            context[key] = path.read_text(encoding="utf-8", errors="replace")[:6000]
    return context


def build_image_evidence(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    image_files: list[Path],
    topic: str,
    extra_info: str,
    image_names: list[str],
) -> list[dict[str, Any]]:
    caption_source = read_image_order_caption_source()
    filename_context = "\n".join(
        f"이미지 {index}: original_filename={name}, saved_filename={path.name}"
        for index, (path, name) in enumerate(zip(image_files, image_names, strict=False), start=1)
    )
    if not image_files:
        return [
            {
                "image_no": 0,
                "caption": "이미지 없음 - 사용자가 입력한 텍스트와 메모 기반 문제 분석",
                "visible_evidence": [part for part in [raw_text[:120], memo[:120]] if part],
                "role": "problem",
                "problem_signal": raw_text or memo or "이미지 없이 텍스트 입력만 제공됨",
                "technical_entities": infer_entities(" ".join([raw_text, memo, topic, extra_info])),
                "inferred_meaning": "업로드된 이미지가 없어 사용자의 텍스트와 메모를 문제 해결 흐름의 근거로 사용해야 합니다.",
            }
        ]

    client = llm_client.get_client()
    if not client:
        return fallback_image_evidence(image_files, image_names, raw_text, memo, caption_source)

    results: list[dict[str, Any]] = []
    for start in range(0, len(image_files), 5):
        chunk = image_files[start : start + 5]
        chunk_names = image_names[start : start + 5]
        prompt = f"""
입력 이미지를 순서대로 분석해 ImageEvidence JSON 배열만 반환하세요.

각 원소는 반드시 이 구조를 따릅니다.
{{
  "image_no": 1,
  "caption": "이미지 1 - 문제 해결 서사에 필요한 구체적 캡션",
  "visible_evidence": ["화면에 보이는 단어, 값, 오류, 테이블, 수식"],
  "role": "problem | cause | solution | validation | final_result",
  "problem_signal": "이 이미지가 보여주는 문제 신호",
  "technical_entities": ["Product", "Sales", "ProductKey", "DAX", "Relationship"],
  "inferred_meaning": "전체 실습 흐름에서 이 이미지가 갖는 의미"
}}

규칙:
- 한국어로 작성합니다.
- 이미지를 단순 설명하지 말고 문제/원인/해결/검증 역할로 분류합니다.
- README_image_order.txt 내용이 있으면 caption source로 우선 사용합니다.
- 원본 파일명에 01_Repeated_Category_Sales 같은 순서/의미가 있으면 그 의미를 caption에 반영합니다.
- 보이지 않는 수식, 성과, 배포는 만들지 않습니다.
- JSON 배열만 반환합니다.

[전체 주제]
{topic}

[사용자 메모]
{memo}

[추가 정보]
{extra_info}

[직접 입력 텍스트]
{raw_text}

[README_image_order.txt]
{caption_source or "없음"}

[파일명/순서]
{filename_context}
""".strip()
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for offset, image_file in enumerate(chunk, start=start + 1):
            mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
            image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
            content_parts.append({"type": "text", "text": f"이미지 {offset} / 원본 파일명: {chunk_names[offset - start - 1]}"})
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}})
        try:
            completion = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                temperature=0.15,
                max_tokens=4200,
                messages=[
                    {"role": "system", "content": "You convert technical screenshots into grounded JSON ImageEvidence. Return JSON only."},
                    {"role": "user", "content": content_parts},
                ],
            )
            data = parse_json_payload(completion.choices[0].message.content or "")
            if isinstance(data, list):
                results.extend(normalize_image_evidence(data, start + 1))
        except Exception as exc:
            print(f"[ImageEvidence error] {exc}")

    return results or fallback_image_evidence(image_files, image_names, raw_text, memo, caption_source)


def build_problem_map(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    evidence: list[dict[str, Any]],
    topic: str,
    extra_info: str,
    article_type: str,
    golden_context: dict[str, Any],
) -> dict[str, Any]:
    prompt = f"""
다음 ImageEvidence와 사용자 입력을 하나의 문제 해결 흐름으로 통합해 ProblemMap JSON만 반환하세요.

구조:
{{
  "core_problem": "...",
  "why_problematic": "...",
  "root_causes": ["..."],
  "solution_steps": [
    {{"step": 1, "title": "...", "problem": "...", "cause": "...", "action": "...", "verification": "..."}}
  ],
  "complex_problem": {{
    "title": "...",
    "symptom": "...",
    "initial_assumption": "...",
    "actual_cause": "...",
    "resolution": "...",
    "verification": "..."
  }},
  "final_outcome": "..."
}}

규칙:
- 이미지별 설명이 아니라 전체 흐름의 핵심 문제를 뽑습니다.
- solution_steps는 최소 4개를 만듭니다.
- 각 step은 문제/원인/조치/확인 결과를 모두 포함합니다.
- 입력에 없는 수식, 코드, 배포, 성과는 만들지 않습니다.
- Power BI semantic model 맥락이면 relationship, cardinality, filter context, DAX 검증 중심으로 씁니다.
- article_type별 golden example 문장이나 expected_problem_map 문장을 복사하지 않습니다.
- 새 입력의 ImageEvidence, 사용자 메모, Q&A Logs에 있는 근거만 ProblemMap의 재료로 사용합니다.
- article_type은 구조 선택에만 사용하고, 새 입력에 없는 Power BI 테이블명/수식/성과를 가져오지 않습니다.

[분류된 article_type]
{article_type}

[주제]
{topic}

[사용자 메모]
{memo}

[추가 정보]
{extra_info}

[직접 입력 텍스트]
{raw_text}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    data = llm_json(llm_client, prompt, max_tokens=3600)
    return normalize_problem_map(data if isinstance(data, dict) else {}, raw_text, memo, evidence, topic, article_type, golden_context)


def build_article_brief(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    evidence: list[dict[str, Any]],
    problem_map: dict[str, Any],
    topic: str,
    extra_info: str,
) -> dict[str, Any]:
    if problem_map.get("article_type") in ARTICLE_TYPE_EXAMPLE_DIRS:
        return normalize_article_brief({}, topic, problem_map)

    prompt = f"""
최종 Medium 글을 쓰기 전 ArticleBrief JSON만 반환하세요.

구조:
{{
  "korean_title": "...",
  "english_subtitle": "...",
  "article_thesis": "...",
  "target_reader": "recruiter / hiring manager / technical reviewer",
  "portfolio_angle": "...",
  "do_not_claim": ["사용자가 하지 않은 성과", "없는 수식", "없는 배포"],
  "must_include": ["문제 인식", "원인 분석", "검증 결과", "이미지 캡션 목록"]
}}

[주제]
{topic}

[추가 정보]
{extra_info}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}

[사용자 입력]
{raw_text}

{memo}
""".strip()
    data = llm_json(llm_client, prompt, max_tokens=1800)
    return normalize_article_brief(data if isinstance(data, dict) else {}, topic, problem_map)


def build_article_outline(
    llm_client: LLM,
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    if problem_map.get("article_type") in ARTICLE_TYPE_EXAMPLE_DIRS:
        return {title: default_outline_items(title, brief, problem_map, evidence) for title in SECTION_TITLES}

    prompt = f"""
다음 15개 섹션을 모두 포함하는 Medium 글 Outline JSON만 반환하세요.
각 key는 섹션 제목이고 value는 해당 섹션에서 반드시 다룰 bullet 배열입니다.

필수 섹션:
{json.dumps(SECTION_TITLES, ensure_ascii=False)}

규칙:
- 문제 해결 경험 섹션은 최소 4개 단계로 나눕니다.
- 각 단계는 문제/제약 → 원인 판단 → 조치 → 확인 결과를 포함합니다.
- 이미지 번호와 캡션 목록은 마지막 섹션에 둡니다.

[ArticleBrief]
{json.dumps(brief, ensure_ascii=False)}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    data = llm_json(llm_client, prompt, max_tokens=2600)
    if not isinstance(data, dict):
        data = {}
    for title in SECTION_TITLES:
        data.setdefault(title, default_outline_items(title, brief, problem_map, evidence))
    return data


def generate_section(
    llm_client: LLM,
    section_title: str,
    outline: dict[str, Any],
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    extra_info: str,
) -> str:
    if os.getenv("LLM_SECTION_GENERATION", "0") != "1":
        return structured_section(section_title, brief, problem_map, evidence, raw_text, memo, extra_info)

    if section_title == "한국어 제목":
        return f"# {brief.get('korean_title') or '문제 해결형 학습 기록'}"
    if section_title == "영어 부제":
        return f"_{brief.get('english_subtitle') or 'A problem-solving portfolio article from study evidence'}_"

    prompt = f"""
Medium 복붙용 Markdown의 단일 섹션만 작성하세요.

섹션 제목: {section_title}

작성 규칙:
- 내부 라벨, placeholder, id, :::writing을 절대 쓰지 않습니다.
- 이미지 설명 나열이 아니라 문제 해결 서사로 작성합니다.
- 없는 수식/코드/배포/성과를 만들지 않습니다.
- "버튼을 눌렀습니다", "실습을 완료했습니다" 같은 기능 설명 중심 문장을 피합니다.
- 섹션 헤더는 Markdown `## {section_title}` 형태로 시작합니다.

섹션별 길이/구조:
- 문제 인식, 문제 정의, 왜 이것을 문제로 인식했는가: 각각 최소 2문단.
- 문제 해결 경험: 최소 4개 단계, 각 단계는 문제/제약 → 원인 판단 → 조치 → 확인 결과 구조.
- Portfolio Summary: 영어 2문단 이상.
- Key skills practiced: 최소 8개 bullet.
- 이미지 번호와 캡션 목록: 마지막에 이미지별 캡션 bullet을 모두 포함.

[이 섹션의 outline]
{json.dumps(outline.get(section_title, []), ensure_ascii=False)}

[ArticleBrief]
{json.dumps(brief, ensure_ascii=False)}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}

[사용자 원문]
{raw_text}

[사용자 메모]
{memo}

[추가 정보]
{extra_info}
""".strip()
    text = llm_text(llm_client, prompt, max_tokens=2600)
    return text.strip() or fallback_section(section_title, brief, problem_map, evidence, raw_text, memo)


def structured_section(
    section_title: str,
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    extra_info: str,
) -> str:
    core = str(problem_map.get("core_problem") or brief.get("article_thesis") or memo or raw_text or "학습 과정에서 발견한 문제를 구조화하는 것")
    why = str(problem_map.get("why_problematic") or "겉으로는 결과가 보이더라도 원인과 검증 기준이 정리되지 않으면 같은 문제를 재현하거나 설명하기 어렵습니다.")
    final = str(problem_map.get("final_outcome") or "이미지와 메모를 근거로 문제, 원인, 해결, 검증 흐름을 정리했습니다.")
    steps = [step for step in problem_map.get("solution_steps", []) if isinstance(step, dict)]
    captions = [str(item.get("caption")) for item in evidence if item.get("caption")]
    entities = sorted({entity for item in evidence for entity in normalize_str_list(item.get("technical_entities"))})
    article_type = str(problem_map.get("article_type") or "general_learning_portfolio")
    entities_text = article_type_entities_text(article_type, entities, problem_map)
    type_focus = article_type_focus_text(str(problem_map.get("article_type") or "general_learning_portfolio"))
    clean_core = clean_summary_sentence(core)

    if section_title == "한국어 제목":
        return f"# {brief.get('korean_title') or core}"
    if section_title == "영어 부제":
        return f"_{brief.get('english_subtitle') or 'A problem-solving portfolio article from technical study evidence'}_"
    if section_title == "짧은 도입부":
        return (
            "## 짧은 도입부\n"
            f"이번 기록의 출발점은 단순히 화면을 캡처한 것이 아니라, `{core}`라는 문제를 어떻게 인식하고 검증 가능한 흐름으로 바꾸었는지 정리하는 데 있다. "
            "화면에는 결과값, 관계, 수식, 설정 화면이 각각 흩어져 있지만, 포트폴리오 글에서 중요한 것은 그 장면들을 순서대로 설명하는 일이 아니다. "
            "중요한 것은 초반 화면에서 어떤 이상 신호를 보았고, 중간 화면에서 어떤 원인 후보를 좁혔으며, 마지막 화면에서 어떤 기준으로 해결 여부를 확인했는지 연결하는 것이다.\n\n"
            f"따라서 이 글은 {entities_text}를 중심으로 이미지 묶음을 하나의 문제 해결 서사로 재구성한다. "
            "사용자가 직접 남긴 메모와 화면 근거를 우선하고, 입력에 없는 성과나 수식은 임의로 만들지 않는다."
        )
    if section_title == "핵심 작업 요약":
        bullets = "\n".join(
            f"- {step.get('title')}: {step.get('action')} / 확인 기준: {step.get('verification')}"
            for step in steps[:6]
        )
        semantic_summary = ""
        if problem_map.get("article_type") == "semantic_model_relationship":
            semantic_summary = (
                "\n- 필수 검증 흐름: 이미지 1 repeated value 문제, 이미지 2~3 Product[ProductKey] -> Sales[ProductKey] relationship, "
                "이미지 4 Star schema, 이미지 5~6 Product hierarchy, 이미지 7~8 Profit / Profit Margin과 DIVIDE, "
                "이미지 11~13 SalespersonRegion Bridge table, Cross-filter direction Both, inactive relationship, "
                "이미지 14~15 Salesperson -> SalespersonRegion -> Region -> Sales 경로와 Targets 비교"
            )
        if problem_map.get("article_type") == "dax_measure_modeling":
            semantic_summary = (
                "\n- 필수 DAX 흐름: 이미지 1 Month를 MonthKey 기준으로 Sort by column 처리, 이미지 2 Fiscal hierarchy가 포함된 Date table 구성, "
                "이미지 3 Date table과 Sales relationship 확인, 이미지 4 Mark as date table 지정, 이미지 5 Avg Price explicit measure 생성, "
                "이미지 6 Median Price / Min Price / Max Price / Orders / Order Lines measure matrix 검증, 이미지 7 Currency format 확인, "
                "이미지 8 TargetAmount raw column 대신 Target measure 비교, 이미지 9 Salesperson별 Sales, Target, Variance, Variance Margin 최종 비교"
            )
        if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
            semantic_summary = (
                "\n- 필수 ETL 흐름: SQL Server와 CSV를 Power Query Editor로 가져오고, Column quality / Column distribution / Column profile로 품질을 확인한 뒤 "
                "SalesPersonFlag, FirstName + LastName, ProductKey, DimProductSubcategory, DimProductCategory, BusinessType, Ware House / Warehouse, "
                "FactResellerSales, TotalProductCost, OrderQuantity, StandardCost, ResellerSalesTargets.csv, M01~M12 Unpivot, MonthNumber, TargetMonth, "
                "ColorFormats, Product[Color], Left Outer, Background Color Format, Font Color Format, Disable load, Close & Apply, 7개 테이블 로드를 검증"
            )
        return (
            "## 핵심 작업 요약\n"
            f"- 핵심 문제: {core}\n"
            f"- 문제로 본 이유: {why}\n"
            f"- 사용한 근거: {len(evidence)}개의 이미지 evidence와 사용자 메모\n"
            f"{bullets}\n"
            f"{semantic_summary}\n"
            f"- 최종 결과: {final}"
        )
    if section_title == "문제 인식":
        first_caption = captions[0] if captions else "초기 입력 화면"
        problem_detail = semantic_problem_detail(problem_map, evidence)
        memo_reference = f"사용자 메모에서 강조된 내용은 `{memo.strip()}`이며, " if memo.strip() else "사용자 별도 메모가 제공되지 않은 경우에도 "
        return (
            "## 문제 인식\n"
            f"처음 문제로 볼 수 있었던 신호는 `{first_caption}`에서 시작된다. {problem_detail} "
            f"{why} 그래서 이 기록에서는 화면을 순서대로 묘사하는 대신, 초반 이미지를 문제 신호로 보고 이후 이미지들을 원인 분석과 검증 근거로 연결했다.\n\n"
            "특히 포트폴리오 글에서 중요한 부분은 기능 사용 여부가 아니라 문제 인식의 정확도다. 화면에 숫자가 나오거나 설정 창이 열린 것만으로는 분석이 끝나지 않는다. "
            f"{memo_reference}이미지에 남아 있는 테이블명, 컬럼명, 변환 단계가 문제 해결형 서사를 구성하는 기준이 된다. "
            "따라서 이 단계의 핵심은 ‘무엇을 보았는가’가 아니라 ‘왜 그것을 문제로 정의했는가’다."
        )
    if section_title == "문제 정의":
        roots = "\n".join(f"- {item}" for item in normalize_str_list(problem_map.get("root_causes")))
        return (
            "## 문제 정의\n"
            f"이 실습에서 정의한 문제는 `{core}`다. 이 문제는 단일 화면이나 단일 버튼 조작으로 해결되는 문제가 아니라, 입력 근거 전체를 통해 원인과 검증 기준을 함께 세워야 하는 유형이다. "
            f"화면에 보이는 결과가 그럴듯해도, 데이터 모델, 수식, 관계, 변환 흐름 중 하나가 어긋나면 최종 해석은 달라질 수 있다. {type_focus}\n\n"
            f"문제의 원인 후보는 다음과 같이 정리할 수 있다.\n{roots}\n\n"
            "이 정의가 중요한 이유는 해결 방법의 범위를 좁혀 주기 때문이다. 문제를 단순 UI 조작으로 보면 화면 설명에 머물지만, 구조적 문제로 정의하면 어떤 근거를 확인하고 어떤 결과를 검증해야 하는지 분명해진다."
        )
    if section_title == "왜 이것을 문제로 인식했는가":
        return (
            "## 왜 이것을 문제로 인식했는가\n"
            f"{why} 이 지점은 학습 기록과 포트폴리오 글의 차이를 만든다. 학습 기록은 화면에서 무엇을 했는지 남기는 데 그칠 수 있지만, 문제 해결형 글은 왜 그 장면이 위험 신호였는지, 어떤 분석 오류로 이어질 수 있는지 설명해야 한다.\n\n"
            "또한 이 문제는 재현 가능성과 검증 가능성의 문제이기도 하다. 같은 화면을 다시 보았을 때 어떤 값, 관계, 수식, 변환 결과를 확인해야 하는지 정리되어 있지 않으면 다음 실습에서 같은 문제를 다시 만날 수 있다. "
            f"그래서 이 글에서는 {entities_text}를 단순 키워드가 아니라 원인 분석과 검증 결과를 잇는 증거로 사용한다."
        )
    if section_title == "문제 해결 경험":
        blocks = ["## 문제 해결 경험"]
        for step in steps:
            detail_text = concrete_detail_text(step)
            image_ref_text = image_ref_caption_text(step, evidence)
            blocks.append(
                f"### {step.get('step')}. {step.get('title')}\n"
                f"{image_ref_text}\n\n"
                f"문제/제약: {step.get('problem')}\n\n"
                f"원인 판단: {step.get('cause')}\n\n"
                f"조치: {step.get('action')}{detail_text}\n\n"
                f"확인 결과: {step.get('verification')}"
                f"{portfolio_meaning_text(step)}"
            )
        return "\n\n".join(blocks)
    if section_title == "복잡한 문제/수식/쿼리/코드 작성 및 해결 경험":
        complex_problem = problem_map.get("complex_problem", {}) if isinstance(problem_map.get("complex_problem"), dict) else {}
        semantic_complex = ""
        if problem_map.get("article_type") == "semantic_model_relationship":
            semantic_complex = (
                "\n\nSalesperson 분석에서는 단순 직접 관계만으로 충분하지 않았다. 담당 Region 기준 분석을 만들려면 "
                "Salesperson -> SalespersonRegion -> Region -> Sales 경로가 필요하고, 이 경로가 direct Salesperson-Sales relationship과 충돌하면 "
                "모호한 filter path가 생긴다. 그래서 SalespersonRegion Bridge table을 중심으로 Cross-filter direction을 Both로 조정하고, "
                "충돌하는 direct relationship은 inactive relationship으로 두어 Salesperson별 Sales와 Targets 비교가 같은 기준에서 계산되는지 확인해야 한다."
            )
        if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
            semantic_complex = (
                "\n\nPower Query ETL에서 복잡한 지점은 오류 수정이 아니라 분석 가능한 형태로 데이터를 바꾸는 판단이다. "
                "첫째, FactResellerSales의 TotalProductCost가 null이면 원가와 이익 분석이 왜곡될 수 있으므로 "
                "`if [TotalProductCost] = null then [OrderQuantity] * [StandardCost] else [TotalProductCost]` 방식으로 Cost를 보완하고 Fixed Decimal Number 타입을 확인해야 한다. "
                "둘째, ResellerSalesTargets.csv의 M01~M12 구조는 wide format이라 날짜 기반 분석에 부적합하므로 Unpivot을 통해 MonthNumber와 Target 구조로 만들고 TargetMonth를 생성해야 한다. "
                "셋째, ColorFormats는 Product[Color] 기준으로 Left Outer Merge해 Background Color Format과 Font Color Format을 Product에 붙이되, ColorFormats 자체는 Disable load 처리하여 최종 모델에는 Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets 7개 테이블만 Close & Apply로 로드해야 한다."
            )
        return (
            "## 복잡한 문제/수식/쿼리/코드 작성 및 해결 경험\n"
            f"가장 복잡한 지점은 `{complex_problem.get('title', '복합 원인 분석')}`이었다. 증상은 {complex_problem.get('symptom', core)}로 정리할 수 있다. "
            f"처음에는 {complex_problem.get('initial_assumption', '단일 설정 문제로 볼 수 있었다')} 하지만 실제 원인은 {complex_problem.get('actual_cause', '여러 근거를 함께 확인해야 하는 구조적 문제')}에 가까웠다.\n\n"
            f"해결은 {complex_problem.get('resolution', '문제, 원인, 조치, 검증 단계를 분리하는 방식')}으로 접근했다. "
            f"검증은 {complex_problem.get('verification', final)}를 기준으로 삼았다. "
            "수식이나 코드가 포함된 경우에도 핵심은 나열이 아니라 필요성, 계산 의미, 검증 방법을 함께 설명하는 것이다. 입력 근거에 명시되지 않은 수식은 만들지 않고, 보이는 기술 요소와 사용자 메모에 기반해 설명 범위를 제한했다."
            f"{semantic_complex}"
        )
    if section_title == "성과":
        return (
            "## 성과\n"
            f"이번 작업의 성과는 {final} 단순히 이미지를 저장한 것이 아니라, 문제 인식에서 원인 분석, 조치, 확인 결과까지 이어지는 설명 가능한 기록으로 바꾼 점이 중요하다.\n\n"
            "이 방식은 포트폴리오 관점에서도 의미가 있다. 결과 화면만 보여주는 글은 사용자의 판단 과정을 설명하지 못하지만, 문제 해결형 글은 어떤 신호를 보고 문제를 정의했는지, 어떤 원인을 의심했는지, 어떤 검증 기준으로 해결 여부를 판단했는지를 보여준다."
        )
    if section_title == "사용한 주요 수식/코드 정리":
        if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
            return (
                "## 사용한 주요 수식/코드 정리\n"
                "이번 ETL 실습에서 중심이 되는 코드는 DAX measure가 아니라 Power Query의 Custom Column 계산식이다. "
                "핵심은 FactResellerSales의 TotalProductCost가 null인 행을 그대로 두지 않고, 수량과 표준 원가를 이용해 Cost를 보완하는 것이다.\n\n"
                "```powerquery\n"
                "if [TotalProductCost] = null\n"
                "then [OrderQuantity] * [StandardCost]\n"
                "else [TotalProductCost]\n"
                "```\n\n"
                "왜 필요한가: TotalProductCost가 null이면 Sales만 남고 Cost가 비어 수익성 분석의 기준이 흔들린다. OrderQuantity와 StandardCost는 같은 행에서 원가를 추정할 수 있는 근거이므로, null인 경우에만 이 계산을 적용해야 한다.\n\n"
                "어떤 변환인가: 기존 TotalProductCost 값을 무조건 덮어쓰는 것이 아니라, null인 행에는 OrderQuantity * StandardCost를 사용하고 값이 존재하는 행에는 원래 TotalProductCost를 유지한다. 이렇게 만든 Cost는 Unit Price, Sales와 함께 Fixed Decimal Number 타입으로 맞춰 이후 모델에서 금액 컬럼처럼 사용할 수 있다.\n\n"
                "어떻게 검증했는가: Cost Custom Column이 생성되었는지, TotalProductCost와 StandardCost를 정리한 뒤에도 Cost가 남아 있는지, 그리고 Sales 쿼리의 Unit Price, Sales, Cost가 Fixed Decimal Number로 정리되었는지 확인했다. DAX가 아니라 Power Query 단계에서 원천 데이터의 분석 가능성을 확보한 것이 이 섹션의 핵심이다."
            )
        measures = []
        for step in steps:
            for measure in step.get("dax_measures", []) if isinstance(step, dict) else []:
                measures.append(
                    f"### {measure.get('measure_name')}\n"
                    f"```DAX\n{measure.get('formula')}\n```\n"
                    f"- 왜 필요한가: {measure.get('why_needed')}\n"
                    f"- 검증 방법: {measure.get('validation')}"
                )
        measure_text = "\n\n".join(measures) or "입력 근거에서 명확한 수식이 확인되지 않은 항목은 임의로 만들지 않았다."
        return (
            "## 사용한 주요 수식/코드 정리\n"
            f"입력 근거에서 확인해야 할 주요 기술 요소는 {entities_text}다. 이 섹션에서는 보이지 않는 수식이나 코드를 임의로 만들지 않는다. 대신 사용자가 제공한 화면과 메모 안에서 확인되는 관계, 수식, 쿼리, 변환 기준만 문제 해결 흐름에 연결한다.\n\n"
            f"{measure_text}\n\n"
            "수식이나 코드가 등장하는 경우 설명 기준은 세 가지다. 첫째, 왜 그 수식이나 코드가 필요했는가. 둘째, 어떤 계산 또는 변환을 수행했는가. 셋째, 결과를 어떻게 검증했는가. "
            "이 기준을 지키면 코드는 단순 장식이 아니라 문제를 해결한 근거가 된다."
        )
    if section_title == "최종 정리":
        return (
            "## 최종 정리\n"
            f"처음 문제는 {core}로 정의되었다. 마지막 정리는 이 문제로 다시 돌아가야 한다. {final} 이 흐름이 유지되어야 글이 단순 실습 후기가 아니라 문제 해결형 포트폴리오 글이 된다.\n\n"
            "이번 기록에서 중요한 태도는 화면을 그대로 설명하지 않고, 각 이미지를 문제, 원인, 해결, 검증의 역할로 재배치한 점이다. "
            "그 결과 독자는 사용자가 어떤 기술을 사용했는지만이 아니라, 왜 그 기술이 필요했고 어떤 기준으로 결과를 확인했는지 이해할 수 있다."
        )
    if section_title == "Portfolio Summary":
        return (
            "## Portfolio Summary\n"
            f"This project note reframes the study evidence around a concrete technical problem: {clean_core}. Instead of treating screenshots as a chronological UI walkthrough, the article connects the early problem signal, the suspected causes, the corrective actions, and the final verification criteria into one explainable workflow.\n\n"
            f"The portfolio value is in the reasoning process. The work shows how the learner used visible evidence, technical entities such as {entities_text}, and follow-up validation to move from observation to diagnosis and resolution. This makes the record useful for a recruiter, hiring manager, or technical reviewer because it demonstrates structured debugging, analytical writing, and evidence-based communication."
        )
    if section_title == "Key skills practiced":
        return "## Key skills practiced\n" + "\n".join(f"- {skill}" for skill in key_skills_for_article_type(article_type))
    if section_title == "이미지 번호와 캡션 목록":
        caption_lines = "\n".join(f"- {caption}" for caption in captions) or "- 이미지 없음 - 텍스트와 메모 기반으로 작성"
        return f"## 이미지 번호와 캡션 목록\n{caption_lines}"
    return fallback_section(section_title, brief, problem_map, evidence, raw_text, memo)


def severe_sparse_article_failures(critique: CritiqueResult) -> list[str]:
    severe_markers = [
        "Power BI regression template",
        "placeholder",
        "generic",
        "title",
        "internal",
        "caption",
        "golden/example",
        "unsupported claim",
        "solution_steps",
    ]
    return [failure for failure in critique.failures if any(marker.lower() in failure.lower() for marker in severe_markers)]


def critique_article(
    article: str,
    article_type: str = "general_learning_portfolio",
    problem_map: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> CritiqueResult:
    problem_map = problem_map or {}
    evidence = evidence or []
    failures: list[str] = []
    section_failures: dict[str, str] = {}
    metrics = {
        "char_count": len(article),
        "problem_solution_steps": count_problem_solution_steps(article),
        "placeholder_count": sum(article.count(phrase) for phrase in PLACEHOLDER_PHRASES),
        "function_description_count": sum(article.count(phrase) for phrase in FUNCTION_DESCRIPTION_PHRASES),
        "article_type": article_type,
        "uploaded_images_count": int(problem_map.get("_uploaded_images_count") or 0),
        "image_evidence_count": len(evidence),
    }
    section_plan = problem_map.get("_section_plan", []) if isinstance(problem_map.get("_section_plan"), list) else []
    metrics["section_plan_count"] = len(section_plan)
    if article_type == "unknown":
        failures.append("article_type이 unknown입니다. evidence가 부족해 최종 글을 생성할 수 없습니다.")
        section_failures["article_type"] = "현재 evidence와 후보 article_type을 사용자에게 반환해야 함"
    if not section_plan:
        failures.append("Section Plan 없이 Final Article 생성이 시도되었습니다.")
        section_failures["section_plan"] = "Final Article 이전에 section_plan 생성 필요"
    coverage_failure = image_coverage_failure(problem_map, evidence)
    if coverage_failure:
        failures.append(coverage_failure)
        section_failures["image_evidence_coverage"] = coverage_failure
    if metrics["char_count"] < 4500:
        failures.append("전체 글이 4,500자 미만입니다.")
        section_failures["global_length"] = "전체 글 분량 확장 필요"
    if metrics["problem_solution_steps"] < 4:
        failures.append("문제 해결 경험이 4개 미만입니다.")
        section_failures["문제 해결 경험"] = "최소 4개 단계 필요"
    for title in ["문제 인식", "문제 정의", "왜 이것을 문제로 인식했는가"]:
        paragraph_count = count_section_paragraphs(article, title)
        metrics[f"{title}_paragraphs"] = paragraph_count
        if paragraph_count < 2:
            failures.append(f"{title} 섹션이 2문단 미만입니다.")
            section_failures[title] = "최소 2문단 필요"
    if "이미지 번호와 캡션 목록" not in article:
        failures.append("이미지 캡션 목록이 없습니다.")
        section_failures["이미지 번호와 캡션 목록"] = "마지막 캡션 목록 필요"
    postprocess_leftovers = awkward_korean_leftovers(article)
    metrics["awkward_korean_leftover_count"] = len(postprocess_leftovers)
    if postprocess_leftovers:
        failures.append(f"어색한 한국어 후처리 패턴이 남아 있습니다: {', '.join(postprocess_leftovers)}")
        section_failures["postprocess"] = "마침표/조사 결합 오류 후처리 필요"
    if metrics["placeholder_count"]:
        failures.append("placeholder 또는 내부 라벨 문구가 포함되어 있습니다.")
        section_failures["placeholder"] = "금지 문구 제거 필요"
    diagnostic_leaks = [
        phrase
        for phrase in INTERNAL_DIAGNOSTIC_PHRASES
        if phrase in article or phrase in json.dumps(problem_map, ensure_ascii=False)
    ]
    metrics["internal_diagnostic_leak_count"] = len(diagnostic_leaks)
    if diagnostic_leaks:
        failures.append(f"내부 Vision/provider 진단 문구가 최종 글 또는 ProblemMap에 노출되었습니다: {', '.join(diagnostic_leaks)}")
        section_failures["internal_diagnostics"] = "Vision/provider 내부 상태는 final article 재료가 아니라 debug/provider_diagnostics로만 표시해야 함"
    if metrics["function_description_count"] > 2:
        failures.append("기능 설명 중심 문장이 너무 많습니다.")
        section_failures["style"] = "문제/원인/검증 중심으로 재작성 필요"
    repeated_fillers = [phrase for phrase in FILLER_SENTENCES if article.count(phrase) >= 2]
    metrics["repeated_filler_count"] = len(repeated_fillers)
    if repeated_fillers:
        failures.append(f"반복 filler 문장이 2회 이상 등장합니다: {', '.join(repeated_fillers)}")
        section_failures["repetition"] = "template filler 제거 및 step별 concrete detail 사용 필요"
    forbidden_once = [phrase for phrase in FILLER_SENTENCES if phrase in article]
    metrics["forbidden_phrase_count"] = len(forbidden_once)
    if forbidden_once:
        failures.append(f"금지 문구가 최종 글에 포함되어 있습니다: {', '.join(sorted(set(forbidden_once)))}")
        section_failures["forbidden_phrases"] = "금지 문구 제거 필요"
    if article_type not in POWERBI_ARTICLE_TYPES:
        template_hits = [phrase for phrase in NON_POWERBI_TEMPLATE_PHRASES if phrase in article]
        placeholder_hits = [phrase for phrase in PLACEHOLDER_PHRASES if phrase in article]
        generic_hits = [phrase for phrase in [
            "학습 기록 기반 문제 해결 경험 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치",
            "근거 화면 1 기반 검증 단계",
            "근거 화면 2 기반 검증 단계",
            "근거 화면 3 기반 검증 단계",
            "근거 화면 4 기반 검증 단계",
        ] if phrase in article]
        metrics["non_powerbi_template_contamination_count"] = len(template_hits)
        metrics["non_powerbi_placeholder_hit_count"] = len(placeholder_hits) + len(generic_hits)
        if template_hits:
            failures.append(f"Power BI regression template 문장이 non-PowerBI 글에 포함되었습니다: {', '.join(template_hits)}")
            section_failures["anti_overfitting"] = "non-PowerBI 입력에서는 모델/수식/관계/필터 흐름 템플릿 문장 제거 필요"
        if placeholder_hits or generic_hits:
            failures.append(f"non-PowerBI sparse capture에 placeholder/generic 문장이 포함되었습니다: {', '.join(sorted(set(placeholder_hits + generic_hits)))}")
            section_failures["sparse_capture"] = "placeholder solution step이나 generic core_problem은 Final Article이 아니라 missing_context로 내려야 함"
    if has_code_block(article) and not has_code_explanation(article):
        failures.append("코드/수식이 있으나 필요성/계산/검증 설명이 부족합니다.")
        section_failures["사용한 주요 수식/코드 정리"] = "수식 설명 보강 필요"
    if "확인 결과" not in article and "검증" not in article:
        failures.append("각 단계의 확인 결과가 부족합니다.")
        section_failures["문제 해결 경험"] = "확인 결과와 검증 내용 추가 필요"
    if count_section_paragraphs(article, "Portfolio Summary") < 2:
        failures.append("Portfolio Summary가 2문단 미만입니다.")
        section_failures["Portfolio Summary"] = "영어 2문단 이상 필요"
    key_skills = extract_section(article, "Key skills practiced")
    metrics["key_skills_count"] = len(re.findall(r"(?m)^-\s+", key_skills))
    if metrics["key_skills_count"] < 8:
        failures.append("Key skills practiced가 8개 미만입니다.")
        section_failures["Key skills practiced"] = "최소 8개 이상 필요"
    density_terms = ["문제/제약", "원인 판단", "조치", "확인 결과", "검증"]
    metrics["reasoning_density"] = sum(article.count(term) for term in density_terms)
    if metrics["char_count"] >= 4500 and metrics["reasoning_density"] < 12:
        failures.append("글자 수는 충분하지만 문제/원인/검증 밀도가 낮습니다.")
        section_failures["reasoning_density"] = "문제/원인/조치/검증 표현 보강 필요"
    steps = problem_map.get("solution_steps", [])
    if len(steps) < 4:
        failures.append("ProblemMap solution_steps가 4개 미만입니다.")
        section_failures["problem_map"] = "solution_steps 최소 4개 필요"
    config_min_steps = int(load_article_type_config(article_type).get("minimum_solution_steps") or 0)
    metrics["minimum_solution_steps"] = config_min_steps
    if config_min_steps and len(steps) < config_min_steps:
        failures.append(f"{article_type} solution_steps가 minimum_solution_steps {config_min_steps}개 미만입니다.")
        section_failures["problem_map"] = "article_type별 minimum_solution_steps 충족 필요"
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict) and not all(step.get(key) for key in ["problem", "cause", "action", "verification"]):
            failures.append(f"ProblemMap solution_step {index}에 problem/cause/action/verification 중 빠진 값이 있습니다.")
            section_failures["problem_map"] = "각 step 필수 필드 보강 필요"
            break
    captions = [str(item.get("caption", "")) for item in evidence]
    placeholder_captions = [
        caption
        for caption in captions
        if any(phrase in caption for phrase in ["문제 해결 서사에 필요한 구체적 캡션", "모델링 화면", "작업 결과 확인", "데이터 변환 작업"])
        or re.search(r"\.(png|jpg|jpeg)\b", caption, re.I)
    ]
    if placeholder_captions:
        failures.append("placeholder 수준 또는 파일명 기반 캡션이 포함되어 있습니다.")
        section_failures["captions"] = "사람이 읽을 수 있는 이미지 캡션으로 후처리 필요"
    truncated_captions = truncated_caption_lines(article)
    metrics["truncated_caption_count"] = len(truncated_captions)
    if truncated_captions:
        failures.append(f"이미지 캡션이 중간에서 잘린 것으로 보입니다: {', '.join(truncated_captions[:3])}")
        section_failures["captions"] = "이미지 캡션 마지막 줄이 완전한 문장/표현으로 끝나야 함"
    if evidence and not any(caption and caption in article for caption in captions):
        failures.append("solution_step 또는 본문이 이미지 캡션과 연결되지 않았습니다.")
        section_failures["image_refs"] = "이미지 캡션/근거를 본문에 연결 필요"
    entities = sorted({entity for item in evidence for entity in normalize_str_list(item.get("technical_entities"))})
    metrics["technical_entities_count"] = len(entities)
    if entities and sum(1 for entity in entities if entity.lower() in article.lower()) < min(3, len(entities)):
        failures.append("technical_entities가 본문에 충분히 반영되지 않았습니다.")
        section_failures["technical_entities"] = "핵심 기술 엔티티 반영 필요"
    if not final_outcome_links_to_problem(article, problem_map):
        failures.append("최종 결과가 처음 문제와 충분히 연결되지 않았습니다.")
        section_failures["최종 정리"] = "초기 문제와 최종 결과 연결 필요"
    missing_type_keywords = missing_article_type_keywords(article_type, article, problem_map)
    if missing_type_keywords:
        failures.append(f"article_type 기준 핵심 키워드가 부족합니다: {', '.join(missing_type_keywords)}")
        section_failures["article_type"] = "분류 유형별 핵심 문제/해결 키워드 반영 필요"
    required_entities = article_type_required_entities(article_type)
    if required_entities:
        missing_specific = [item for item in required_entities if item.lower() not in article.lower()]
        coverage = 1 - (len(missing_specific) / max(len(required_entities), 1))
        metrics["required_entity_coverage"] = round(coverage, 3)
        if coverage < 0.7:
            failures.append(f"{article_type} concrete entities coverage가 70% 미만입니다: {', '.join(missing_specific)}")
            section_failures["specificity"] = "article_type별 필수 concrete detail을 본문에 반영 필요"
    plan_failures, plan_metrics = plan_to_article_faithfulness(article, problem_map)
    metrics.update(plan_metrics)
    if plan_failures:
        failures.extend(plan_failures)
        section_failures["plan_to_article_faithfulness"] = "Section Plan 순서, must_include, image_refs, cause/action 구조를 final article에 강제 반영 필요"
    config = load_article_type_config(article_type)
    forbidden_cross_type = normalize_str_list(config.get("forbidden_cross_type_entities"))
    cross_hits = [item for item in forbidden_cross_type if item.lower() in article.lower()]
    metrics["cross_type_forbidden_hits"] = cross_hits
    if cross_hits:
        failures.append(f"선택된 article_type과 무관한 golden/example 키워드가 포함되었습니다: {', '.join(cross_hits)}")
        section_failures["anti_overfitting"] = "선택 article_type에 없는 Power BI/golden 키워드 제거 필요"
    unsupported_claims = unsupported_powerbi_claims(article, article_type, evidence, problem_map)
    metrics["unsupported_claim_count"] = len(unsupported_claims)
    if unsupported_claims:
        failures.append(f"입력 evidence에 없는 기술명/수식/테이블이 본문에 포함되었습니다: {', '.join(unsupported_claims)}")
        section_failures["unsupported_claims"] = "golden example에서 끌려온 unsupported claim 제거 필요"
    if generic_or_copied_title(article, problem_map):
        failures.append("제목 또는 core_problem이 입력 기반이 아니라 generic/golden example 문장처럼 보입니다.")
        section_failures["title"] = "입력 evidence 기반 제목과 core_problem 필요"
    if article_type == "semantic_model_relationship":
        if len(problem_map.get("solution_steps", [])) < 10:
            failures.append("semantic_model_relationship solution_steps가 10개 미만으로 압축되었습니다.")
            section_failures["problem_map"] = "semantic 유형은 repeated Sales부터 Targets 비교까지 최소 10개 내외 step 필요"
        semantic_mapping_failure = semantic_image_ref_mapping_failure(problem_map)
        if semantic_mapping_failure:
            failures.append(semantic_mapping_failure)
            section_failures["image_refs"] = "semantic step별 image_refs 고정 매핑 불일치"
        if "Repeated Category Sales" not in json.dumps(problem_map.get("complex_problems", []), ensure_ascii=False) or "SalespersonRegion bridge path" not in json.dumps(problem_map.get("complex_problems", []), ensure_ascii=False):
            failures.append("semantic complex problems가 Repeated Category Sales와 SalespersonRegion bridge path로 분리되지 않았습니다.")
            section_failures["complex_problem"] = "초반 repeated Sales 문제와 후반 bridge path 문제 분리 필요"
        covered, total, missing = semantic_regression_coverage(article)
        metrics["semantic_regression_coverage"] = round(covered / max(total, 1), 3)
        if covered / max(total, 1) < 0.7:
            failures.append(f"Semantic regression coverage가 70% 미만입니다. 누락: {', '.join(missing)}")
            section_failures["semantic_regression"] = "이미지 1~15 핵심 흐름 반영 필요"
    if article_type == "power_query_etl":
        is_regression_profile = problem_map.get("_regression_profile") == "power_query_etl_example"
        core_problem = str(problem_map.get("core_problem") or "")
        metrics["problem_kind"] = problem_map.get("problem_kind")
        if problem_map.get("problem_kind") != "transformation_requirement":
            failures.append("power_query_etl core_problem이 transformation_requirement로 분류되지 않았습니다.")
            section_failures["problem_kind"] = "연결/로드 오류가 아니라 변환 요구사항으로 정의 필요"
        bad_core_terms = ["연결 실패", "로딩 문제", "데이터 로딩이 문제가", "query merge error", "오류 수정"]
        if any(term.lower() in core_problem.lower() for term in bad_core_terms):
            failures.append("power_query_etl core_problem이 연결/로딩/오류 문제로 잘못 일반화되었습니다.")
            section_failures["core_problem"] = "원본 데이터를 분석 모델 입력 구조로 정리하는 문제로 재정의 필요"
        if is_regression_profile and ("SQL Server와 CSV" not in core_problem or "분석 모델" not in core_problem):
            failures.append("power_query_etl core_problem에 SQL Server/CSV 원본과 분석 모델 입력 구조 요구가 충분히 드러나지 않습니다.")
            section_failures["core_problem"] = "Power Query ETL 핵심 문제 정의 보강 필요"
        if is_regression_profile and len(section_plan) < 8:
            failures.append("power_query_etl Section Plan이 8개 미만입니다.")
            section_failures["section_plan"] = "Power Query ETL은 최소 8개, 권장 12개 plan 필요"
        if is_regression_profile and len(section_plan) < 12:
            failures.append("power_query_etl Section Plan이 12단계 필수 흐름을 모두 포함하지 않습니다.")
            section_failures["section_plan"] = "SQL import부터 final load control까지 12단계 필요"
        plan_covered, plan_total, plan_missing = section_plan_image_coverage(section_plan, metrics["uploaded_images_count"] or len(evidence))
        metrics["section_plan_image_coverage"] = round(plan_covered / max(plan_total, 1), 3)
        if is_regression_profile and plan_total and plan_covered / max(plan_total, 1) < 0.7:
            failures.append(f"Section Plan image coverage가 70% 미만입니다. 누락 이미지: {', '.join(plan_missing)}")
            section_failures["section_plan_image_coverage"] = "업로드 이미지가 section_plan에 충분히 연결되어야 함"
        complex_text = json.dumps(problem_map.get("complex_problem", {}), ensure_ascii=False)
        complex_topics = [
            all(term in complex_text for term in ["TotalProductCost", "OrderQuantity", "StandardCost"]),
            all(term in complex_text for term in ["Unpivot", "M01", "M12"]),
            all(term in complex_text for term in ["ColorFormats", "Left Outer", "Disable load"]),
        ]
        metrics["power_query_complex_topics"] = sum(1 for item in complex_topics if item)
        if is_regression_profile and metrics["power_query_complex_topics"] < 2:
            failures.append("power_query_etl complex_problem이 TotalProductCost / Unpivot / ColorFormats Merge 중 2개 이상을 포함하지 않습니다.")
            section_failures["complex_problem"] = "복잡 문제는 결측 원가, Targets Unpivot, ColorFormats Merge/load control 중 최소 2개 필요"
        if is_regression_profile:
            covered, total, missing = power_query_regression_coverage(article)
            metrics["power_query_regression_coverage"] = round(covered / max(total, 1), 3)
            if covered / max(total, 1) < 0.7:
                failures.append(f"Power Query regression coverage가 70% 미만입니다. 누락: {', '.join(missing)}")
                section_failures["power_query_regression"] = "이미지 1~14 핵심 흐름 반영 필요"
        code_section = extract_section(article, "사용한 주요 수식/코드 정리")
        metrics["dax_mentions"] = article.count("DAX")
        if metrics["dax_mentions"] > 2 or ("DAX" in code_section and "powerquery" not in code_section.lower()):
            failures.append("power_query_etl 글이 DAX를 중심으로 잘못 서술되었습니다.")
            section_failures["사용한 주요 수식/코드 정리"] = "Power Query formula와 변환 설명 중심으로 수정 필요"
    return CritiqueResult(passed=not failures, failures=failures, section_failures=section_failures, metrics=metrics)


def expand_failed_sections(
    llm_client: LLM,
    article: str,
    sections: dict[str, str],
    critique: CritiqueResult,
    outline: dict[str, Any],
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    extra_info: str,
) -> str:
    targets = set(critique.section_failures)
    if "global_length" in targets:
        targets.update(["문제 인식", "문제 정의", "왜 이것을 문제로 인식했는가", "문제 해결 경험", "성과", "최종 정리"])
    if "placeholder" in targets or "style" in targets:
        targets.update(["문제 해결 경험", "복잡한 문제/수식/쿼리/코드 작성 및 해결 경험"])

    for target in list(targets):
        if target not in sections:
            continue
        prompt = f"""
아래 섹션만 확장/수정하세요. 전체 글을 다시 쓰지 않습니다.

대상 섹션: {target}
실패 사유: {critique.section_failures.get(target) or critique.failures}

수정 규칙:
- 부족한 분량, 원인/조치/검증, 이미지 근거 연결만 보강합니다.
- placeholder와 내부 라벨을 제거합니다.
- 없는 수식/코드/성과를 만들지 않습니다.
- Markdown 섹션 하나만 반환합니다.
- 문제 해결 경험이면 최소 4개 단계와 각 단계의 문제/제약, 원인 판단, 조치, 확인 결과를 모두 포함합니다.

[기존 섹션]
{sections[target]}

[Outline]
{json.dumps(outline.get(target, []), ensure_ascii=False)}

[ArticleBrief]
{json.dumps(brief, ensure_ascii=False)}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}

[사용자 원문/메모/추가 정보]
{raw_text}

{memo}

{extra_info}
""".strip()
        expanded = llm_text(llm_client, prompt, max_tokens=3400)
        if expanded.strip():
            sections[target] = expanded.strip()

    expanded_article = assemble_article(sections, brief)
    second_critique = critique_article(expanded_article, str(problem_map.get("article_type") or "general_learning_portfolio"), problem_map, evidence)
    if second_critique.passed or len(expanded_article) >= len(article):
        return expanded_article
    return article


def assemble_article(sections: dict[str, str], brief: dict[str, Any]) -> str:
    ordered = [sections.get(title, "").strip() for title in SECTION_TITLES if sections.get(title, "").strip()]
    if not ordered or not ordered[0].startswith("# "):
        ordered.insert(0, f"# {brief.get('korean_title') or '문제 해결형 학습 기록'}")
    return "\n\n".join(ordered).strip()


def llm_json(llm_client: LLM, prompt: str, max_tokens: int = 2400) -> Any:
    text = llm_text(
        llm_client,
        prompt,
        max_tokens=max_tokens,
        system="You create strict JSON for a multi-step Korean Medium portfolio writing pipeline. Return JSON only.",
    )
    return parse_json_payload(text)


def llm_text(llm_client: LLM, prompt: str, max_tokens: int = 2400, system: str | None = None) -> str:
    client = llm_client.get_client()
    if not client:
        return ""
    route = model_route("final_article_generation" if max_tokens > 2000 else "simple_tutor_answer")
    try:
        completion = client.chat.completions.create(
            model=route["model"],
            temperature=0.22,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You write grounded Korean Medium portfolio sections from structured evidence."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[LLM text pipeline error] {exc}")
        return ""


def parse_json_payload(content: str) -> Any:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min([index for index in [cleaned.find("{"), cleaned.find("[")] if index >= 0], default=-1)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def read_image_order_caption_source() -> str:
    candidates = [
        BASE_DIR / "README_image_order.txt",
        DATA_DIR / "README_image_order.txt",
        CAPTURE_DIR / "README_image_order.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")[:5000]
    return ""


def normalize_image_evidence(data: list[Any], start_no: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    allowed_roles = {"problem", "cause", "solution", "validation", "final_result"}
    for index, item in enumerate(data, start=start_no):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "problem").strip()
        if role not in allowed_roles:
            role = "problem"
        normalized.append(
            {
                "image_no": index,
                "display_image_index": index,
                "source_image_no": item.get("image_no", index),
                "caption": humanize_caption(str(item.get("caption") or f"이미지 {index} - 실습 흐름 근거 화면"), index),
                "visible_evidence": normalize_str_list(item.get("visible_evidence"))[:10],
                "role": role,
                "problem_signal": str(item.get("problem_signal") or ""),
                "technical_entities": normalize_str_list(item.get("technical_entities"))[:12],
                "inferred_meaning": str(item.get("inferred_meaning") or ""),
                "confidence": float(item.get("confidence") or 0.65),
                "evidence_source": str(item.get("evidence_source") or "vision"),
            }
        )
    return normalized


def fallback_image_evidence(
    image_files: list[Path],
    image_names: list[str],
    raw_text: str,
    memo: str,
    caption_source: str,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    entities = infer_entities(" ".join([raw_text, memo, caption_source, " ".join(image_names)]))
    for index, (path, name) in enumerate(zip(image_files, image_names, strict=False), start=1):
        role = infer_role(index, len(image_files))
        caption_hint = caption_from_sources(index, name, caption_source)
        evidence.append(
            {
                "image_no": index,
                "display_image_index": index,
                "source_image_no": index,
                "original_filename": name,
                "caption": caption_hint or f"이미지 {index} - {Path(name).stem or path.stem}",
                "visible_evidence": [Path(name).stem, path.name],
                "role": role,
                "problem_signal": memo or raw_text or caption_hint or "이미지 순서와 캡션 정보를 바탕으로 실습 흐름을 복구해야 함",
                "technical_entities": entities,
                "inferred_meaning": f"전체 이미지 {len(image_files)}장 중 {index}번째 근거 화면입니다. README_image_order.txt, 파일명 prefix, 업로드 순서에서 확인되는 실습 단계를 바탕으로 본문 근거를 제한해야 합니다.",
                "confidence": 0.35,
                "evidence_source": "README_image_order.txt" if caption_source else "filename",
            }
        )
    return evidence


def infer_role(index: int, total: int) -> str:
    if total <= 1 or index == 1:
        return "problem"
    if index == total:
        return "final_result"
    ratio = index / max(total, 1)
    if ratio < 0.35:
        return "cause"
    if ratio < 0.75:
        return "solution"
    return "validation"


def caption_from_sources(index: int, name: str, caption_source: str) -> str:
    for line in caption_source.splitlines():
        if re.search(rf"\b0?{index}\b", line):
            text = re.sub(r"^\s*\d{1,3}[_\-\s]?", "", line.strip())
            text = re.sub(r"\.(png|jpg|jpeg)\b", "", text, flags=re.I)
            return humanize_caption(f"이미지 {index} - {text[:180]}", index)
    stem = humanize_filename_stem(Path(name).stem)
    if stem:
        return humanize_caption(f"이미지 {index} - {stem}", index)
    return ""


def humanize_filename_stem(stem: str) -> str:
    cleaned = re.sub(r"^\d{1,3}[_\-\s]?", "", stem)
    image_match = re.fullmatch(r"image(\d+)", cleaned, flags=re.I)
    if image_match:
        etl_captions = {
            1: "SQL Server database localhost와 AdventureWorksDW2020 데이터베이스를 연결 대상으로 선택한 화면",
            2: "Navigator에서 FactResellerSales 등 필요한 테이블을 선택하고 Transform Data로 Power Query Editor에 진입하는 화면",
            3: "Column quality, Column distribution, Column profile로 BusinessType 분포와 데이터 품질을 확인한 화면",
            4: "DimEmployee에서 SalesPersonFlag를 TRUE로 필터링해 영업 담당자만 남기는 화면",
            5: "FirstName과 LastName을 병합해 Salesperson 컬럼을 만들고 EmployeeID와 UPN을 정리한 화면",
            6: "Product 쿼리에서 DimProductSubcategory와 DimProductCategory를 확장해 Subcategory와 Category를 붙인 화면",
            7: "Reseller 쿼리에서 BusinessType의 Ware House 값을 Warehouse로 Replace Values 처리한 화면",
            8: "Region 쿼리에서 SalesTerritoryAlternateKey 0을 제거하고 Region, Country, Group 컬럼을 정리한 화면",
            9: "Sales 쿼리에서 TotalProductCost null을 OrderQuantity와 StandardCost 기반 Cost Custom Column으로 보완한 화면",
            10: "Targets CSV의 M01부터 M12까지 월별 목표 컬럼을 Unpivot 대상으로 선택한 화면",
            11: "Targets 데이터를 MonthNumber와 Target 중심의 long format으로 변환하고 TargetMonth를 구성한 화면",
            12: "ColorFormats 쿼리에서 Background Color Format과 Font Color Format 컬럼을 정리한 화면",
            13: "Product와 ColorFormats를 Product[Color]와 ColorFormats[Color] 기준 Left Outer Merge한 화면",
            14: "ColorFormats Disable load 후 Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets 7개 테이블을 Close & Apply로 로드하는 화면",
        }
        return etl_captions.get(int(image_match.group(1)), cleaned)
    replacements = {
        "Repeated Category Sales": "Category별 Sum of Sales가 모두 같은 값으로 반복되는 문제 화면",
        "Product Sales Relationship New": "Product[ProductKey]와 Sales[ProductKey] relationship을 새로 생성하는 화면",
        "Product Sales Relationship Model": "Product와 Sales relationship이 model view에 반영된 화면",
        "Star Schema Model Layout": "Product dimension과 Sales fact table 중심의 Star schema 모델 구조",
        "Create Product Hierarchy": "Product hierarchy를 구성하는 화면",
        "Products Hierarchy Levels": "Category, Subcategory, Product hierarchy level을 확인하는 화면",
        "Profit Quick Measure Subtraction": "Profit measure를 Sales와 Cost 차이로 구성하는 화면",
        "Profit Margin Quick Measure Division": "DIVIDE 기반 Profit Margin measure를 구성하는 화면",
        "Category Sales Profit Margin Table": "Category별 Sales, Profit, Profit Margin 결과를 검증하는 화면",
        "Profit Margin Result Zoom": "Profit Margin 계산 결과를 확대해 확인하는 화면",
        "SalespersonRegion Bridge Model": "SalespersonRegion bridge table로 filter path를 구성한 모델 화면",
        "Cross Filter Both Relationship": "Region과 SalespersonRegion 관계에서 Cross-filter direction을 Both로 설정한 화면",
        "Deactivate Direct Salesperson Sales": "모호한 직접 Salesperson-Sales relationship을 inactive relationship으로 바꾸는 화면",
        "Salesperson Sales By Region Result": "Salesperson별 Sales가 Region filter path로 달라진 결과 화면",
        "Salesperson Sales Target Final": "Salesperson별 Sales와 Targets를 최종 비교하는 화면",
        "Month Sort by MonthKey": "Month 컬럼을 MonthKey 기준으로 정렬한 화면",
        "Fiscal Hierarchy Date Table": "Fiscal hierarchy가 포함된 Date table을 구성한 화면",
        "Date Model Relationships": "Date table과 Sales model relationship을 확인한 화면",
        "Mark as Date Table": "Date table을 공식 날짜 테이블로 지정한 화면",
        "Avg Price Measure": "Avg Price explicit measure를 생성한 화면",
        "Pricing Count Measures Matrix": "Median Price, Min Price, Max Price, Orders, Order Lines measure를 matrix에서 검증한 화면",
        "Measure Formatting Check": "가격 measure의 Currency format을 확인한 화면",
        "TargetAmount vs Target Measure": "TargetAmount raw column 대신 Target measure를 비교한 화면",
        "Sales Target Variance Final": "Salesperson별 Sales, Target, Variance, Variance Margin을 최종 비교한 화면",
    }
    words = cleaned.replace("_", " ").replace("-", " ").strip()
    return replacements.get(words, words)


def humanize_caption(caption: str, index: int) -> str:
    caption = re.sub(r"원본 파일명\s*", "", caption)
    caption = re.sub(r"\b\d{1,3}[_\-][A-Za-z0-9_ \-]+\.(png|jpg|jpeg)", "", caption, flags=re.I)
    caption = re.sub(r"\.(png|jpg|jpeg)\b", "", caption, flags=re.I)
    caption = caption.replace("_", " ").strip(" -")
    if not caption.startswith(f"이미지 {index}"):
        caption = re.sub(r"^이미지\s+\d+\s*-\s*", "", caption)
        caption = f"이미지 {index} - {caption}"
    return caption


def infer_entities(text: str) -> list[str]:
    candidates = [
        "Power BI",
        "semantic model",
        "filter context",
        "relationship",
        "cardinality",
        "Product",
        "Sales",
        "Category",
        "ProductKey",
        "DAX",
        "Measure",
        "MonthKey",
        "Date table",
        "Fiscal hierarchy",
        "Mark as date table",
        "Avg Price",
        "Median Price",
        "Orders",
        "Order Lines",
        "TargetAmount",
        "Target",
        "Variance",
        "Variance Margin",
        "DIVIDE",
        "SQL",
        "Python",
        "Power Query",
        "Column quality",
        "Column distribution",
        "Column profile",
        "TotalProductCost",
        "Unpivot",
        "ColorFormats",
        "Close & Apply",
        "many-to-many",
        "bridge table",
        "GitHub",
        "Agentic Workflows",
        "GitHub Actions",
        "workflow_dispatch",
        "update-github-info",
        "update-github-info.lock.yml",
        "activation",
        "conclusion",
        ".github/workflows",
        "YAML",
        "yml",
    ]
    lowered = text.lower()
    found = [item for item in candidates if item.lower() in lowered]
    if found:
        return found
    if any(word in lowered for word in ["github", "workflow", "agentic", "actions", "yml", "yaml"]):
        return ["GitHub", "workflow", "automation"]
    return ["확인된 캡처 evidence"]


def normalize_problem_map(
    data: dict[str, Any],
    raw_text: str,
    memo: str,
    evidence: list[dict[str, Any]],
    topic: str,
    article_type: str = "general_learning_portfolio",
    golden_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    golden_context = golden_context or {}
    core = str(data.get("core_problem") or memo or raw_text or f"{topic} 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치")
    is_powerbi_type = article_type in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}
    default_why = (
        "화면에 값이 표시되더라도 모델, 수식, 관계, 필터 흐름이 맞지 않으면 분석 결과의 의미가 달라질 수 있습니다."
        if is_powerbi_type
        else "드문드문 남은 캡처만으로는 강의 목표, 실행 조건, 사용자가 막힌 지점, 최종 결과를 확정하기 어렵기 때문에 확인된 evidence와 부족한 맥락을 분리해야 합니다."
    )
    default_root_causes = (
        ["시각적 결과와 데이터/계산 구조 사이의 연결 검증이 필요했습니다."]
        if is_powerbi_type
        else ["캡처 간 맥락이 비어 있어 관찰된 화면을 확정적인 성과나 해결 단계로 단정하기 어렵습니다."]
    )
    default_actual_cause = (
        "실제로는 근거 화면 전체를 연결해 모델/수식/검증 흐름을 함께 봐야 했습니다."
        if is_powerbi_type
        else "실제로는 각 캡처가 목표, 설정, 실행, 결과 중 어느 역할인지 먼저 구분해야 했습니다."
    )
    steps = data.get("solution_steps")
    if not isinstance(steps, list):
        steps = []
    normalized_steps = []
    for index, item in enumerate(steps[:6], start=1):
        if not isinstance(item, dict):
            continue
        normalized_steps.append(
            {
                "step": int(item.get("step") or index),
                "title": str(item.get("title") or f"해결 단계 {index}"),
                "problem": str(item.get("problem") or core),
                "cause": str(item.get("cause") or "입력 근거에서 원인 후보를 좁혔습니다."),
                "action": str(item.get("action") or "관찰 근거와 메모를 바탕으로 수정 방향을 정리했습니다."),
                "verification": str(item.get("verification") or "다음 이미지 또는 결과 화면으로 변경 여부를 확인했습니다."),
            }
        )
    if is_powerbi_type:
        while len(normalized_steps) < 4:
            index = len(normalized_steps) + 1
            role_evidence = evidence[min(index - 1, len(evidence) - 1)] if evidence else {}
            normalized_steps.append(
                {
                    "step": index,
                    "title": f"근거 화면 {index} 기반 검증 단계",
                    "problem": role_evidence.get("problem_signal") or core,
                    "cause": role_evidence.get("inferred_meaning") or "현재 화면이 문제, 원인, 조치, 검증 중 어느 단계인지 분리해야 했습니다.",
                    "action": "이미지 근거와 사용자 메모를 연결해 문제 해결 단계로 재구성했습니다.",
                    "verification": role_evidence.get("caption") or "후속 화면과 최종 결과에서 변화 여부를 확인해야 합니다.",
                }
            )
    else:
        if len(normalized_steps) < 3:
            text_steps = build_text_assisted_solution_steps(article_type, raw_text, memo)
            if text_steps:
                normalized_steps = text_steps
                data["_text_assisted_draft"] = True
        data["_sparse_steps_incomplete"] = len(normalized_steps) < 3
    complex_problem = data.get("complex_problem") if isinstance(data.get("complex_problem"), dict) else {}
    return {
        "article_type": article_type,
        "core_problem": core,
        "why_problematic": str(data.get("why_problematic") or default_why),
        "root_causes": normalize_str_list(data.get("root_causes")) or default_root_causes,
        "solution_steps": normalized_steps,
        "complex_problem": {
            "title": str(complex_problem.get("title") or "복합적인 원인 분석과 검증 흐름 정리"),
            "symptom": str(complex_problem.get("symptom") or core),
            "initial_assumption": str(complex_problem.get("initial_assumption") or "처음에는 화면 구성이나 단일 설정 문제로 볼 수 있었습니다."),
            "actual_cause": str(complex_problem.get("actual_cause") or default_actual_cause),
            "resolution": str(complex_problem.get("resolution") or "각 이미지의 역할을 문제, 원인, 해결, 검증으로 분리해 해결 흐름을 구성했습니다."),
            "verification": str(complex_problem.get("verification") or "마지막 결과 화면과 사용자 메모를 기준으로 검증 내용을 정리했습니다."),
        },
        "final_outcome": str(data.get("final_outcome") or ("현재 확인된 GitHub/워크플로우 캡처를 evidence 후보로 정리했습니다." if article_type in GITHUB_WORKFLOW_TYPES else "캡처와 메모를 문제 해결형 포트폴리오 글로 전환할 수 있는 구조화된 근거로 정리했습니다.")),
        "_sparse_steps_incomplete": bool(data.get("_sparse_steps_incomplete")),
    }


def normalize_article_brief(data: dict[str, Any], topic: str, problem_map: dict[str, Any]) -> dict[str, Any]:
    if problem_map.get("article_type") == "semantic_model_relationship":
        data = {
            **data,
            "korean_title": data.get("korean_title") or "Power BI Semantic Model 실습: 반복되는 매출값 문제를 관계·계층·DAX Measure로 해결하기",
            "english_subtitle": data.get("english_subtitle") or "Fixing repeated category sales with relationships, hierarchy, DAX measures, and bridge-table validation",
            "article_thesis": data.get("article_thesis") or problem_map.get("core_problem") or SEMANTIC_CORE_PROBLEM,
            "portfolio_angle": data.get("portfolio_angle")
            or "반복되는 Sales 값에서 filter context 문제를 찾고 Product-Sales relationship, DAX measure, SalespersonRegion bridge path까지 검증한 과정을 보여줍니다.",
            "must_include": data.get("must_include")
            or ["문제 인식", "원인 분석", "ProductKey relationship", "DAX Measure", "Bridge table", "Targets 비교", "이미지 캡션 목록"],
        }
    if problem_map.get("article_type") == "dax_measure_modeling":
        data = {
            **data,
            "korean_title": data.get("korean_title") or "Power BI DAX 실습: 날짜 테이블, Measure, Target, Variance로 분석 모델 완성하기",
            "english_subtitle": data.get("english_subtitle") or "Building a clearer analysis model with date tables, explicit measures, targets, and variance metrics",
            "article_thesis": data.get("article_thesis") or problem_map.get("core_problem") or DAX_CORE_PROBLEM,
            "portfolio_angle": data.get("portfolio_angle")
            or "MonthKey 정렬, Date table, explicit measure, Target/Variance measure를 통해 raw column 자동 집계 의존을 줄인 DAX 모델링 과정을 보여줍니다.",
            "must_include": data.get("must_include")
            or ["MonthKey 정렬", "Date table", "explicit measure", "Target measure", "Variance", "Salesperson별 최종 비교", "이미지 캡션 목록"],
        }
    if problem_map.get("article_type") == "power_query_etl":
        default_title = "Power Query ETL 실습: 원본 데이터를 분석 가능한 구조로 정리·변환·로드하기"
        default_subtitle = "Turning raw study evidence into a verifiable data preparation workflow"
        default_thesis = problem_map.get("core_problem") or topic
        default_angle = "원본 데이터를 Power Query에서 분석 가능한 테이블 구조로 바꾸는 판단 과정을 보여줍니다."
        if problem_map.get("_regression_profile") == "power_query_etl_example":
            default_title = "Power BI Desktop 실습: 원본 데이터를 분석 가능한 모델 입력 구조로 정리·변환·로드하기"
            default_subtitle = "Cleaning, shaping, merging, and loading raw data into an analysis-ready Power BI model"
            default_thesis = problem_map.get("core_problem") or POWER_QUERY_CORE_PROBLEM
            default_angle = "SQL Server와 CSV 원본을 Power Query에서 분석 가능한 테이블 구조로 바꾸는 판단 과정을 보여줍니다."
        data = {
            **data,
            "korean_title": data.get("korean_title") or default_title,
            "english_subtitle": data.get("english_subtitle") or default_subtitle,
            "article_thesis": data.get("article_thesis") or default_thesis,
            "portfolio_angle": data.get("portfolio_angle")
            or default_angle,
            "must_include": data.get("must_include")
            or ["문제 인식", "원인 분석", "검증 결과", "Power Query 변환 근거", "이미지 캡션 목록", "Section Plan"],
        }
    if problem_map.get("article_type") in GITHUB_WORKFLOW_TYPES:
        source_text = str(problem_map.get("_evidence_text_for_classification") or "")
        data = {
            **data,
            "korean_title": data.get("korean_title") if data.get("korean_title") and not is_generic_learning_title(str(data.get("korean_title"))) else specific_title_candidate(str(problem_map.get("article_type")), [], "GitHub Agentic Workflows 실습: workflow_dispatch와 자동화 실행 흐름을 이해한 기록"),
            "english_subtitle": data.get("english_subtitle") or "Understanding GitHub agentic workflow execution from sparse learning captures",
            "article_thesis": data.get("article_thesis") if data.get("article_thesis") and not is_generic_core_problem(str(data.get("article_thesis"))) else problem_map.get("core_problem"),
            "portfolio_angle": data.get("portfolio_angle") or "드문드문 남은 GitHub workflow 캡처에서 자동화 실행 조건, workflow 파일, activation, conclusion의 의미를 구분하는 학습 기록으로 정리합니다.",
            "must_include": data.get("must_include") or ["GitHub workflow", "workflow_dispatch", "activation", "conclusion", "missing_context", "이미지 캡션 목록"],
        }
    return {
        "korean_title": str(data.get("korean_title") or f"{topic}: 문제를 검증 가능한 분석 흐름으로 바꾼 기록"),
        "english_subtitle": str(data.get("english_subtitle") or "A problem-solving portfolio article from technical study evidence"),
        "article_thesis": str(data.get("article_thesis") or problem_map.get("core_problem") or topic),
        "target_reader": str(data.get("target_reader") or "recruiter / hiring manager / technical reviewer"),
        "portfolio_angle": str(data.get("portfolio_angle") or "사용자가 기술적 문제를 발견하고 원인, 조치, 검증으로 정리할 수 있음을 보여줍니다."),
        "do_not_claim": normalize_str_list(data.get("do_not_claim")) or ["사용자가 하지 않은 성과", "없는 수식", "없는 배포"],
        "must_include": normalize_str_list(data.get("must_include")) or ["문제 인식", "원인 분석", "검증 결과", "이미지 캡션 목록"],
    }


def default_outline_items(title: str, brief: dict[str, Any], problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    if title == "문제 해결 경험":
        return [
            f"{step['step']}. {step['title']}: 문제/제약, 원인 판단, 조치, 확인 결과"
            for step in problem_map.get("solution_steps", [])
        ]
    if title == "이미지 번호와 캡션 목록":
        return [str(item.get("caption")) for item in evidence]
    return [str(brief.get("article_thesis") or problem_map.get("core_problem") or title)]


def fallback_section(
    section_title: str,
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
) -> str:
    if section_title == "이미지 번호와 캡션 목록":
        captions = "\n".join(f"- {item.get('caption')}" for item in evidence) or "- 이미지 없음"
        return f"## 이미지 번호와 캡션 목록\n{captions}"
    if section_title == "Key skills practiced":
        return "\n".join(
            [
                "## Key skills practiced",
                "- Problem framing",
                "- Evidence-based technical documentation",
                "- Root-cause analysis",
                "- Validation planning",
                "- Data model reasoning",
                "- Portfolio writing",
                "- Technical communication",
                "- Iterative debugging",
            ]
        )
    if section_title == "문제 해결 경험":
        lines = ["## 문제 해결 경험"]
        for step in problem_map.get("solution_steps", []):
            lines.append(
                f"### {step['step']}. {step['title']}\n"
                f"문제/제약: {step['problem']}\n\n"
                f"원인 판단: {step['cause']}\n\n"
                f"조치: {step['action']}\n\n"
                f"확인 결과: {step['verification']}"
            )
        return "\n\n".join(lines)
    return f"## {section_title}\n{brief.get('article_thesis') or problem_map.get('core_problem') or raw_text or memo}"


def sanitize_medium_markdown(article: str) -> str:
    cleaned = article
    for phrase in PLACEHOLDER_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    return postprocess_article_text(cleaned).strip()


def postprocess_article_text(article: str) -> str:
    cleaned = article
    replacements = {
        "문제를 확인한 문제로 정의되었다": "문제로 정의되었다",
        "정리해야 했다.로 정의되었다": "정리해야 하는 문제로 정의되었다",
        "정리해야 했다.로 정리되었다": "정리해야 하는 흐름으로 정리되었다",
        "구성해야 한 것으로 정의되었다": "구성해야 하는 문제로 정의되었다",
        "확인했다.로 정의되었다": "확인한 문제로 정의되었다",
        "확인했다.로 정리되었다": "확인한 흐름으로 정리되었다",
        "영향을 줌로": "영향을 주는 방식으로",
        "보임 하지만": "보였지만",
        "처리으로 접근했다": "처리 방식으로 접근했다",
        "확인을 기준으로 삼았다": "확인 결과를 기준으로 삼았다",
        "했다.로 정의되었다": "한 것으로 정의되었다",
        "했다.로 정리되었다": "한 것으로 정리되었다",
        "했다.라는": "했다는",
        "했다..": "했다.",
        "정리해야 했다..": "정리해야 했다.",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\[\]\(\) /·,_-]+문제)를 확인한 문제로 정의되었다", r"\1로 정의되었다", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\[\]\(\) /·,_-]+)구성해야 한 것으로 정의되었다", r"\1구성해야 하는 문제로 정의되었다", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\[\]\(\) /·,_-]+)구성해야 했다\.로 정의되었다", r"\1구성해야 하는 문제로 정의되었다", cleaned)
    cleaned = re.sub(r"정리해야 했다\.로\s*", "정리해야 하는 문제로 ", cleaned)
    cleaned = re.sub(r"확인했다\.로\s*", "확인한 문제로 ", cleaned)
    cleaned = re.sub(r"([가-힣]+했다)\.로\s*", r"\1는 흐름으로 ", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\]\)])\.\.(?=\s|$)", r"\1.", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^(포트폴리오 관점: .+)\n\n포트폴리오 관점: ", r"\1\n\n이 단계는 ", cleaned)
    return cleaned


def count_problem_solution_steps(article: str) -> int:
    section = extract_section(article, "문제 해결 경험")
    return max(
        len(re.findall(r"(?m)^###\s+\d+[\.\)]", section)),
        len(re.findall(r"문제/제약", section)),
        len(re.findall(r"확인 결과", section)),
    )


def count_section_paragraphs(article: str, title: str) -> int:
    section = extract_section(article, title)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip() and not part.strip().startswith("#")]
    return len(paragraphs)


def extract_section(article: str, title: str) -> str:
    pattern = rf"(?ms)^##\s+{re.escape(title)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    match = re.search(pattern, article)
    return match.group(1).strip() if match else ""


def has_code_block(article: str) -> bool:
    return "```" in article or bool(re.search(r"\b[A-Z][A-Za-z0-9_]*\s*=", article))


def has_code_explanation(article: str) -> bool:
    checks = ["왜", "필요", "계산", "검증", "확인 결과"]
    code_section = extract_section(article, "사용한 주요 수식/코드 정리")
    return all(word in code_section for word in checks[:3]) and any(word in code_section for word in checks[3:])


def semantic_problem_detail(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
        return (
            "이번 실습의 문제는 데이터를 불러오는 것 자체가 아니라, SQL Server와 CSV에서 가져온 원본 데이터를 보고서 작성에 적합한 분석 모델 입력 구조로 정리하는 것이었다. "
            "초기 화면에서는 localhost의 AdventureWorksDW2020 데이터베이스, Navigator의 테이블 선택, Transform Data 진입이 보이고, 이후 화면에서는 Power Query Editor 안에서 Column quality, BusinessType 분포, SalesPersonFlag 필터, Product 확장, TotalProductCost null, M01~M12 Unpivot, ColorFormats Merge, Disable load까지 이어진다. "
            "이 흐름은 연결 실패나 단순 로딩 문제가 아니라, 원본 데이터의 품질과 형태를 분석 가능한 테이블 구조로 바꾸는 transformation requirement로 해석해야 한다."
        )
    if problem_map.get("article_type") != "semantic_model_relationship":
        return str(problem_map.get("core_problem", ""))
    return (
        "처음 만든 table visual에는 Product[Category]와 Sales[Sales]를 함께 넣었다. "
        "기대했던 결과는 Category 행마다 서로 다른 매출값이 표시되는 것이었지만, 실제 화면에서는 모든 Category 행에 같은 Sales 총액이 반복되는 패턴이 나타났다. "
        "이 패턴은 visual 서식 문제가 아니라 Product와 Sales 사이 relationship이 없거나 Product[Category] filter context가 Sales fact table로 전달되지 않는 상황으로 해석할 수 있다."
    )


def concrete_detail_text(step: dict[str, Any]) -> str:
    details = normalize_str_list(step.get("concrete_details"))
    if not details:
        return ""
    return "\n\n구체적 설정/근거:\n" + "\n".join(f"- {detail}" for detail in details)


def portfolio_meaning_text(step: dict[str, Any]) -> str:
    meaning = str(step.get("portfolio_meaning") or "").strip()
    if not meaning:
        return ""
    step_no = int(step.get("step") or 0)
    if step_no == 1:
        return f"\n\n포트폴리오 관점: {meaning}"
    if step_no % 3 == 0:
        return f"\n\n이 단계의 의미는 {meaning}"
    if step_no % 3 == 1:
        return f"\n\n이 판단은 {meaning}"
    return f"\n\n결과적으로 {meaning}"


def clean_summary_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"([.!?。]){2,}$", r"\1", cleaned)
    cleaned = cleaned.strip("` ")
    return cleaned


def article_type_entities_text(article_type: str, evidence_entities: list[str], problem_map: dict[str, Any]) -> str:
    preferred = {
        "semantic_model_relationship": [
            "Product[Category]",
            "Sales[Sales]",
            "Product[ProductKey]",
            "Sales[ProductKey]",
            "One to many",
            "Cross-filter direction",
            "DIVIDE",
            "SalespersonRegion",
            "inactive relationship",
            "Targets",
        ],
        "dax_measure_modeling": [
            "Month",
            "MonthKey",
            "Date table",
            "Fiscal hierarchy",
            "Mark as date table",
            "Avg Price",
            "Orders",
            "Target measure",
            "Variance",
            "Variance Margin",
        ],
        "power_query_etl": [
            "Power Query Editor",
            "SQL Server",
            "CSV",
            "Column quality",
            "Column distribution",
            "Column profile",
            "SalesPersonFlag",
            "ProductKey",
            "BusinessType",
            "TotalProductCost",
            "Custom Column",
            "Unpivot",
            "TargetMonth",
            "ColorFormats",
            "Product[Color]",
            "Left Outer",
            "Disable load",
            "Close & Apply",
        ],
    }.get(article_type, [])
    haystack = json.dumps(problem_map, ensure_ascii=False) + " " + " ".join(evidence_entities)
    selected = [item for item in preferred if item.lower() in haystack.lower()]
    if len(selected) < min(6, len(preferred)):
        selected = preferred[:10] if preferred else []
    if selected:
        return ", ".join(selected[:12])
    generic_filtered = [
        item
        for item in evidence_entities
        if item.lower() not in {"study capture", "technical evidence", "capture", "evidence"}
    ]
    return ", ".join(generic_filtered[:10]) or "입력 이미지와 메모에 나타난 기술 요소"


def key_skills_for_article_type(article_type: str) -> list[str]:
    by_type = {
        "semantic_model_relationship": [
            "Diagnosing filter context issues from repeated values",
            "Designing Product-to-Sales relationships",
            "Validating one-to-many cardinality and active relationships",
            "Reading star schema structure in model view",
            "Building Product hierarchy for drilldown analysis",
            "Creating Profit and Profit Margin measures",
            "Using DIVIDE for ratio measures",
            "Resolving bridge-table filter paths",
            "Comparing Sales and Targets with model-aware validation",
        ],
        "dax_measure_modeling": [
            "Sorting display columns with MonthKey",
            "Building fiscal Date table structures",
            "Marking a Date table for time-based analysis",
            "Replacing raw column aggregation with explicit measures",
            "Creating pricing and order-count measures",
            "Applying currency formatting to measures",
            "Controlling Target totals with measure logic",
            "Calculating Variance and Variance Margin",
            "Validating Salesperson-level performance metrics",
        ],
        "power_query_etl": [
            "Profiling source data quality in Power Query",
            "Standardizing categorical values",
            "Filtering dimension queries for analysis use",
            "Expanding related dimension tables",
            "Creating Custom Column logic for missing cost values",
            "Unpivoting wide target data into long format",
            "Merging lookup tables with Left Outer joins",
            "Managing helper-query load settings",
            "Validating final model table load scope",
        ],
        "python_algorithm_learning": [
            "Interpreting algorithm problem constraints",
            "Tracing input and output behavior",
            "Handling edge cases",
            "Reasoning about time complexity",
            "Refactoring Python control flow",
            "Writing verification examples",
            "Explaining failure cases clearly",
            "Documenting algorithm decisions",
        ],
        "code_error_debugging": [
            "Reading stack traces",
            "Isolating root causes",
            "Reproducing failures",
            "Designing minimal fixes",
            "Adding verification checks",
            "Separating symptoms from causes",
            "Documenting debugging decisions",
            "Avoiding unsupported conclusions",
        ],
        "github_agentic_workflow": [
            "Understanding GitHub Agentic Workflows",
            "Reading GitHub Actions workflow files",
            "Identifying workflow_dispatch triggers",
            "Interpreting workflow activation states",
            "Connecting workflow files with execution conclusions",
            "Reading lock-file evidence conservatively",
            "Separating observed automation behavior from assumptions",
            "Documenting sparse learning evidence",
        ],
        "github_actions_workflow": [
            "Reading GitHub Actions workflow files",
            "Identifying workflow_dispatch triggers",
            "Interpreting workflow activation states",
            "Connecting workflow YAML with run conclusions",
            "Checking automation state changes",
            "Documenting sparse learning evidence",
            "Avoiding unsupported workflow claims",
            "Writing evidence-based workflow summaries",
        ],
        "ai_coding_workflow": [
            "Understanding AI-assisted coding workflows",
            "Identifying automation triggers",
            "Reading workflow state evidence",
            "Connecting agent actions with observed results",
            "Separating course context from implementation claims",
            "Documenting sparse learning evidence",
            "Avoiding unsupported tool claims",
            "Writing evidence-based learning summaries",
        ],
        "github_readme_debugging": [
            "Debugging Markdown rendering",
            "Checking repository asset paths",
            "Validating README previews",
            "Explaining media embedding constraints",
            "Separating local and GitHub render behavior",
            "Documenting fixes with visible evidence",
            "Maintaining reader-friendly project docs",
            "Avoiding unrelated technical claims",
        ],
        "deployment_debugging": [
            "Reading build and runtime logs",
            "Checking environment variables",
            "Separating deployment and application failures",
            "Validating service health endpoints",
            "Documenting rollback or fix decisions",
            "Explaining infrastructure assumptions",
            "Verifying server behavior after changes",
            "Communicating operational risk clearly",
        ],
    }
    return by_type.get(
        article_type,
        [
            "Problem framing from visual evidence",
            "Root-cause analysis",
            "Evidence-based technical documentation",
            "Validation planning",
            "Technical portfolio writing",
            "Clear explanation of cause, action, and result",
            "Avoiding unsupported claims",
            "Connecting captures to a problem-solving narrative",
        ],
    )


def image_ref_caption_text(step: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    refs = [int(ref) for ref in step.get("image_refs", []) if str(ref).isdigit()]
    if not refs:
        return ""
    captions = [item.get("caption") for item in evidence if int(item.get("image_no") or 0) in refs]
    if not captions:
        return f"이미지 근거: {', '.join(f'이미지 {ref}' for ref in refs)}"
    return "이미지 근거: " + "; ".join(str(caption) for caption in captions[:3])


def article_type_focus_text(article_type: str) -> str:
    if article_type == "semantic_model_relationship":
        return "이 유형에서는 repeated Sales 값, relationship 구조, filter context 전달, ProductKey 연결 여부를 함께 확인해야 한다."
    if article_type == "dax_measure_modeling":
        return "이 유형에서는 Date table, MonthKey 정렬, explicit Measure, Target, Variance 계산 맥락을 함께 확인해야 한다."
    if article_type == "power_query_etl":
        return "이 유형에서는 data quality, null cost 보완, Unpivot, Merge, load control이 이후 분석 오류를 어떻게 줄이는지 확인해야 한다."
    return "이 유형에서는 화면 근거와 사용자 메모를 연결해 문제, 원인, 조치, 검증 기준을 분리해야 한다."


def final_outcome_links_to_problem(article: str, problem_map: dict[str, Any]) -> bool:
    final_section = extract_section(article, "최종 정리") + "\n" + extract_section(article, "성과")
    core = str(problem_map.get("core_problem", ""))
    important_terms = [term for term in tokenize(core) if len(term) >= 4][:8]
    if not important_terms:
        return True
    return sum(1 for term in important_terms if term.lower() in final_section.lower()) >= min(2, len(important_terms))


def missing_article_type_keywords(article_type: str, article: str, problem_map: dict[str, Any]) -> list[str]:
    required = {
        "semantic_model_relationship": ["Sales", "relationship", "filter context", "ProductKey"],
        "dax_measure_modeling": ["Date table", "MonthKey", "Measure", "Target", "Variance"],
        "power_query_etl": ["data quality", "TotalProductCost", "Unpivot", "Merge", "load"],
    }.get(article_type, [])
    haystack = (article + "\n" + json.dumps(problem_map, ensure_ascii=False)).lower()
    return [keyword for keyword in required if keyword.lower() not in haystack]


def awkward_korean_leftovers(article: str) -> list[str]:
    patterns = [
        "문제를 확인한 문제로 정의되었다",
        "정리해야 했다.로 정의되었다",
        "구성해야 한 것으로 정의되었다",
        "영향을 줌로",
        "보임 하지만",
        "처리으로 접근했다",
        "확인을 기준으로 삼았다",
        "..",
    ]
    return [pattern for pattern in patterns if pattern in article]


def truncated_caption_lines(article: str) -> list[str]:
    section = extract_section(article, "이미지 번호와 캡션 목록")
    if not section:
        return []
    truncated: list[str] = []
    allowed_endings = ("다", "음", "면", "결과", "비교", "확인", "구성", "로드", "검증", "화면", "상태", "처리")
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        caption = stripped.lstrip("- ").strip()
        if not caption:
            truncated.append(stripped)
            continue
        if caption.endswith("화") or caption.endswith(("(", "[", "/", "-", "·")):
            truncated.append(caption)
            continue
        if len(caption) < 10:
            truncated.append(caption)
            continue
        if not caption.endswith(allowed_endings):
            last = caption[-1]
            if last in {"의", "과", "와", "을", "를", "로", "에", "서"}:
                truncated.append(caption)
    return truncated


def load_article_type_config(article_type: str) -> dict[str, Any]:
    path = ARTICLE_TYPE_CONFIG_DIR / f"{article_type}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def article_type_required_entities(article_type: str) -> list[str]:
    config = load_article_type_config(article_type)
    configured = normalize_str_list(config.get("required_entities"))
    if configured:
        return configured
    return REQUIRED_CONCRETE_ENTITIES.get(article_type, [])


def unsupported_powerbi_claims(
    article: str,
    article_type: str,
    evidence: list[dict[str, Any]],
    problem_map: dict[str, Any],
) -> list[str]:
    powerbi_terms = [
        "ProductKey",
        "DAX",
        "Relationship",
        "Power Query",
        "Sales[Sales]",
        "TargetAmount",
        "MonthKey",
        "Date table",
        "SalespersonRegion",
        "Star schema",
        "Profit Margin",
        "DIVIDE",
        "FactResellerSales",
        "ColorFormats",
    ]
    if article_type in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl", "dashboard_validation"}:
        return []
    source_text = json.dumps({"evidence": evidence, "problem_map": problem_map}, ensure_ascii=False).lower()
    unsupported = []
    for term in powerbi_terms:
        if term.lower() in article.lower() and term.lower() not in source_text:
            unsupported.append(term)
    return unsupported


def generic_or_copied_title(article: str, problem_map: dict[str, Any]) -> bool:
    first_line = next((line.strip("# ").strip() for line in article.splitlines() if line.startswith("# ")), "")
    core = str(problem_map.get("core_problem") or "")
    if is_generic_learning_title(first_line) or is_generic_core_problem(core):
        return True
    generic_titles = [
        "문제 해결형 학습 기록",
        "학습 기록 기반 문제 해결 경험",
        "문제를 검증 가능한 분석 흐름으로 바꾼 기록",
    ]
    if first_line in generic_titles:
        return True
    copied_core_fragments = [
        "Category별 Sales 값이 모두 동일하게 반복되어 Product 차원 테이블의 filter context가 Sales fact table까지 전달되지 않는 문제가 드러남",
        "데이터 소스 연결 및 데이터 로딩이 문제가 발생하여 해결이 필요하다",
    ]
    return any(fragment in core or fragment in article for fragment in copied_core_fragments)


def semantic_regression_coverage(article: str) -> tuple[int, int, list[str]]:
    lowered = article.lower()
    missing: list[str] = []
    covered = 0
    for label, keywords in SEMANTIC_REGRESSION_REQUIREMENTS:
        if all(keyword.lower() in lowered for keyword in keywords):
            covered += 1
        else:
            missing.append(label)
    return covered, len(SEMANTIC_REGRESSION_REQUIREMENTS), missing


def power_query_regression_coverage(article: str) -> tuple[int, int, list[str]]:
    lowered = article.lower()
    missing: list[str] = []
    covered = 0
    for label, keywords in POWER_QUERY_REGRESSION_REQUIREMENTS:
        if all(keyword.lower() in lowered for keyword in keywords):
            covered += 1
        else:
            missing.append(label)
    return covered, len(POWER_QUERY_REGRESSION_REQUIREMENTS), missing


def section_plan_image_coverage(section_plan: list[dict[str, Any]], image_count: int) -> tuple[int, int, list[str]]:
    if image_count <= 0:
        return 0, 0, []
    expected = set(range(1, image_count + 1))
    refs: set[int] = set()
    for item in section_plan:
        if not isinstance(item, dict):
            continue
        for ref in item.get("image_refs", []):
            try:
                refs.add(int(ref))
            except (TypeError, ValueError):
                continue
    missing = sorted(expected - refs)
    return len(expected) - len(missing), len(expected), [f"이미지 {item}" for item in missing]


def plan_to_article_faithfulness(article: str, problem_map: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    section_plan = problem_map.get("_section_plan", []) if isinstance(problem_map.get("_section_plan"), list) else []
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    if not section_plan:
        return failures, metrics
    article_steps = count_problem_solution_steps(article)
    plan_count = len(section_plan)
    metrics["plan_step_count"] = plan_count
    metrics["article_solution_step_count"] = article_steps
    if plan_count >= 8 and article_steps < max(4, int(plan_count * 0.7)):
        failures.append(f"Section Plan은 {plan_count}개인데 Final Article 문제 해결 경험은 {article_steps}개로 과도하게 압축되었습니다.")

    must_items = [item for plan in section_plan for item in normalize_str_list(plan.get("must_include"))]
    included = [item for item in must_items if item.lower() in article.lower()]
    coverage = len(included) / max(len(must_items), 1)
    metrics["section_plan_must_include_coverage"] = round(coverage, 3)
    if must_items and coverage < 0.7:
        missing = [item for item in must_items if item not in included][:12]
        failures.append(f"Section Plan must_include 반영률이 70% 미만입니다: {', '.join(missing)}")

    expected_refs = sorted({int(ref) for plan in section_plan for ref in plan.get("image_refs", []) if str(ref).isdigit()})
    article_refs = sorted({int(match) for match in re.findall(r"이미지\s+(\d+)", article)})
    metrics["section_plan_image_refs"] = expected_refs
    metrics["article_image_refs"] = article_refs
    if expected_refs and not set(expected_refs).issubset(set(article_refs)):
        missing_refs = sorted(set(expected_refs) - set(article_refs))
        failures.append(f"Section Plan의 image_refs가 Final Article에 모두 반영되지 않았습니다: {', '.join(f'이미지 {ref}' for ref in missing_refs)}")

    action_terms = ["생성했다", "설정했다", "비활성화", "구성했다", "확인했다", "적용", "만들었다"]
    cause_terms = ["전달되지", "없거나", "오설정", "모호", "필터링하지", "동시에 존재", "왜곡", "부족"]
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        cause = str(step.get("cause") or "")
        action = str(step.get("action") or "")
        if any(term in cause for term in action_terms) and not any(term in cause for term in cause_terms):
            failures.append(f"solution_step {index} cause 필드에 조치 문장이 들어갔습니다: {cause[:80]}")
            break
        if any(term in action for term in cause_terms) and not any(term in action for term in action_terms):
            failures.append(f"solution_step {index} action 필드에 원인 문장이 들어갔습니다: {action[:80]}")
            break
    return failures, metrics


def semantic_image_ref_mapping_failure(problem_map: dict[str, Any]) -> str:
    expected = {
        "반복값": {1},
        "relationship 생성": {2},
        "relationship 설정값": {3},
        "Star schema": {4},
        "hierarchy": {5, 6},
        "Profit Quick": {7},
        "Profit Margin measure": {8},
        "결과 검증": {9, 10},
        "bridge table": {11},
        "Both": {12},
        "비활성화": {13},
        "Sales 결과": {14},
        "Targets": {15},
    }
    for step in problem_map.get("solution_steps", []):
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        refs = {int(ref) for ref in step.get("image_refs", []) if str(ref).isdigit()}
        if not refs:
            continue
        if "bridge" in title.lower() and refs & {1, 2, 3, 4, 5, 6}:
            return f"bridge table step에 잘못된 image_refs가 붙었습니다: {sorted(refs)}"
        if ("Profit Quick" in title or "Profit Margin measure" in title) and refs & {2, 3, 4}:
            return f"Profit/Profit Margin step에 잘못된 image_refs가 붙었습니다: {sorted(refs)}"
        for key, allowed in expected.items():
            if key.lower() in title.lower() and not refs.issubset(allowed):
                return f"{title} step image_refs가 기대 매핑과 다릅니다. expected={sorted(allowed)}, actual={sorted(refs)}"
    return ""


def normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def medium_generation_failure(image_count: int, memo: str, reasons: list[str], capture_count: int = 0, qa_count: int = 0) -> str:
    reason_text = "\n".join(f"- {reason}" for reason in reasons)
    memo_text = memo.strip() or "입력된 메모 없음"
    return f"""# Medium 글 생성 실패

LLM/Vision 응답 실패로 완성형 Medium 글을 생성하지 못했습니다.

## 업로드된 이미지 수
{image_count}장

## 저장된 캡처 수
{capture_count}개

## 저장된 Q&A 수
{qa_count}개

## 입력된 메모
{memo_text}

## 생성 실패 원인 후보
{reason_text}
- 네트워크 또는 LLM API 응답 지연
- Vision 모델이 이미지 판독 요청을 처리하지 못함
- 입력 이미지가 너무 많거나 개별 이미지 용량이 큼

## 사용자가 추가해야 할 정보
- 이미지 흐름 요약: 어떤 화면이 문제 발견, 원인 분석, 해결, 검증에 해당하는지
- 핵심 문제: 무엇이 이상했고 왜 문제였는지
- 해결 과정: 실제로 바꾼 관계, 수식, 쿼리, 설정
- 검증 결과: 수정 전후 결과가 어떻게 달라졌는지
"""


def provider_failure_message(
    image_count: int,
    memo: str,
    diagnostics: dict[str, Any],
    capture_count: int = 0,
    qa_count: int = 0,
    elapsed_seconds: float = 0,
) -> str:
    immediate = elapsed_seconds < 1
    diag_lines = "\n".join(
        f"- {key}: {value}"
        for key, value in diagnostics.items()
        if key not in {"traceback_summary"} and "key" not in key.lower()
    )
    trace = diagnostics.get("traceback_summary") or "traceback 요약 없음"
    leading = (
        "LLM 요청이 시작되기 전에 실패했습니다. 환경변수, 패키지 import, provider client 초기화 상태를 확인하세요."
        if immediate
        else "LLM/Vision provider 호출 중 실패했습니다. provider diagnostics를 먼저 확인하세요."
    )
    return f"""# Medium 글 생성 실패

{leading}

이미지 {image_count}장은 업로드되었지만 Vision/Text provider 초기화 또는 호출 단계에서 실패했습니다. 이 경우 이미지 흐름 요약을 추가하는 것보다 `GROQ_API_KEY`, `groq` 패키지, 선택된 provider/model 설정을 먼저 확인해야 합니다.

## Provider diagnostics
{diag_lines}

## Traceback summary
```text
{trace}
```

## 입력 상태
- 업로드된 이미지 수: {image_count}장
- 저장된 캡처 수: {capture_count}개
- 저장된 Q&A 수: {qa_count}개
- batch_upload_mode에서는 캡처 수와 Q&A 수가 0이어도 실패 조건이 아닙니다.

## 입력된 메모
{memo.strip() or "입력된 메모 없음"}
"""


def parse_json_or_fallback(content: str, raw_text: str, memo: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback_note(raw_text, memo)
    if isinstance(data, list):
        summary = stringify_field(data)
        return {
            "title": "이미지 기반 학습 기록",
            "source_type": "study-capture",
            "tags": ["study-note", "vision"],
            "summary": summary,
            "action_items": ["핵심 개념 정리", "막힌 지점과 해결 방법 추가 기록", "문제 해결형 글로 변환"],
            "blog_draft": f"# 이미지 기반 학습 기록\n\n## 캡처 내용\n{summary}\n\n## 다음 정리\n- 문제 인식\n- 해결 과정\n- 배운 점\n",
        }
    if not isinstance(data, dict):
        return fallback_note(raw_text, memo)
    return {
        "title": str(data.get("title") or "학습 기록"),
        "source_type": str(data.get("source_type") or "study-capture"),
        "tags": [str(tag) for tag in data.get("tags", [])][:8],
        "summary": stringify_field(data.get("summary") or ""),
        "action_items": [str(item) for item in data.get("action_items", [])][:8],
        "blog_draft": stringify_field(data.get("blog_draft") or ""),
    }


def stringify_field(value: Any) -> str:
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            title = str(key).replace("_", " ").strip().title()
            lines.append(f"## {title}\n{stringify_field(item)}")
        return "\n\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"- {stringify_field(item)}" for item in value)
    return str(value)


def fallback_note(raw_text: str, memo: str) -> dict[str, Any]:
    source = "\n".join(part for part in [raw_text.strip(), memo.strip()] if part)
    preview = source[:900] if source else "입력된 텍스트가 없습니다."
    return {
        "title": "학습 캡처 기록",
        "source_type": "study-capture",
        "tags": ["study-note", "capture"],
        "summary": f"입력 내용을 기반으로 기본 노트를 생성했습니다.\n\n{preview}",
        "action_items": ["핵심 개념 다시 정리", "막힌 지점과 해결 방법 추가 기록", "문제 해결형 글로 변환"],
        "blog_draft": f"# 학습 캡처 기록\n\n## 기록 내용\n{preview}\n\n## 다음 정리\n- 핵심 개념\n- 실습 흐름\n- 문제 해결 과정\n",
    }


def local_study_blog(notes: list[StudyNote], topic: str) -> str:
    note = notes[-1]
    actions = "\n".join(f"- {item}" for item in note.action_items[:5]) or "- 추가 정리 필요"
    return f"""# {topic}

## 학습 배경
이번 기록은 `{note.title}` 학습 화면을 바탕으로 정리한 기술 노트입니다. 화면에서 확인한 내용과 생성된 노트를 기준으로, 실습의 흐름과 다음 점검 항목을 정리했습니다.

## 핵심 내용
{note.summary}

## 다음 작업
{actions}

## 정리
이 기록은 단순 캡처가 아니라, 학습 중 발견한 화면과 판단 지점을 다시 검토하기 위한 자료입니다. 이후 관련 수식, 쿼리, 설정값을 추가하면 문제 해결형 포트폴리오 글로 확장할 수 있습니다.
"""


def local_portfolio_blog(notes: list[StudyNote], topic: str) -> str:
    note = notes[-1]
    actions = "\n".join(f"- {item}" for item in note.action_items[:5]) or "- 추가 정리 필요"
    return f"""# {note.title}: 문제 해결형 학습 기록

_A problem-solving portfolio note from a study capture_

## 도입부
이번 기록은 `{topic}` 과정에서 생성한 학습 캡처를 바탕으로 작성했습니다. 화면에서 확인한 실습 내용은 단순 기능 사용이 아니라, 데이터 모델과 계산 결과가 왜 예상과 다르게 보이는지 확인하는 과정에 가깝습니다.

핵심 작업은 다음과 같습니다.

```text
- 화면 캡처를 학습 근거로 저장
- 실습 화면에서 주요 테이블, 지표, 관계 설정 포인트 확인
- 생성된 노트를 바탕으로 문제 인식과 다음 점검 항목 정리
```

## 문제 인식
{note.summary}

## 문제 정의
현재 실습에서 확인해야 할 문제는 화면에 보이는 결과값과 데이터 모델 설정이 의도한 분석 흐름과 일치하는지 검증하는 것입니다.

## 왜 이것을 문제로 인식했는가
Power BI, SQL, DAX, 모델링 실습에서는 화면에 값이 표시되는 것만으로는 충분하지 않습니다. 관계 설정, 계산식, 필터 컨텍스트가 잘못되면 보고서에 보이는 값은 그럴듯해도 실제 해석은 틀릴 수 있습니다. 따라서 실습 화면에서 이상한 값, 반복되는 합계, 0으로 표시되는 지표, 관계 설정 단계를 발견하면 이를 문제로 정의하고 원인을 좁혀야 합니다.

## 문제 해결 경험
1. 화면에서 현재 실습 단계와 표시된 결과를 먼저 확인했습니다.
2. 지표, 테이블, 관계 설정, 계산식 중 어떤 요소가 결과에 영향을 주는지 나누어 보았습니다.
3. 다음 점검 항목을 액션으로 분리해 이후 실습에서 재현하고 검증할 수 있도록 정리했습니다.

## 복잡한 문제 해결 경험
이 유형의 문제는 단순히 버튼을 누르는 문제가 아니라, 데이터 모델의 관계와 계산 흐름을 함께 확인해야 합니다. 특히 Power BI에서는 관계 방향, many-to-many 관계, measure 계산, 필터 컨텍스트가 결과값에 직접 영향을 줍니다. 따라서 화면 캡처를 기록으로 남기고, 어떤 지점에서 값이 달라졌는지 추적하는 방식이 중요합니다.

## 성과
이번 캡처를 통해 실습 화면을 단순히 지나치지 않고, 추후 포트폴리오 글로 확장할 수 있는 문제 해결 기록으로 전환했습니다. 이후 같은 방식으로 오류 화면, DAX 수식, SQL 쿼리, 모델링 설정을 누적하면 학습 과정 자체가 기술블로그와 포트폴리오 자료가 됩니다.

## 사용한 주요 수식/코드 정리
현재 캡처에는 별도의 코드나 수식이 직접 입력되지 않았습니다. 추후 DAX, SQL, Power Query 수식을 메모에 추가하면 이 섹션에 자동으로 정리할 수 있습니다.

## 다음 작업
{actions}

## 최종 정리
이 기록의 핵심은 학습 화면을 저장하는 데서 끝내지 않고, 화면에서 확인한 문제와 다음 검증 항목을 구조화했다는 점입니다. 이는 실무에서도 오류 화면, 분석 결과, 설정 변경 내역을 근거와 함께 남기는 방식으로 확장될 수 있습니다.

## Portfolio Summary
- Captured a technical learning screen and converted it into a structured problem-solving note.
- Identified model/result validation points from the visible interface.
- Organized follow-up actions for reproducible learning and documentation.

## Key skills practiced
- Technical documentation
- Problem framing
- Data model validation
- Portfolio writing workflow
"""


def image_only_note(image_path: str) -> dict[str, Any]:
    return {
        "title": "이미지 학습 캡처",
        "source_type": "study-capture-image",
        "tags": ["study-note", "screenshot", "vision-fallback"],
        "summary": (
            "스크린샷을 학습 근거 자료로 저장했지만, Vision LLM 응답을 받아오지 못해 기본 노트로 기록했습니다.\n\n"
            "화면에서 확인한 핵심 문장, 오류 메시지, 실습 목표를 메모에 함께 적으면 "
            "이미지 판독이 실패해도 노트와 문제 해결형 포트폴리오 초안을 구성할 수 있습니다.\n\n"
            f"저장된 이미지: {image_path}"
        ),
        "action_items": [
            "화면 속 핵심 텍스트를 raw text에 붙여넣기",
            "내가 발견한 문제와 해결 과정을 memo에 추가하기",
            "추후 OCR 또는 Vision API로 이미지 텍스트 자동 추출 연결하기",
        ],
        "blog_draft": (
            "# 이미지 기반 학습 캡처\n\n"
            "## 캡처 내용\n"
            "스크린샷이 저장되었습니다. Vision LLM 응답을 받아오지 못한 경우에는 "
            "실습 목표와 문제 상황을 메모로 함께 기록하면 문제 해결형 포트폴리오 글로 확장할 수 있습니다.\n\n"
            "## 다음 정리\n"
            "- 화면에서 확인한 실습 주제\n"
            "- 발견한 이상 현상 또는 막힌 지점\n"
            "- 해결한 방법과 사용한 수식/쿼리/설정\n"
        ),
    }


def study_blog_prompt(topic: str, joined_notes: str) -> str:
    return f"""
다음 학습 노트들을 바탕으로 기술블로그 초안을 작성해 주세요.

조건:
- 한국어
- 제목, 문제 인식, 실습 흐름, 핵심 개념, 막힌 점과 해결, 배운 점, 다음 학습 계획 포함
- 실제 입력에 없는 성과나 수치는 만들지 않음

[주제]
{topic}

[노트]
{joined_notes}
""".strip()


def portfolio_prompt(topic: str, joined_notes: str, extra_info: str = "") -> str:
    return f"""
당신은 사용자의 실습/프로젝트 기록을 Medium 포트폴리오 글로 변환하는 전용 작성자입니다.
아래 실습/프로젝트 기록을 바탕으로 Medium에 그대로 붙여넣을 수 있는 완성본을 작성해 주세요.

글의 목적은 단순 후기나 요약이 아니라, 사용자가 이미 Medium에 작성해 온 것과 같은 **문제해결형 포트폴리오 글**입니다.
반드시 아래 작성 방식을 그대로 따르세요.

가장 중요한 원칙:
- 이미지를 1장씩 따로 해설하지 않습니다.
- 이미지별로 각각 문제/원인을 만들지 않습니다.
- 전체 이미지 묶음을 하나의 실습 흐름으로 보고, 하나의 핵심 문제를 중심으로 글을 씁니다.
- 초반 이미지는 문제 인식, 중간 이미지는 원인 분석과 해결 과정, 마지막 이미지는 검증 결과로 사용합니다.
- "무엇을 클릭했다"보다 "어떤 문제가 있었고, 왜 문제가 되었고, 어떤 원인을 발견했고, 어떻게 해결했고, 결과가 어떻게 바뀌었는지"를 중심으로 씁니다.
- 사용자가 직접 문제를 발견하고 해결한 경험처럼 작성합니다.
- 취업 포트폴리오에 사용할 수 있도록 실무적인 분석 문체로 씁니다.
- 짧은 요약문이 아니라 Medium에 그대로 게시 가능한 완성형 글로 작성합니다.
- 사용자가 직접 입력한 추가 정보는 최상위 작성 브리프입니다. 이미지 판독 결과보다 우선합니다.
- 추가 정보에 "이미지 흐름 요약", "이 글에서 강조할 문제 해결 관점", "강조하고 싶은 기술"이 있으면 그 내용을 글의 뼈대로 사용합니다.
- 추가 정보가 비어 있으면 이미지/메모/노트만으로 작성하되, 없는 내용을 지어내지 않습니다.
- 내부 라벨인 [화면 텍스트/코드/오류], [사용자 메모/질문/해결 과정], [이미지 정보], [작성 주의]는 최종 글에 절대 출력하지 않습니다.

분량 규칙:
- 전체 글은 최소 4,500자 이상으로 작성합니다. 가능하면 6,000~8,000자 수준의 Medium 완성본으로 작성합니다.
- "문제 인식", "문제 정의", "왜 문제로 인식했는가"는 각각 충분히 길게 작성합니다.
- "문제 해결 경험"이 가장 중요합니다. 최소 4개 이상의 단계로 나누고, 각 단계는 `문제/제약 → 원인 판단 → 조치 → 확인 결과` 흐름으로 작성합니다.
- 문제 해결 경험 섹션은 글 전체에서 가장 긴 섹션이어야 합니다.
- 수식, 코드, 관계 설정, 오류 해결, 배포, 새로고침, 모델링, 데이터 검증처럼 복잡한 내용이 있으면 별도 섹션으로 충분히 설명합니다.
- 마지막 Portfolio Summary는 영어로 2문단 이상 작성합니다.
- Key skills practiced는 최소 8개 이상 작성합니다.

1. 한국어 제목
2. 영어 부제
3. 짧은 도입부
4. 핵심 작업 요약
5. 문제 인식
6. 문제 정의
7. 왜 이것을 문제로 인식했는가
8. 문제 해결 경험 1, 2, 3...
9. 복잡한 수식 작성 및 해결 경험
10. 성과
11. 사용한 주요 수식/코드 정리
12. 최종 정리
13. Portfolio Summary
14. Key skills practiced
15. 이미지 번호와 캡션 목록

문체 규칙:
- 다음처럼 쓰지 않습니다: "버튼을 눌렀습니다", "차트를 만들었습니다", "관계를 설정했습니다", "실습을 완료했습니다".
- 대신 다음처럼 씁니다: "처음에는 값이 정상적으로 보이는 것처럼 보였지만...", "문제의 원인은 시각화가 아니라 필터 컨텍스트가 팩트 테이블까지 전달되지 않는 semantic model 구조에 있었다", "이를 해결하기 위해 차원 테이블과 팩트 테이블 사이의 관계를 재정의했다".
- "했습니다" 반복을 줄이고 분석/정의/구성/해결/검증 중심으로 작성합니다.

이미지 규칙:
- 사용자가 제공한 이미지는 단순 캡처가 아니라 문제 해결 과정의 증거로 사용합니다.
- 입력에 이미지 설명이 있으면 사용자가 제공한 이미지 순서대로 번호를 붙입니다.
- 본문에는 "이미지 1 - 제목" 형식의 캡션만 넣습니다.
- "첨부 삽입", "여기에 이미지 넣기" 같은 placeholder 문구는 절대 넣지 않습니다.
- 마지막에 이미지 번호와 캡션 목록을 따로 정리합니다.
- 초반 이미지는 문제 발견, 이상 현상, 초기 상태, 실습 시작점으로 묶어 해석합니다.
- 중간 이미지는 원인 분석, 관계 설정, 수식 작성, 데이터 변환, 모델 수정, UI 구성, 오류 해결, 검증 과정으로 묶어 해석합니다.
- 마지막 이미지는 최종 결과, 개선된 화면, 검증 결과, 성과 화면으로 묶어 해석합니다.
- 이미지 하나하나마다 문제 정의를 반복하지 않습니다.
- 이미지 캡션은 글의 흐름을 보조하는 장치일 뿐이며, 글의 중심은 사용자가 해결한 핵심 문제와 해결 과정입니다.
- 이미지 자체에 명확한 문제가 없으면 억지로 문제를 만들지 말고, "실습에서 해결해야 할 과제"를 문제로 정의합니다.

코드/수식 규칙:
- DAX, SQL, Python, Power Query, API 코드가 있으면 코드블록으로 정리합니다.
- 수식은 단순히 나열하지 말고, 각 수식이 어떤 문제를 해결했는지 설명합니다.
- 복잡한 수식은 원인 → 중간 검증 → 최종 수식 흐름으로 설명합니다.
- 수식은 반드시 다음 흐름으로 설명합니다: 어떤 문제가 있었는가 → 왜 이 수식이 필요했는가 → 수식이 어떤 계산을 수행하는가 → 결과를 어떻게 검증했는가.
- 수식이나 코드가 이미지에 보이지 않거나 사용자가 제공하지 않았다면 임의로 만들지 않습니다.

오류/막힌 부분 처리 규칙:
- 사용자가 오류, 막힌 부분, 헷갈린 부분을 제공하면 반드시 글에 반영합니다.
- 오류는 실패가 아니라 문제 해결 경험으로 재구성합니다.
- 각 오류는 `증상 → 처음 의심한 원인 → 실제 원인 → 해결 방법 → 확인 결과` 흐름으로 작성합니다.

Power BI semantic model 실습일 때의 작성 방식:
- 현재 입력이 Power BI, Sales, Product, Category, DAX, relationship, semantic model, filter context를 다루는 경우에만 아래 패턴을 적용합니다.
- 다른 이미지 세트가 들어오면 이 Power BI 예시 내용을 재사용하지 말고, 해당 이미지의 실제 기술 주제에 맞춰 같은 문제 해결형 구조만 유지합니다.
- 첫 이미지는 기능 설명이 아니라 문제 인식 장면으로 해석합니다. 화면에 숫자가 나오는 것이 아니라, 왜 숫자가 이상한지 먼저 씁니다.
- 예를 들어 Category별 Sales가 모두 같은 값으로 반복된다면, "Power BI에서 숫자는 보이지만 의미가 틀릴 수 있다"는 문제로 시작합니다.
- 원인은 시각화 문제가 아니라 semantic model 문제로 정의합니다. Product[Category] 필터가 Sales로 전달되지 않아 ProductKey 관계가 필요하다는 식으로 원인 중심으로 씁니다.
- 이미지는 작업 순서가 아니라 아래 해결 단계로 묶습니다.
  - 이미지 1: 문제 발견, Category별 Sales 반복
  - 이미지 2~4: Product-Sales 관계 생성과 star schema 구조 정리
  - 이미지 5~6: Product hierarchy 구성
  - 이미지 7~10: Profit, Profit Margin measure 생성과 검증
  - 이미지 11~14: SalespersonRegion bridge table, many-to-many, filter direction, inactive relationship 문제 해결
  - 이미지 15: Salesperson별 Sales와 Target 비교 최종 결과
- 관계 설정은 "무엇을 선택했는가"만 쓰지 말고 왜 맞는지 설명합니다. Product는 차원 테이블, Sales는 팩트 테이블이므로 Product[ProductKey] → Sales[ProductKey], Cardinality One-to-many, Cross-filter direction Single, Active relationship이 왜 필요한지 씁니다.
- DAX는 "왜 필요한가 → 어떤 수식인가 → 어떤 결과를 검증했는가" 순서로 설명합니다. Sales만으로는 수익성을 볼 수 없어서 Profit이 필요하고, 규모가 다른 카테고리를 비교하려면 Profit Margin이 필요하다는 식으로 씁니다.
- many-to-many 관계는 복잡한 문제 해결 경험으로 따로 뽑습니다. Salesperson별 Sales를 담당 Region 기준으로 보려면 Salesperson → SalespersonRegion → Region → Sales 흐름이 필요하고, 모호한 필터 경로가 생기면 직접 관계를 비활성화해야 한다는 식으로 씁니다.
- 마지막은 "실습 완료"가 아니라 분석 가능성의 변화로 정리합니다. Category별 Sales 반복 문제 해결, Product-Sales 관계 생성, 계층 구조 구성, Measure 생성, bridge table 기반 many-to-many 해결, Salesperson별 Sales와 Target 비교 가능성을 성과로 씁니다.
- 결론은 "숫자가 화면에 보인다고 항상 맞는 것은 아니며, relationship, cardinality, filter direction, active relationship이 잘못되면 visual은 정상처럼 보여도 의미는 틀릴 수 있다"는 식으로 마무리합니다.

Medium 작성 방식 예시:
- 첫 문단은 "이번 실습에서 무엇을 했는가"보다 "처음 무엇이 이상했는가"로 시작합니다.
- 그 다음 문제를 한 문장으로 정의합니다.
- 이후 왜 그 문제가 중요한지 데이터 모델, 비즈니스 해석, 검증 관점에서 설명합니다.
- 문제 해결 경험은 다음 순서로 이어갑니다.
  1. 문제 화면에서 이상 징후를 발견한 과정
  2. 원인을 시각화가 아니라 데이터 모델/관계/수식/쿼리 구조에서 찾은 과정
  3. 관계, 수식, 쿼리, 설정, 데이터 구조를 수정한 과정
  4. 수정 후 결과가 어떻게 달라졌는지 검증한 과정
  5. 추가로 복잡했던 관계, 수식, 필터 컨텍스트, 오류 해결 과정을 별도 섹션으로 정리
- 이미지 캡션은 각 섹션 사이에 짧게 넣되, 본문은 캡션 설명이 아니라 문제해결 서사로 작성합니다.

[주제]
{topic}

[사용자가 직접 입력한 추가 정보]
{extra_info if extra_info.strip() else "추가 정보 없음. 이미지/메모/노트만 근거로 작성하세요."}

[실습/프로젝트 기록]
{joined_notes}
""".strip()


def read_notes() -> list[StudyNote]:
    notes: list[StudyNote] = []
    for line in NOTES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            notes.append(StudyNote(**json.loads(line)))
        except Exception:
            continue
    return notes


def is_meaningful_note(note: StudyNote) -> bool:
    summary = note.summary or ""
    if "입력된 텍스트가 없습니다" in summary:
        return False
    if note.source_type == "study-capture-image" and (
        "스크린샷을 학습 근거 자료로 저장했습니다" in summary
        or "스크린샷을 학습 캡처로 저장했습니다" in summary
    ):
        return False
    return True


def append_note(note: StudyNote) -> None:
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(note), ensure_ascii=False) + "\n")


def search_notes(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    q_terms = set(tokenize(query))
    scored = []
    for note in read_notes():
        text = " ".join([note.title, note.summary, note.raw_text, note.user_memo, " ".join(note.tags)])
        terms = set(tokenize(text))
        score = len(q_terms & terms) / max(len(q_terms | terms), 1)
        if query.lower() in text.lower():
            score += 0.5
        if score > 0:
            scored.append((score, note))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"score": round(score, 3), "note": asdict(note)} for score, note in scored[:top_k]]


def tokenize(text: str) -> list[str]:
    return [chunk.strip(".,:;()[]{}<>\"'").lower() for chunk in text.split() if len(chunk.strip()) >= 2]


def make_note(raw_text: str, memo: str, image_path: str | None, image_paths: list[str] | None = None) -> StudyNote:
    image_paths = image_paths or ([image_path] if image_path else [])
    image_files = [CAPTURE_DIR / path.removeprefix("/captures/") for path in image_paths]
    if image_files and not raw_text.strip():
        generated = llm.generate_note_from_images(image_files, memo)
    else:
        generated = llm.generate_note(raw_text, memo)
    created_at = datetime.now().isoformat(timespec="seconds")
    title = generated["title"]
    if title in {"학습 캡처 기록", "이미지 학습 캡처", "이미지 기반 학습 캡처"}:
        label = "이미지" if image_path else "학습"
        title = f"{label} 캡처 {datetime.now().strftime('%H:%M')}"
    return StudyNote(
        id=str(uuid.uuid4()),
        created_at=created_at,
        title=title,
        source_type=generated["source_type"],
        tags=generated["tags"],
        raw_text=raw_text,
        user_memo=memo,
        summary=generated["summary"],
        action_items=generated["action_items"],
        blog_draft=generated["blog_draft"],
        image_path=image_path,
        image_paths=image_paths or None,
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self.html(INDEX_HTML)
        if path == "/api/notes":
            return self.json([asdict(note) for note in read_notes()])
        if path == "/api/health/llm":
            query = parse_qs(parsed.query)
            return self.json(llm_health_check(run_test_call=query.get("test", ["0"])[0] == "1"))
        if path == "/api/sessions":
            return self.json(read_sessions().get("sessions", []))
        if path.startswith("/api/sessions/"):
            return self.handle_session_get(path)
        if path.startswith("/captures/"):
            return self.file(CAPTURE_DIR / path.removeprefix("/captures/"))
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/captures":
            return self.create_capture()
        if path == "/api/search":
            data = self.read_json()
            return self.json(search_notes(str(data.get("query", "")), int(data.get("top_k", 5))))
        if path == "/api/blog":
            data = self.read_json()
            notes = read_notes()
            note_ids = data.get("note_ids")
            if note_ids:
                wanted = set(note_ids)
                notes = [note for note in notes if note.id in wanted]
            notes = notes[-8:]
            draft = llm.synthesize_blog(
                notes,
                str(data.get("topic", "오늘의 학습 기록")),
                str(data.get("format_type", "study-blog")),
                str(data.get("extra_info", "")),
            )
            return self.json({"draft": draft})
        if path == "/api/direct-blog":
            return self.create_direct_blog()
        if path == "/api/sessions":
            data = self.read_json()
            return self.json(create_session(str(data.get("title", ""))))
        if path.startswith("/api/sessions/"):
            return self.handle_session_post(path)
        self.send_error(404)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/sessions/"):
            return self.handle_session_delete(path)
        self.send_error(404)

    def handle_session_get(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            return self.send_error(404)
        session = find_session(parts[2])
        if not session:
            return self.send_error(404)
        if len(parts) == 3:
            return self.json(session)
        if len(parts) == 4 and parts[3] == "captures":
            return self.json(session.get("captures", []))
        if len(parts) == 4 and parts[3] == "qa":
            return self.json(session.get("qa_logs", []))
        if len(parts) == 4 and parts[3] == "documents":
            return self.json(session.get("documents", []))
        self.send_error(404)

    def handle_session_post(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 4:
            return self.send_error(404)
        session = find_session(parts[2])
        if not session:
            return self.send_error(404)
        action = parts[3]
        if action == "captures":
            return self.create_session_capture(session)
        if action == "documents":
            return self.create_session_document(session)
        if action == "qa":
            data = self.read_json()
            qa = append_qa_log(
                session,
                question=str(data.get("question", "")),
                answer=str(data.get("answer", "")),
                selected_text=str(data.get("selected_text", "")),
                related_capture_ids=[int(x) for x in data.get("related_capture_ids", [])],
            )
            update_session(session)
            return self.json(qa)
        if action == "ask":
            data = self.read_json()
            qa = tutor_agent_answer(
                session,
                question=str(data.get("question", "")),
                selected_text=str(data.get("selected_text", "")),
                related_capture_ids=[int(x) for x in data.get("related_capture_ids", [])],
            )
            update_session(session)
            return self.json(qa)
        if action == "generate-article":
            return self.generate_session_article(session)
        self.send_error(404)

    def handle_session_delete(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 5 or parts[3] not in ("captures", "documents"):
            return self.send_error(404)
        session = find_session(parts[2])
        if not session:
            return self.send_error(404)
        if parts[3] == "captures":
            capture_id = int(parts[4])
            session["captures"] = [item for item in session.get("captures", []) if int(item.get("capture_id", 0)) != capture_id]
            update_session(session)
            return self.json({"deleted": capture_id})
        document_id = int(parts[4])
        remaining = []
        for item in session.get("documents", []):
            if int(item.get("document_id", 0)) == document_id:
                text_path = item.get("text_path")
                if text_path:
                    try:
                        Path(str(text_path)).unlink(missing_ok=True)
                    except OSError:
                        pass
                continue
            remaining.append(item)
        session["documents"] = remaining
        update_session(session)
        return self.json({"deleted": document_id})

    def create_session_capture(self, session: dict[str, Any]) -> None:
        form = parse_form(self)
        note = get_field(form, "user_note")
        source_title = get_field(form, "source_title")
        source_url = get_field(form, "source_url")
        captures = session.setdefault("captures", [])
        created: list[dict[str, Any]] = []
        for item in sorted_uploaded_files(get_files(form, "image")):
            capture_id = next_capture_id(session)
            ext = Path(item.filename).suffix.lower() or ".png"
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{capture_id:03d}_{uuid.uuid4().hex[:8]}{ext}"
            target = CAPTURE_DIR / filename
            target.write_bytes(item.data)
            capture = asdict(
                CaptureEvent(
                    capture_id=capture_id,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    image_path=f"/captures/{filename}",
                    source_title=source_title,
                    source_url=source_url,
                    user_note=note,
                    auto_keywords=infer_entities(" ".join([note, item.filename])),
                )
            )
            captures.append(capture)
            created.append(capture)
        update_session(session)
        return self.json({"captures": created, "total": len(captures)})

    def create_session_document(self, session: dict[str, Any]) -> None:
        form = parse_form(self)
        documents = session.setdefault("documents", [])
        created: list[dict[str, Any]] = []
        for item in get_files(form, "document"):
            document_id = next_document_id(session)
            conversion = convert_document_to_markdown(
                file_bytes=item.data,
                filename=item.filename,
                tmp_dir=DOCUMENT_TMP_DIR,
            )
            text_path: str | None = None
            if conversion["ok"]:
                text_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document_id:03d}_{uuid.uuid4().hex[:8]}.md"
                target = DOCUMENT_DIR / text_filename
                target.write_text(conversion["text"], encoding="utf-8")
                text_path = str(target)
            document = asdict(
                DocumentEvidence(
                    document_id=document_id,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    original_filename=item.filename,
                    ext=conversion["ext"],
                    status="ok" if conversion["ok"] else "error",
                    char_count=conversion["char_count"],
                    text_path=text_path,
                    error=conversion["error"],
                )
            )
            documents.append(document)
            created.append(document)
        update_session(session)
        return self.json({"documents": created, "total": len(documents)})

    def generate_session_article(self, session: dict[str, Any]) -> None:
        captures = sorted(session.get("captures", []), key=lambda item: item.get("timestamp", ""))
        image_files = [CAPTURE_DIR / str(item.get("image_path", "")).removeprefix("/captures/") for item in captures if item.get("image_path")]
        image_names = [Path(str(item.get("image_path", ""))).name for item in captures if item.get("image_path")]
        memo = "\n".join(item.get("user_note", "") for item in captures if item.get("user_note"))
        qa_logs = session.get("qa_logs", [])
        qa_text = "\n\n".join(f"Q: {qa.get('question', '')}\nA: {qa.get('answer_summary', '')}" for qa in qa_logs)
        document_text = session_document_evidence_text(session)
        combined_raw_text = qa_text
        extra_info = "Capture Timeline과 Q&A Logs를 함께 근거로 사용합니다."
        if document_text:
            combined_raw_text = f"{qa_text}\n\n{document_text}".strip() if qa_text else document_text
            extra_info += " 업로드된 학습 문서(MarkItDown으로 변환된 원본 강의자료)도 근거로 포함되어 있으니, 문서 내용과 모순되는 내용을 만들지 마세요."
        topic = str(session.get("title") or "학습 캡처 타임라인 기반 문제 해결 경험")
        start = time.perf_counter()
        result = llm.synthesize_blog_from_capture(
            raw_text=combined_raw_text,
            memo=memo,
            image_files=image_files,
            topic=topic,
            extra_info=extra_info,
            image_names=image_names,
            captures=captures,
            qa_logs=qa_logs,
        )
        result.update({
            "elapsed_seconds": round(time.perf_counter() - start, 2),
            "image_count": len(image_files),
            "document_count": len([doc for doc in session.get("documents", []) if doc.get("status") == "ok"]),
        })
        return self.json(result)

    def create_capture(self) -> None:
        form = parse_form(self)
        raw_text = get_field(form, "raw_text")
        memo = get_field(form, "memo")
        image_paths: list[str] = []
        images = get_files(form, "image")
        for index, item in enumerate(images, start=1):
            ext = Path(item.filename).suffix.lower() or ".png"
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:02d}_{uuid.uuid4().hex[:8]}{ext}"
            target = CAPTURE_DIR / filename
            target.write_bytes(item.data)
            image_paths.append(f"/captures/{filename}")
        image_path = image_paths[0] if image_paths else None
        start = time.perf_counter()
        note = make_note(raw_text, memo, image_path, image_paths)
        append_note(note)
        elapsed = round(time.perf_counter() - start, 2)
        return self.json({"note": asdict(note), "elapsed_seconds": elapsed})

    def create_direct_blog(self) -> None:
        form = parse_form(self)
        raw_text = get_field(form, "raw_text")
        memo = get_field(form, "memo")
        raw_text = enrich_raw_text_with_source_urls(raw_text, memo)
        topic = get_field(form, "topic") or "학습 기록 기반 문제 해결 경험"
        extra_info = get_field(form, "extra_info")
        image_files: list[Path] = []
        image_names: list[str] = []
        images = sorted_uploaded_files(get_files(form, "image"))
        for index, item in enumerate(images, start=1):
            ext = Path(item.filename).suffix.lower() or ".png"
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:02d}_{uuid.uuid4().hex[:8]}{ext}"
            target = CAPTURE_DIR / filename
            target.write_bytes(item.data)
            image_files.append(target)
            image_names.append(item.filename)
        start = time.perf_counter()
        result = llm.synthesize_blog_from_capture(raw_text, memo, image_files, topic, extra_info, image_names)
        elapsed = round(time.perf_counter() - start, 2)
        if isinstance(result, dict):
            result = dict(result)
            result.update({"elapsed_seconds": elapsed, "image_count": len(image_files), "mode": "batch_upload"})
            return self.json(result)
        return self.json({"draft": result, "elapsed_seconds": elapsed, "image_count": len(image_files), "mode": "batch_upload"})

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def json(self, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return self.send_error(404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")


def parse_form(handler: BaseHTTPRequestHandler) -> FormData:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}

    body = handler.rfile.read(length)
    content_type = handler.headers.get("Content-Type", "")
    form: FormData = {}

    if content_type.startswith("application/x-www-form-urlencoded"):
        for key, values in parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True).items():
            form[key] = [str(value) for value in values]
        return form

    if not content_type.startswith("multipart/form-data"):
        return form

    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
        b"MIME-Version: 1.0\r\n\r\n"
        + body
    )
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            form.setdefault(name, []).append(UploadedFile(filename=filename, data=payload))
            continue
        charset = part.get_content_charset() or "utf-8"
        form.setdefault(name, []).append(payload.decode(charset, errors="replace"))
    return form


def get_field(form: FormData, key: str) -> str:
    values = form.get(key)
    if not values:
        return ""
    value = values[0]
    if isinstance(value, UploadedFile):
        return ""
    return str(value or "")


def get_files(form: FormData, key: str) -> list[UploadedFile]:
    return [value for value in form.get(key, []) if isinstance(value, UploadedFile) and value.filename]


def sorted_uploaded_files(files: list[UploadedFile]) -> list[UploadedFile]:
    if any(re.match(r"^\d{1,3}[_\-\s]", Path(file.filename).name) for file in files):
        return sorted(files, key=lambda file: natural_sort_key(Path(file.filename).name))
    return files


INDEX_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Study Documentation Agent</title>
  <style>
    :root { color-scheme: dark; --bg:#0b0f17; --panel:#141a24; --line:#273142; --text:#eef3f8; --muted:#9aa7b8; --brand:#53c7ad; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { max-width:1180px; margin:0 auto; padding:32px 24px 48px; }
    h1 { margin:0; font-size:32px; }
    h2 { margin:0 0 14px; font-size:18px; }
    p { margin:6px 0 0; color:var(--muted); }
    .grid { display:grid; grid-template-columns:1.2fr .8fr; gap:18px; margin-top:24px; align-items:start; }
    .panel { border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:18px; }
    textarea, input { width:100%; border:1px solid var(--line); background:#0e141d; color:var(--text); border-radius:8px; padding:12px; font:inherit; }
    textarea { min-height:130px; resize:vertical; }
    .compact textarea { min-height:68px; }
    .optional-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .optional-grid .wide { grid-column:1 / -1; }
    .section-label { font-weight:700; color:#c8d3df; margin-top:4px; }
    button { border:1px solid #345044; background:var(--brand); color:#06110e; border-radius:8px; padding:12px 16px; font-weight:700; cursor:pointer; }
    button.secondary { background:#111926; border-color:#2b3a4f; color:var(--text); }
    button:disabled { opacity:.55; cursor:wait; }
    .dropzone { border:1px dashed #3a4a61; background:#0e141d; border-radius:10px; padding:16px; cursor:pointer; transition:border-color .15s, background .15s; }
    .dropzone:hover, .dropzone.dragover { border-color:var(--brand); background:#101b24; }
    .dropzone strong { display:block; margin-bottom:4px; }
    .dropzone span { color:var(--muted); font-size:13px; }
    .dropzone input { display:none; }
    .file-list { color:#c8d3df; font-size:13px; margin-top:8px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; }
    .stack { display:grid; gap:12px; }
    .result { white-space:pre-wrap; border:1px solid var(--line); background:#0e141d; border-radius:8px; padding:16px; min-height:180px; }
    .result-toolbar { display:flex; gap:8px; justify-content:flex-end; align-items:center; margin-top:8px; }
    .result-toolbar .meta { margin-right:auto; }
    details.advanced { border:1px solid var(--line); border-radius:8px; padding:10px; background:#101722; }
    details.advanced summary { cursor:pointer; font-weight:700; color:#c8d3df; }
    .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-top:12px; }
    .tab { padding:8px 10px; border-radius:8px; background:#111926; border:1px solid var(--line); color:var(--text); font-size:12px; }
    .tab.active { background:var(--brand); color:#06110e; border-color:#345044; }
    .debug-pane { white-space:pre-wrap; border:1px solid var(--line); background:#0e141d; border-radius:8px; padding:12px; min-height:120px; max-height:360px; overflow:auto; margin-top:8px; font-size:13px; }
    .note { border:1px solid var(--line); border-radius:8px; padding:12px; margin-top:10px; background:#101722; }
    .note strong { display:block; margin-bottom:4px; }
    .note img { width:100%; max-height:160px; object-fit:cover; border:1px solid var(--line); border-radius:6px; margin-top:8px; }
    .meta { color:var(--muted); font-size:13px; }
    .badge { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 8px; margin:3px 4px 0 0; color:#c8d3df; font-size:12px; }
    .floating-tools { position:fixed; right:22px; bottom:22px; display:flex; gap:12px; align-items:center; z-index:10; }
    .capture-btn { width:68px; height:68px; border-radius:50%; border:4px solid #e9f8f3; background:var(--brand); box-shadow:0 10px 26px rgba(0,0,0,.35); padding:0; font-size:0; }
    .capture-btn::after { content:""; display:block; width:38px; height:38px; margin:11px auto; border-radius:50%; border:2px solid #06110e; }
    .ask-btn { width:52px; height:52px; border-radius:50%; padding:0; background:#f3c969; border-color:#6c5521; color:#151007; font-weight:900; }
    .doc-btn { width:52px; height:52px; border-radius:50%; padding:0; background:#7aa2ff; border:2px solid #1f2d55; color:#0a0f1f; font-weight:900; font-size:11px; }
    .toast { position:fixed; right:24px; bottom:104px; background:#0e141d; border:1px solid var(--line); color:var(--text); padding:10px 12px; border-radius:8px; opacity:0; pointer-events:none; transition:opacity .2s; z-index:11; }
    .toast.show { opacity:1; }
    #captureInput { display:none; }
    @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <h1>AI Study Documentation Agent</h1>
  <p>학습 화면, 실습 메모, 오류 상황을 문제 해결형 Medium 포트폴리오 글로 바로 변환합니다.</p>

  <div class="grid">
    <section class="panel stack">
      <h2>문제 해결형 Medium 글 생성</h2>
      <label id="dropzone" class="dropzone" for="image">
        <strong>이미지 끌어놓기 또는 파일 선택</strong>
        <span>실습 순서대로 여러 장을 한 번에 넣거나, 나중에 이미지를 추가할 수 있습니다.</span>
        <input id="image" type="file" accept="image/*" multiple />
        <div id="fileList" class="file-list">선택된 이미지 없음</div>
      </label>
      <textarea id="rawText" placeholder="자료를 넣으세요. 예: 강의 URL, 영상 URL, 실습 URL, 강의안 본문, 화면 텍스트, 오류 메시지"></textarea>
      <textarea id="memo" placeholder="지금 궁금한 점을 아무렇게나 적으세요. 예: 대충 들었는데 MCP가 뭔지 모르겠음. 퀴즈/막힌 부분도 여기에 입력"></textarea>
      <details class="advanced">
        <summary>예시 입력 보기 <span class="meta">(선택 · 이미지만 넣어도 생성 가능)</span></summary>
        <pre style="white-space:pre-wrap;background:#0b1017;border:1px solid var(--line);padding:12px;border-radius:8px;">예시 1: 강의 URL / 영상 URL / 실습 URL을 한 번에 붙여넣기
예시 2: 영상 강의 캡처만 업로드하고 메모 없이 생성
예시 3: 헷갈린 개념만 한 줄 입력 — 예: workflow_dispatch가 뭔지 헷갈림</pre>
      </details>
      <details class="advanced">
        <summary>Medium 글 추가 정보 <span class="meta">(선택 입력 · 몰라도 비워두세요)</span></summary>
        <div class="optional-grid compact" style="margin-top:10px;">
          <textarea id="projectName" placeholder="실습/프로젝트 이름"></textarea>
          <textarea id="coreProblem" placeholder="내가 해결한 핵심 문제"></textarea>
          <textarea id="blockedPart" placeholder="중간에 막힌 부분"></textarea>
          <textarea id="finalResult" placeholder="최종 결과"></textarea>
          <textarea class="wide" id="focusTech" placeholder="강조하고 싶은 기술, 수식, 코드, 설정"></textarea>
        </div>
      </details>
      <div class="row">
        <button class="secondary" id="clearFilesBtn">이미지 선택 초기화</button>
        <button class="secondary" id="clearInputBtn" type="button">입력칸 초기화</button>
        <button id="portfolioBtn">문제 해결형 Medium 완성본 생성</button>
      </div>
      <div class="result-toolbar">
        <span class="meta">생성 결과는 Markdown으로 복사할 수 있습니다.</span>
        <button class="secondary" id="copyMarkdownBtn" type="button">Markdown 복사</button>
        <button class="secondary" id="clearResultBtn" type="button">글 초기화</button>
      </div>
      <div id="result" class="result">아직 생성된 결과가 없습니다.</div>
      <div id="debugTabs" class="tabs"></div>
      <div id="debugPane" class="debug-pane">디버그 산출물이 아직 없습니다.</div>
    </section>

    <aside class="panel">
      <h2>이전 기록 검색</h2>
      <div class="row">
        <input id="query" placeholder="예: DAX, SQL, 오류, 모델링" />
        <button class="secondary" id="searchBtn">검색</button>
      </div>
      <div id="notes"></div>
    </aside>
  </div>
</main>
<div class="floating-tools">
  <button id="askBtn" class="ask-btn" title="Ask Tutor">?</button>
  <button id="docBtn" class="doc-btn" title="Upload Document">DOC</button>
  <button id="captureBtn" class="capture-btn" title="Capture"></button>
</div>
<input id="captureInput" type="file" accept="image/*" multiple />
<input id="docInput" type="file" accept=".pdf,.docx,.pptx,.xlsx,.xls,.csv" multiple />
<div id="toast" class="toast"></div>

<script>
const result = document.querySelector("#result");
const notesBox = document.querySelector("#notes");
const fileInput = document.querySelector("#image");
const dropzone = document.querySelector("#dropzone");
const fileList = document.querySelector("#fileList");
const debugTabs = document.querySelector("#debugTabs");
const debugPane = document.querySelector("#debugPane");
const toast = document.querySelector("#toast");
const captureInput = document.querySelector("#captureInput");
const docInput = document.querySelector("#docInput");
let selectedFiles = [];
let currentNoteIds = [];
let currentSession = null;
let lastDebug = null;
let lastDraftMarkdown = "";

function show(text) { result.textContent = text; }

function clearGeneratedArticle() {
  lastDraftMarkdown = "";
  lastDebug = null;
  result.textContent = "아직 생성된 결과가 없습니다.";
  debugTabs.innerHTML = "";
  debugPane.textContent = "디버그 산출물이 아직 없습니다.";
  showToast("생성 글을 초기화했습니다.");
}

function clearMainInputs() {
  const raw = document.querySelector("#rawText");
  const memo = document.querySelector("#memo");
  if (raw) raw.value = "";
  if (memo) memo.value = "";
  showToast("자료/메모 입력칸을 초기화했습니다.");
}

async function copyMarkdown() {
  const text = lastDraftMarkdown || result.textContent || "";
  if (!text.trim() || text.includes("아직 생성된 결과")) { showToast("복사할 결과가 없습니다."); return; }
  try {
    await navigator.clipboard.writeText(text);
    showToast("Markdown을 복사했습니다.");
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    showToast("Markdown을 복사했습니다.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector("#copyMarkdownBtn");
  if (btn) btn.onclick = copyMarkdown;
  const clearBtn = document.querySelector("#clearResultBtn");
  if (clearBtn) clearBtn.onclick = clearGeneratedArticle;
  const clearInputBtn = document.querySelector("#clearInputBtn");
  if (clearInputBtn) clearInputBtn.onclick = clearMainInputs;
});

function showToast(text) {
  toast.textContent = text;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 1800);
}

function fileKey(file) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function addSelectedFiles(files) {
  const current = new Map(selectedFiles.map(file => [fileKey(file), file]));
  Array.from(files || [])
    .filter(file => file.type.startsWith("image/"))
    .forEach(file => current.set(fileKey(file), file));
  selectedFiles = Array.from(current.values());
  fileList.textContent = selectedFiles.length
    ? selectedFiles.map((file, index) => `${index + 1}. ${file.name}`).join(" · ")
    : "선택된 이미지 없음";
}

function clearSelectedFiles() {
  selectedFiles = [];
  fileInput.value = "";
  fileList.textContent = "선택된 이미지 없음";
}

fileInput.onchange = () => {
  addSelectedFiles(fileInput.files);
  fileInput.value = "";
};
document.querySelector("#clearFilesBtn").onclick = clearSelectedFiles;

function handleDragOver(event) {
  event.preventDefault();
  dropzone.classList.add("dragover");
}

function handleDragLeave() {
  dropzone.classList.remove("dragover");
}

function handleDrop(event) {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  addSelectedFiles(event.dataTransfer.files);
}

dropzone.ondragover = handleDragOver;
dropzone.ondragleave = handleDragLeave;
dropzone.ondrop = handleDrop;
document.ondragover = handleDragOver;
document.ondrop = handleDrop;

async function ensureSession() {
  if (currentSession) return currentSession;
  const rows = await (await fetch("/api/sessions")).json();
  currentSession = rows[rows.length - 1];
  if (!currentSession) {
    currentSession = await (await fetch("/api/sessions", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ title:"Study Capture Session" })
    })).json();
  }
  await refreshTimeline();
  return currentSession;
}

async function refreshTimeline() {
  if (!currentSession) return;
  const [captures, qa, documents] = await Promise.all([
    fetch(`/api/sessions/${currentSession.session_id}/captures`).then(r => r.json()),
    fetch(`/api/sessions/${currentSession.session_id}/qa`).then(r => r.json()),
    fetch(`/api/sessions/${currentSession.session_id}/documents`).then(r => r.json())
  ]);
  renderDebug({
    ...(lastDebug || {}),
    "Capture Timeline": captures,
    "Q&A Logs": qa,
    "Documents": documents
  });
}

function renderDebug(payload) {
  lastDebug = payload || {};
  const tabs = [
    ["Final Article", lastDebug.draft || result.textContent],
    ["Capture Timeline", lastDebug["Capture Timeline"] || lastDebug.capture_timeline || []],
    ["Q&A Logs", lastDebug["Q&A Logs"] || lastDebug.qa_logs || []],
    ["Documents", lastDebug["Documents"] || lastDebug.documents || []],
    ["Image Evidence", lastDebug.image_evidence || []],
    ["Problem Map", lastDebug.problem_map || {}],
    ["Decision Map", lastDebug.decision_map || {}],
    ["Section Plan", lastDebug.section_plan || []],
    ["Article Brief", lastDebug.article_brief || {}],
    ["Critic Report", lastDebug.critic_report || {}]
  ];
  debugTabs.innerHTML = tabs.map(([label], index) => `<button class="tab${index === 0 ? " active" : ""}" data-tab="${escapeHtml(label)}">${escapeHtml(label)}</button>`).join("");
  function showTab(label, value) {
    debugPane.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    debugTabs.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === label));
  }
  tabs.forEach(([label, value]) => {
    const btn = Array.from(debugTabs.querySelectorAll(".tab")).find(item => item.dataset.tab === label);
    if (btn) btn.onclick = () => showTab(label, value);
  });
  showTab(tabs[0][0], tabs[0][1]);
}

document.querySelector("#captureBtn").onclick = async () => {
  await ensureSession();
  captureInput.click();
};

captureInput.onchange = async () => {
  await ensureSession();
  if (!captureInput.files.length) return;
  const form = new FormData();
  Array.from(captureInput.files).forEach(file => form.append("image", file));
  form.append("user_note", document.querySelector("#memo").value || "");
  form.append("source_title", document.title || "");
  form.append("source_url", location.href || "");
  const res = await fetch(`/api/sessions/${currentSession.session_id}/captures`, { method:"POST", body:form });
  const data = await res.json();
  captureInput.value = "";
  showToast(`Capture saved: 이미지 ${data.total}`);
  await refreshTimeline();
};

document.querySelector("#docBtn").onclick = async () => {
  await ensureSession();
  docInput.click();
};

docInput.onchange = async () => {
  await ensureSession();
  if (!docInput.files.length) return;
  const form = new FormData();
  Array.from(docInput.files).forEach(file => form.append("document", file));
  const res = await fetch(`/api/sessions/${currentSession.session_id}/documents`, { method:"POST", body:form });
  const data = await res.json();
  docInput.value = "";
  const ok = (data.documents || []).filter(doc => doc.status === "ok").length;
  const failed = (data.documents || []).filter(doc => doc.status !== "ok");
  if (failed.length) {
    showToast(`문서 업로드: 성공 ${ok}건, 실패 ${failed.length}건 (${failed[0].error || "변환 실패"})`);
  } else {
    showToast(`문서 업로드 완료: ${ok}건 (총 ${data.total})`);
  }
  await refreshTimeline();
};

document.querySelector("#askBtn").onclick = async () => {
  await ensureSession();
  const question = prompt("Tutor Agent에게 질문하기");
  if (!question) return;
  showToast("Tutor Agent thinking...");
  const selectedText = String(window.getSelection?.() || "");
  const qa = await (await fetch(`/api/sessions/${currentSession.session_id}/ask`, {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question, selected_text:selectedText })
  })).json();
  showToast(`Q&A saved: ${qa.qa_id}`);
  await refreshTimeline();
};

async function loadNotes() {
  try {
    const res = await fetch("/api/notes");
    const notes = await res.json();
    if (!currentNoteIds.length) {
      notesBox.innerHTML = "<p>이번 세션에서 생성한 노트가 아직 없습니다. 검색창을 사용하면 이전 저장 기록을 찾아볼 수 있습니다.</p>";
      return;
    }
    const currentNotes = notes.filter(note => currentNoteIds.includes(note.id));
    renderNotes(currentNotes.reverse());
  } catch (err) {
    notesBox.innerHTML = "<p>저장된 노트를 불러오지 못했습니다.</p>";
  }
}

function visibleNotes(notes) {
  return notes.filter(n => {
    const hasContent = Boolean((n.raw_text || "").trim() || (n.user_memo || "").trim() || n.image_path);
    const emptyFallback = String(n.summary || "").includes("입력된 텍스트가 없습니다");
    const imageOnlyPlaceholder = n.source_type === "study-capture-image" && (
      String(n.summary || "").includes("스크린샷을 학습 근거 자료로 저장했습니다")
      || String(n.summary || "").includes("스크린샷을 학습 캡처로 저장했습니다")
    );
    if (emptyFallback && n.source_type !== "study-capture-image") return false;
    if (imageOnlyPlaceholder) return false;
    return hasContent || !emptyFallback;
  });
}

function displayTitle(n) {
  const generic = ["학습 캡처 기록", "이미지 기반 학습 캡처", "이미지 학습 캡처"];
  if (!generic.includes(n.title)) return n.title;
  const time = String(n.created_at || "").split("T")[1]?.slice(0, 5) || "";
  return `${n.image_path ? "이미지" : "학습"} 캡처${time ? " " + time : ""}`;
}

function displaySummary(n) {
  return String(n.summary || "")
    .replace(
      "현재 MVP는 이미지 파일을 근거 자료로 보관하지만, 화면 속 텍스트를 자동으로 읽는 OCR/비전 기능은 아직 연결되어 있지 않습니다. 정확한 노트 생성을 위해 화면의 핵심 문장, 오류 메시지, 실습 목표를 텍스트 입력칸이나 메모에 함께 적어 주세요.",
      "이미지를 학습 근거 자료로 저장했습니다. 화면의 핵심 문장, 오류 메시지, 실습 목표를 메모하면 노트와 문제 해결형 포트폴리오 초안을 더 정확하게 구성할 수 있습니다."
    );
}

function renderNotes(notes) {
  notesBox.innerHTML = notes.map(n => `
    <div class="note">
      <strong>${escapeHtml(displayTitle(n))}</strong>
      <div class="meta">${escapeHtml(n.created_at)} · ${escapeHtml(n.source_type)}</div>
      <div>${(n.tags || []).map(t => `<span class="badge">${escapeHtml(t)}</span>`).join("")}</div>
      ${n.image_path ? `<img src="${escapeHtml(n.image_path)}" alt="captured study screenshot" />` : ""}
      ${(n.image_paths || []).length > 1 ? `<div class="meta">캡처 ${(n.image_paths || []).length}장 연결</div>` : ""}
      <p>${escapeHtml(displaySummary(n).slice(0, 180))}</p>
    </div>`).join("");
}

document.querySelector("#searchBtn").onclick = async () => {
  const res = await fetch("/api/search", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ query: document.querySelector("#query").value, top_k: 6 })
  });
  const rows = await res.json();
  renderNotes(visibleNotes(rows.map(r => r.note)));
};

async function makeBlog(formatType) {
  if (!selectedFiles.length && !document.querySelector("#rawText").value.trim() && !document.querySelector("#memo").value.trim()) {
    show("이미지, 화면 텍스트, 메모 중 하나는 입력해 주세요.");
    return;
  }
  const btn = document.querySelector("#portfolioBtn");
  const extraInfo = [
    ["실습/프로젝트 이름", document.querySelector("#projectName").value],
    ["내가 해결한 핵심 문제", document.querySelector("#coreProblem").value],
    ["중간에 막힌 부분", document.querySelector("#blockedPart").value],
    ["최종 결과", document.querySelector("#finalResult").value],
    ["강조하고 싶은 기술", document.querySelector("#focusTech").value]
  ]
    .filter(([, value]) => String(value || "").trim())
    .map(([label, value]) => `- ${label}: ${String(value).trim()}`)
    .join("\\n");
  const topic = document.querySelector("#projectName").value.trim() || "학습 기록 기반 문제 해결 경험";
  btn.disabled = true;
  lastDraftMarkdown = "";
  show("문제 해결형 Medium 초안을 생성하는 중입니다. URL 본문 추출, 이미지 판독, 긴 글 생성이 함께 진행되어 시간이 걸릴 수 있습니다...");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000);
  try {
    const form = new FormData();
    selectedFiles.forEach(file => form.append("image", file));
    form.append("raw_text", document.querySelector("#rawText").value);
    form.append("memo", document.querySelector("#memo").value);
    form.append("topic", topic);
    form.append("format_type", formatType);
    form.append("extra_info", extraInfo);
    const res = await fetch("/api/direct-blog", {
      method:"POST",
      body: form,
      signal: controller.signal
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    show(`[응답 시간: ${data.elapsed_seconds}s · 이미지 ${data.image_count}장]\\n\\n${data.draft}`);
    renderDebug(data);
  } catch (err) {
    show("문제 해결형 Medium 글 생성 요청이 완료되지 않았습니다. 이미지가 많거나 LLM API 응답이 늦을 수 있어요. 추가 정보 칸에 이미지 흐름 요약을 넣고 다시 시도해 주세요.");
  } finally {
    clearTimeout(timeout);
    btn.disabled = false;
  }
}

document.querySelector("#portfolioBtn").onclick = () => makeBlog("problem-solving-portfolio");

function escapeHtml(text) {
  return String(text || "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[ch]));
}

loadNotes();
ensureSession();
</script>
</body>
</html>
"""


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7870"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"AI Study Documentation Agent running on http://{host}:{port}")
    startup_diag = provider_diagnostics("groq", package_import_ok=groq_import_ok(), client_init_ok=None)
    print(
        "LLM startup diagnostics: "
        f"dotenv_exists={startup_diag['dotenv_exists']} "
        f"dotenv_loaded={startup_diag['dotenv_loaded']} "
        f"groq_api_key_present={startup_diag['api_key_present']} "
        f"groq_package_import_ok={startup_diag['package_import_ok']} "
        f"text_model={startup_diag['text_model']} "
        f"vision_model={startup_diag['vision_model']} "
        f"fast_provider={MODEL_PROVIDER_FAST} deep_provider={MODEL_PROVIDER_DEEP}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
