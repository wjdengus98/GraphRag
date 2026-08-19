import json
import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "parsed_docs.jsonl"


# =========================
# 2. Text cleanup patterns
# =========================

PAGE_NUMBER_RE = re.compile(r"^\s*-?\s*\d+\s*-?\s*$")
DOT_LEADER_RE = re.compile(r"[.\u00b7]{5,}\s*\d+\s*$")
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e]")
PRIVATE_USE_RE = re.compile(r"[\uf000-\uf8ff]")

HEADER_FOOTER_RE = re.compile(
    r"^(2026 세금절약 가이드 I|제\d+장 .+|\d+\s*$|발간등록 번호 .+|ISBN .+)$"
)

CHAPTER_RE = re.compile(r"^제\s*\d+\s*장\s+(.+)")
TOPIC_RE = re.compile(r"^\d{2}\s+(.+)")
GUIDE_TITLE_RE = re.compile(r"^\d+\.\s+(.+)")
QUESTION_RE = re.compile(r"^Q\d+\.\s+(.+)")


def get_pdf_path() -> Path:
    load_dotenv()

    configured_path = os.getenv("PDF_PATH", "data/tax_saving_guide_2026.pdf")
    pdf_path = Path(configured_path)

    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path

    return pdf_path


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_common_noise_lines(lines: list[str]) -> list[str]:
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if PAGE_NUMBER_RE.fullmatch(line):
            continue

        if DOT_LEADER_RE.search(line):
            continue

        if HEADER_FOOTER_RE.match(line):
            continue

        cleaned.append(line)

    return cleaned


def clean_tax_guide_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_RE.sub("", text)
    text = PRIVATE_USE_RE.sub("\n- ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("•", "\n- ")
    text = text.replace("●", "\n- ")
    text = text.replace("▶", "\n")
    text = text.replace("✽", "\n")

    lines = [normalize_spaces(line) for line in text.split("\n")]
    lines = remove_common_noise_lines(lines)

    return normalize_spaces("\n".join(lines))


def is_toc_or_cover(page_number: int, text: str) -> bool:
    if page_number <= 2:
        return True

    if page_number <= 10 and text.count("Contents") >= 1:
        return True

    if page_number <= 10 and len(text) > 500 and DOT_LEADER_RE.search(text):
        return True

    toc_items = re.findall(r"^\s*\d{2}\s+.+\s+\d{2,3}\s*$", text, flags=re.MULTILINE)
    if page_number <= 10 and len(toc_items) >= 3:
        return True

    return False


def infer_title(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        for pattern in (CHAPTER_RE, TOPIC_RE, GUIDE_TITLE_RE, QUESTION_RE):
            match = pattern.match(line)
            if match:
                return match.group(1).strip()

        if 8 <= len(line) <= 80 and not line.startswith("- "):
            return line

    return None


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    docs = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        raw_text = page.extract_text() or ""

        if is_toc_or_cover(page_number, raw_text):
            continue

        cleaned_text = clean_tax_guide_text(raw_text)

        if len(cleaned_text) < 30:
            continue

        docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_path.relative_to(BASE_DIR).as_posix(),
                    "page": page_index,
                    "page_number": page_number,
                    "title": infer_title(cleaned_text),
                },
            )
        )

    return docs


def save_documents_to_jsonl(docs: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            row = {
                "doc_id": f"tax-guide-page-{doc.metadata['page_number']}",
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    pdf_path = get_pdf_path()

    print("PDF path:", pdf_path)
    docs = load_pdf_pages(pdf_path)
    print("Loaded documents:", len(docs))

    save_documents_to_jsonl(docs, OUTPUT_PATH)
    print("Saved:", OUTPUT_PATH)

    print("\nPreview")
    print("=" * 80)
    if docs:
        sample = docs[0]
        print(sample.metadata)
        print(sample.page_content[:1000])


if __name__ == "__main__":
    main()
