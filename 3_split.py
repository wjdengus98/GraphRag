import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "outputs" / "parsed_docs.jsonl"
OUTPUT_PATH = BASE_DIR / "outputs" / "chunks.jsonl"
MERGE_SMALL_CHARS = 250
MAX_MERGED_CHARS = 1100


# =========================
# 2. JSONL load
# =========================

def load_parsed_documents(input_path: Path) -> list[Document]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    docs = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            page_content = row["page_content"]
            metadata = dict(row.get("metadata", {}))

            if is_section_toc_document(page_content, metadata):
                continue

            metadata["parent_doc_id"] = row["doc_id"]

            docs.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

    return docs


def is_section_toc_document(text: str, metadata: dict) -> bool:
    if metadata.get("title") != "세금절약 가이드 I":
        return False

    toc_lines = re.findall(r"^\s*(?:\d{2}|Q\d+|◆|납세자가).+\s+\d{2,3}\s*$", text, re.MULTILINE)
    return len(toc_lines) >= 3


# =========================
# 3. Chunk IDs
# =========================

def make_chunk_id(chunk: Document, page_chunk_index: int) -> str:
    source = chunk.metadata.get("source", "unknown")
    parent_doc_id = chunk.metadata.get("parent_doc_id", "unknown")
    start_index = chunk.metadata.get("start_index", 0)
    text_head = chunk.page_content[:120]

    raw = f"{source}:{parent_doc_id}:{page_chunk_index}:{start_index}:{text_head}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    return f"chunk-{digest}"


def normalize_chunk_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def merge_small_chunks(chunks: list[Document]) -> list[Document]:
    merged: list[Document] = []

    for chunk in chunks:
        text = normalize_chunk_text(chunk.page_content)

        if len(text) < 40:
            continue

        chunk.page_content = text

        if not merged:
            merged.append(chunk)
            continue

        prev = merged[-1]
        same_parent = (
            prev.metadata.get("parent_doc_id") == chunk.metadata.get("parent_doc_id")
        )
        combined_text = f"{prev.page_content}\n{chunk.page_content}"

        if same_parent and len(text) < MERGE_SMALL_CHARS and len(combined_text) <= MAX_MERGED_CHARS:
            prev.page_content = combined_text
            prev.metadata["merged_small_chunk"] = True
            continue

        if same_parent and len(prev.page_content) < MERGE_SMALL_CHARS and len(combined_text) <= MAX_MERGED_CHARS:
            prev.page_content = combined_text
            prev.metadata["merged_small_chunk"] = True
            continue

        merged.append(chunk)

    return merged


# =========================
# 4. Split
# =========================

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        keep_separator=True,
        separators=[
            "\n\n",
            "\nGuide ",
            "\n사례로 보는 세금 절약 Guide",
            "\n납세자가 자주 묻는 상담사례",
            "\n관련 법규",
            "\nㅣ",
            "\n- ",
            "\nQ",
            "\n①",
            "\n②",
            "\n③",
            "\n④",
            "\n⑤",
            "\n1)",
            "\n2)",
            "\n3)",
            "\n4)",
            "\n5)",
            "\n1.",
            "\n2.",
            "\n3.",
            "\n4.",
            "\n5.",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    split_chunks = merge_small_chunks(splitter.split_documents(docs))
    page_counts: defaultdict[str, int] = defaultdict(int)
    chunks = []

    for chunk in split_chunks:
        text = normalize_chunk_text(chunk.page_content)

        if len(text) < 40:
            continue

        parent_doc_id = chunk.metadata.get("parent_doc_id", "unknown")
        page_chunk_index = page_counts[parent_doc_id]
        page_counts[parent_doc_id] += 1

        chunk.page_content = text
        chunk.metadata["page_chunk_index"] = page_chunk_index
        chunk.metadata["chunk_index"] = len(chunks)
        chunk.metadata["chunk_id"] = make_chunk_id(chunk, page_chunk_index)
        chunk.metadata["char_count"] = len(text)

        chunks.append(chunk)

    return chunks


# =========================
# 5. JSONL save
# =========================

def save_chunks_to_jsonl(chunks: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata["chunk_index"],
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 6. Run
# =========================

def main() -> None:
    docs = load_parsed_documents(INPUT_PATH)
    print("Loaded documents:", len(docs))

    chunks = chunk_documents(docs)
    print("Created chunks:", len(chunks))

    save_chunks_to_jsonl(chunks, OUTPUT_PATH)
    print("Saved:", OUTPUT_PATH)

    print("\nPreview")
    print("=" * 80)
    for chunk in chunks[:3]:
        print("chunk_id:", chunk.metadata["chunk_id"])
        print("page_number:", chunk.metadata.get("page_number"))
        print("page_chunk_index:", chunk.metadata["page_chunk_index"])
        print("start_index:", chunk.metadata.get("start_index"))
        print("char_count:", chunk.metadata["char_count"])
        print("title:", chunk.metadata.get("title"))
        print(chunk.page_content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
