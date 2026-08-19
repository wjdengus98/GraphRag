import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph


# =========================
# 1. Environment / paths
# =========================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "outputs" / "chunks.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DOC_ID = "tax_saving_guide_2026"
DOC_TITLE = "2026 세금절약 가이드 I"
DOC_SOURCE = "data/tax_saving_guide_2026.pdf"
DOC_PUBLISHER = "국세청"
DOC_YEAR = 2026

BATCH_SIZE = 100


# =========================
# 2. chunks.jsonl load
# =========================

def require_env(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {path}")

    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            metadata = row.get("metadata", {})

            page = metadata.get("page")
            page_number = metadata.get("page_number")
            if page_number is None:
                page_number = page + 1 if isinstance(page, int) else 0

            chunk = {
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "text": row["page_content"],
                "source": metadata.get("source", DOC_SOURCE),
                "page": page,
                "page_number": page_number,
                "page_chunk_index": metadata.get("page_chunk_index"),
                "start_index": metadata.get("start_index"),
                "char_count": metadata.get("char_count", len(row["page_content"])),
                "parent_doc_id": metadata.get("parent_doc_id"),
                "title": metadata.get("title"),
            }

            chunks.append(chunk)

    chunks.sort(key=lambda item: item["chunk_index"])
    return chunks


# =========================
# 3. Neo4j connection
# =========================

def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=require_env("NEO4J_URI", NEO4J_URI),
        username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
        password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
        database=NEO4J_DATABASE,
    )


# =========================
# 4. Constraints
# =========================

def create_constraints(graph: Neo4jGraph) -> None:
    graph.query("""
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (d:Document)
    REQUIRE d.id IS UNIQUE
    """)

    graph.query("""
    CREATE CONSTRAINT page_id_unique IF NOT EXISTS
    FOR (p:Page)
    REQUIRE p.id IS UNIQUE
    """)

    graph.query("""
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE c.id IS UNIQUE
    """)


# =========================
# 5. Document / Page / Chunk save
# =========================

def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def save_document(graph: Neo4jGraph) -> None:
    graph.query(
        """
        MERGE (d:Document {id: $doc_id})
        SET d:TaxGuideDocument,
            d.title = $doc_title,
            d.source = $doc_source,
            d.publisher = $doc_publisher,
            d.year = $doc_year
        """,
        params={
            "doc_id": DOC_ID,
            "doc_title": DOC_TITLE,
            "doc_source": DOC_SOURCE,
            "doc_publisher": DOC_PUBLISHER,
            "doc_year": DOC_YEAR,
        },
    )


def save_chunks(graph: Neo4jGraph, chunks: list[dict[str, Any]]) -> None:
    query = """
    MATCH (d:Document {id: $doc_id})
    UNWIND $chunks AS row

    MERGE (p:Page {id: $doc_id + ":page:" + toString(row.page_number)})
    SET p.page = row.page,
        p.page_number = row.page_number,
        p.source = row.source,
        p.parent_doc_id = row.parent_doc_id,
        p.title = row.title

    MERGE (c:Chunk {id: row.chunk_id})
    SET c.text = row.text,
        c.chunk_index = row.chunk_index,
        c.source = row.source,
        c.page = row.page,
        c.page_number = row.page_number,
        c.page_chunk_index = row.page_chunk_index,
        c.start_index = row.start_index,
        c.char_count = row.char_count,
        c.parent_doc_id = row.parent_doc_id,
        c.title = row.title,
        c.document_id = $doc_id

    MERGE (d)-[:HAS_PAGE]->(p)
    MERGE (p)-[:HAS_CHUNK]->(c)
    """

    for batch in batched(chunks, BATCH_SIZE):
        graph.query(query, params={"doc_id": DOC_ID, "chunks": batch})


# =========================
# 6. Chunk order relationships
# =========================

def create_next_chunk_relationships(graph: Neo4jGraph) -> None:
    graph.query(
        """
        MATCH (c:Chunk {document_id: $doc_id})
        WITH c
        ORDER BY c.chunk_index ASC
        WITH collect(c) AS chunks

        UNWIND range(0, size(chunks) - 2) AS i
        WITH chunks[i] AS current_chunk,
             chunks[i + 1] AS next_chunk

        MERGE (current_chunk)-[:NEXT_CHUNK]->(next_chunk)
        """,
        params={"doc_id": DOC_ID},
    )


# =========================
# 7. Summary
# =========================

def print_summary(graph: Neo4jGraph) -> None:
    result = graph.query(
        """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[next:NEXT_CHUNK]->(:Chunk {document_id: $doc_id})
        RETURN
            count(DISTINCT d) AS documents,
            count(DISTINCT p) AS pages,
            count(DISTINCT c) AS chunks,
            count(DISTINCT next) AS next_chunk_relationships
        """,
        params={"doc_id": DOC_ID},
    )

    row = result[0]
    print("Documents:", row["documents"])
    print("Pages:", row["pages"])
    print("Chunks:", row["chunks"])
    print("NEXT_CHUNK relationships:", row["next_chunk_relationships"])


# =========================
# 8. Run
# =========================

def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    print("Loaded chunks:", len(chunks))

    graph = get_graph()

    create_constraints(graph)
    save_document(graph)
    save_chunks(graph, chunks)
    create_next_chunk_relationships(graph)

    graph.refresh_schema()

    print("\nIngest complete")
    print_summary(graph)

    print("\nGraph schema:")
    print(graph.schema)


if __name__ == "__main__":
    main()
