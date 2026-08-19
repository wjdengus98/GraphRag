import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings


# ============================================================
# 1. 환경 변수 / 인덱스 설정
# ============================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

DOCUMENT_ID = "tax_saving_guide_2026"
VECTOR_INDEX_NAME = "tax_chunk_vector_index"
KEYWORD_INDEX_NAME = "tax_chunk_keyword_index"


def require_env(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"필수 환경 변수가 없습니다: {name}")
    return value


# ============================================================
# 2. Neo4j 연결
# ============================================================

graph = Neo4jGraph(
    url=require_env("NEO4J_URI", NEO4J_URI),
    username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
    password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
    database=NEO4J_DATABASE,
)


# ============================================================
# 3. Chunk 데이터 확인
# ============================================================

def print_chunk_summary() -> None:
    rows = graph.query(
        """
        MATCH (c:Chunk {document_id: $document_id})
        RETURN
            count(c) AS total_chunks,
            count(c.embedding) AS embedded_chunks
        """,
        params={"document_id": DOCUMENT_ID},
    )

    summary = rows[0]
    print("전체 Chunk 수:", summary["total_chunks"])
    print("이미 embedding이 있는 Chunk 수:", summary["embedded_chunks"])

    if summary["total_chunks"] == 0:
        raise ValueError(
            "Neo4j에 Chunk가 없습니다. 먼저 4_ingest_chunks_to_neo4j_.py를 실행하세요."
        )


# ============================================================
# 4. Hybrid Vector Store 생성
# ============================================================

def build_vector_store() -> Neo4jVector:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=require_env("NEO4J_URI", NEO4J_URI),
        username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
        password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
        database=NEO4J_DATABASE,
        index_name=VECTOR_INDEX_NAME,
        keyword_index_name=KEYWORD_INDEX_NAME,
        node_label="Chunk",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        search_type="hybrid",
    )


# ============================================================
# 5. 인덱스 / embedding 상태 확인
# ============================================================

def print_index_summary() -> None:
    graph.query("CALL db.awaitIndexes(30)")

    indexes = graph.query(
        """
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE name IN [$vector_index_name, $keyword_index_name]
        RETURN name, type, labelsOrTypes, properties
        ORDER BY name
        """,
        params={
            "vector_index_name": VECTOR_INDEX_NAME,
            "keyword_index_name": KEYWORD_INDEX_NAME,
        },
    )

    print("\n생성된 인덱스:")
    for row in indexes:
        print(row)

    rows = graph.query(
        """
        MATCH (c:Chunk {document_id: $document_id})
        RETURN
            count(c) AS total_chunks,
            count(c.embedding) AS embedded_chunks
        """,
        params={"document_id": DOCUMENT_ID},
    )

    print("\nEmbedding 저장 확인:")
    print(rows[0])


# ============================================================
# 6. 간단 Hybrid 검색 테스트
# ============================================================

def test_hybrid_search(vector_store: Neo4jVector) -> None:
    queries = [
        "사업자등록을 하지 않으면 어떤 불이익이 있나요?",
        "간이과세자가 되려면 어떤 조건이 필요한가요?",
        "부가가치세 신고 납부기한은 어떻게 되나요?",
    ]

    for query in queries:
        print("\n검색 테스트:", query)
        results = vector_store.similarity_search_with_score(query, k=3)

        for doc, score in results:
            print("=" * 80)
            print("score:", score)
            print("metadata:", doc.metadata)
            print(doc.page_content[:500])


# ============================================================
# 7. 실행
# ============================================================

def main() -> None:
    print("Embedding model:", EMBEDDING_MODEL)
    print("Vector index:", VECTOR_INDEX_NAME)
    print("Keyword index:", KEYWORD_INDEX_NAME)

    print_chunk_summary()

    vector_store = build_vector_store()

    print_index_summary()
    test_hybrid_search(vector_store)


if __name__ == "__main__":
    main()
