import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# =========================
# 1. 환경 변수 / 경로 설정
# =========================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "outputs" / "chunks.jsonl"
OUTPUT_PATH = BASE_DIR / "outputs" / "extracted_kg.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

DOC_ID = "tax_saving_guide_2026"
DOC_TITLE = "2026 세금절약 가이드 I"

# 0이면 전체 chunk를 처리합니다. 테스트할 때는 1, 3처럼 작게 지정하면 됩니다.
KG_CHUNK_LIMIT = int(os.getenv("KG_CHUNK_LIMIT", "0"))
KG_RESET_OUTPUT = os.getenv("KG_RESET_OUTPUT", "false").lower() == "true"


# =========================
# 2. 지식 그래프 스키마 정의
# =========================

NodeType = Literal[
    "Tax",
    "TaxType",
    "Taxpayer",
    "BusinessType",
    "IncomeType",
    "TaxEvent",
    "Deadline",
    "Requirement",
    "Procedure",
    "Deduction",
    "Credit",
    "Exemption",
    "Penalty",
    "SupportProgram",
    "Institution",
    "Law",
    "RequiredDocument",
    "Form",
    "Amount",
    "Rate",
    "Condition",
    "Case",
    "Concept",
    "Unknown",
]

RelationshipType = Literal[
    "IS_A",
    "APPLIES_TO",
    "TRIGGERS",
    "HAS_DEADLINE",
    "HAS_REQUIREMENT",
    "HAS_PROCEDURE",
    "REQUIRES_DOCUMENT",
    "HAS_PENALTY",
    "QUALIFIES_FOR",
    "REDUCES_TAX",
    "EXEMPTS_FROM",
    "DEDUCTS",
    "CALCULATED_BY",
    "HAS_RATE",
    "HAS_AMOUNT",
    "ADMINISTERED_BY",
    "BASED_ON_LAW",
    "DEFINES",
    "RELATED_TO",
]


class KGNode(BaseModel):
    id: str = Field(
        description=(
            "고유 노드 id입니다. 반드시 '<Type>:<표준 한글 이름>' 형식으로 작성합니다. "
            "예: 'Tax:부가가치세', 'Taxpayer:간이과세자', "
            "'Deadline:부가가치세 확정신고 납부기한'"
        )
    )
    name: str = Field(description="엔티티의 표준 한글 표시 이름입니다.")
    type: NodeType


class KGRelationship(BaseModel):
    source: str = Field(description="출발 노드 id입니다. 반드시 nodes 목록 안의 id와 정확히 일치해야 합니다.")
    target: str = Field(description="도착 노드 id입니다. 반드시 nodes 목록 안의 id와 정확히 일치해야 합니다.")
    kind: RelationshipType
    evidence: str = Field(description="이 관계를 뒷받침하는 원문 근거 문장 또는 짧은 구절입니다.")


class KGGraph(BaseModel):
    nodes: list[KGNode]
    relationships: list[KGRelationship]


# =========================
# 3. chunks.jsonl 로드
# =========================

def require_env(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"필수 환경 변수가 없습니다: {name}")
    return value


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl 파일을 찾을 수 없습니다: {path}")

    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            metadata = row.get("metadata", {})

            chunks.append(
                {
                    "chunk_id": row["chunk_id"],
                    "chunk_index": row["chunk_index"],
                    "text": row["page_content"],
                    "page": metadata.get("page"),
                    "page_number": metadata.get("page_number"),
                    "source": metadata.get("source"),
                    "title": metadata.get("title"),
                }
            )

    chunks.sort(key=lambda item: item["chunk_index"])
    return chunks


def load_processed_chunk_ids(path: Path) -> set[str]:
    if KG_RESET_OUTPUT or not path.exists():
        return set()

    processed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            processed.add(json.loads(line)["chunk_id"])

    return processed


# =========================
# 4. Neo4j 연결 / 제약 조건
# =========================

def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=require_env("NEO4J_URI", NEO4J_URI),
        username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
        password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
        database=NEO4J_DATABASE,
    )


def create_constraints(graph: Neo4jGraph) -> None:
    graph.query("""
    CREATE CONSTRAINT kg_entity_id_unique IF NOT EXISTS
    FOR (e:KGEntity)
    REQUIRE e.id IS UNIQUE
    """)


# =========================
# 5. KG 추출 프롬프트
# =========================

def build_prompt(chunk: dict[str, Any]) -> str:
    return f"""
당신은 국세청 세금 가이드 문서를 바탕으로 세금 도메인 지식 그래프를 구축하는 역할입니다.

문서명:
{DOC_TITLE}

현재 프로젝트의 Neo4j 기본 구조:
- (:Document:TaxGuideDocument)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk)
- (:Chunk)-[:NEXT_CHUNK]->(:Chunk)
- 새로 추출하는 세금 엔티티는 (:KGEntity) 노드로 저장되며, 원천 Chunk와 연결됩니다.

chunk 메타데이터:
- chunk_id: {chunk["chunk_id"]}
- chunk_index: {chunk["chunk_index"]}
- page_number: {chunk.get("page_number")}
- title: {chunk.get("title")}

아래 chunk 원문에서 간결한 지식 그래프를 추출하세요.

추출 규칙:
- chunk에 명시된 사실만 사용하세요.
- 추측하지 말고, 외부 세법 지식을 추가하지 마세요.
- 목차, 발간 정보, 단순 연락처 안내만 있는 chunk라면 nodes와 relationships를 빈 목록으로 반환하세요.
- 세금, 납세자 유형, 신고·납부 기한, 요건, 절차, 공제, 세액공제, 감면, 가산세,
  지원 제도, 기관, 법령, 필요 서류, 세율, 금액, 조건처럼 질의에 도움이 되는 정보를 우선 추출하세요.
- 같은 개념은 같은 한글 이름을 사용해 하나의 노드로 재사용할 수 있게 하세요.
- 노드 id는 반드시 '<Type>:<name>' 형식이어야 합니다.
- relationship.source와 relationship.target은 반드시 nodes 안에 있는 id와 정확히 일치해야 합니다.
- evidence에는 관계를 뒷받침하는 원문 문장이나 짧은 근거 구절을 넣으세요.

노드 타입:
- Tax: 소득세, 부가가치세, 법인세, 종합부동산세처럼 이름이 있는 세금
- TaxType: 국세, 지방세, 내국세처럼 세금의 큰 분류
- Taxpayer: 사업자, 근로자, 간이과세자처럼 납세자 유형
- BusinessType: 도매업, 제조업, 법인처럼 업종 또는 사업 형태
- IncomeType: 근로소득, 사업소득처럼 소득 유형
- TaxEvent: 신고, 납부, 사업자등록, 폐업신고, 세무조사, 환급신청 같은 세무 사건
- Deadline: 신고기한, 납부기한, 과세기간 같은 기한 또는 기간
- Requirement: 의무, 요건, 준수해야 하는 조건
- Procedure: 신청 절차, 구제 절차, 처리 절차
- Deduction: 소득공제 또는 공제 항목
- Credit: 세액공제 항목
- Exemption: 감면, 면제, 비과세
- Penalty: 가산세, 체납처분, 불이익
- SupportProgram: 근로장려금, 세금포인트, 국선대리인 등 지원 제도
- Institution: 국세청, 세무서, 조세심판원 등 기관
- Law: 법령명 또는 조문
- RequiredDocument: 세금계산서, 계산서, 영수증, 신고서 등 필요한 서류
- Form: 신고서, 명세서 등 서식 또는 제출 양식
- Amount: 금액, 기준금액, 한도금액
- Rate: 세율, 가산세율, 공제율
- Condition: 예외 조건, 적용 조건, 제한 조건
- Case: 문서에 제시된 사례
- Concept: 절세, 탈세, 기장처럼 일반 세금 개념
- Unknown: 위 분류로 어렵지만 추출 가치가 있는 경우

관계 방향과 의미:
- IS_A: 하위 유형 또는 분류 관계. 예: 간이과세자 -> 사업자
- APPLIES_TO: 세금, 제도, 규칙이 납세자·소득유형·업종에 적용됨
- TRIGGERS: 사건이나 조건이 세금, 의무, 절차, 가산세를 발생시킴
- HAS_DEADLINE: 신고, 납부, 신청 등 사건에 기한이 있음
- HAS_REQUIREMENT: 세금, 납세자, 사건, 제도에 요건 또는 의무가 있음
- HAS_PROCEDURE: 세무 사건, 구제 제도, 지원 제도에 절차가 있음
- REQUIRES_DOCUMENT: 사건, 절차, 요건에 필요한 서류가 있음
- HAS_PENALTY: 위반, 체납, 미신고 등에 가산세나 불이익이 있음
- QUALIFIES_FOR: 납세자, 사업자, 조건이 지원·감면·공제·세액공제 대상이 됨
- REDUCES_TAX: 공제, 세액공제, 감면, 지원 제도가 세 부담을 줄임
- EXEMPTS_FROM: 감면 또는 면제가 세금이나 의무를 면제함
- DEDUCTS: 공제가 소득 또는 과세표준에서 차감됨
- CALCULATED_BY: 세금, 가산세, 공제, 세액공제가 금액·세율·계산식으로 계산됨
- HAS_RATE: 세금, 가산세, 공제, 세액공제에 세율 또는 비율이 있음
- HAS_AMOUNT: 규칙, 기준, 지원 제도, 가산세, 세금에 금액이 있음
- ADMINISTERED_BY: 세금, 절차, 제도를 담당 기관이 관리함
- BASED_ON_LAW: 규칙, 절차, 세금이 법령에 근거함
- DEFINES: 법령, 개념, 문단이 특정 개념을 정의함
- RELATED_TO: 위 관계로 표현하기 어렵지만 원문에 명시적인 관련성이 있음

chunk 원문:
{chunk["text"]}
"""


# =========================
# 6. LLM 추출 / 결과 검증
# =========================

def extract_kg_from_chunk(structured_llm, chunk: dict[str, Any]) -> KGGraph:
    return structured_llm.invoke(build_prompt(chunk))


def normalize_kg(kg: KGGraph) -> KGGraph:
    nodes_by_id: dict[str, KGNode] = {}

    for node in kg.nodes:
        node_id = node.id.strip()
        name = node.name.strip()
        if not node_id or not name:
            continue

        if ":" not in node_id:
            node_id = f"{node.type}:{name}"

        if not node_id.startswith(f"{node.type}:"):
            node_id = f"{node.type}:{name}"

        nodes_by_id[node_id] = KGNode(id=node_id, name=name, type=node.type)

    valid_relationships = []
    node_ids = set(nodes_by_id)
    seen_relationships = set()

    for rel in kg.relationships:
        source = rel.source.strip()
        target = rel.target.strip()
        evidence = rel.evidence.strip()

        key = (source, target, rel.kind)
        if source in node_ids and target in node_ids and key not in seen_relationships:
            valid_relationships.append(
                KGRelationship(
                    source=source,
                    target=target,
                    kind=rel.kind,
                    evidence=evidence[:500],
                )
            )
            seen_relationships.add(key)

    return KGGraph(nodes=list(nodes_by_id.values()), relationships=valid_relationships)


# =========================
# 7. Neo4j에 KG 저장
# =========================

def add_type_labels(graph: Neo4jGraph) -> None:
    for node_type in NodeType.__args__:
        graph.query(
            f"""
            MATCH (e:KGEntity {{type: $node_type}})
            SET e:{node_type}
            """,
            params={"node_type": node_type},
        )


def save_kg_to_neo4j(graph: Neo4jGraph, kg: KGGraph, chunk: dict[str, Any]) -> None:
    nodes = [node.model_dump() for node in kg.nodes]
    relationships = [rel.model_dump() for rel in kg.relationships]

    if not nodes:
        return

    graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})
        UNWIND $nodes AS node

        MERGE (e:KGEntity {id: node.id})
        SET e.name = node.name,
            e.type = node.type,
            e.last_seen_chunk_id = $chunk_id

        MERGE (c)-[:MENTIONS]->(e)
        MERGE (e)-[:SUPPORTED_BY]->(c)
        """,
        params={"chunk_id": chunk["chunk_id"], "nodes": nodes},
    )

    add_type_labels(graph)

    for rel_type in RelationshipType.__args__:
        graph.query(
            f"""
            UNWIND $relationships AS rel
            WITH rel
            WHERE rel.kind = $rel_type

            MATCH (source:KGEntity {{id: rel.source}})
            MATCH (target:KGEntity {{id: rel.target}})

            MERGE (source)-[r:{rel_type}]->(target)
            SET r.evidence = rel.evidence,
                r.last_chunk_id = $chunk_id,
                r.chunk_ids =
                    CASE
                        WHEN r.chunk_ids IS NULL THEN [$chunk_id]
                        WHEN NOT $chunk_id IN r.chunk_ids THEN r.chunk_ids + [$chunk_id]
                        ELSE r.chunk_ids
                    END
            """,
            params={
                "relationships": relationships,
                "rel_type": rel_type,
                "chunk_id": chunk["chunk_id"],
            },
        )


# =========================
# 8. 추출 결과 파일 저장
# =========================

def reset_output_if_requested(path: Path) -> None:
    if KG_RESET_OUTPUT and path.exists():
        path.unlink()


def append_result_to_jsonl(output_path: Path, chunk: dict[str, Any], kg: KGGraph) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk["chunk_index"],
        "page_number": chunk.get("page_number"),
        "title": chunk.get("title"),
        "kg": kg.model_dump(),
    }

    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(graph: Neo4jGraph) -> None:
    result = graph.query(
        """
        MATCH (e:KGEntity)
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
        OPTIONAL MATCH (e)-[r]->(target:KGEntity)
        RETURN
            count(DISTINCT e) AS entities,
            count(DISTINCT c) AS supporting_chunks,
            count(DISTINCT r) AS kg_relationships
        """
    )[0]

    print("KG 엔티티 수:", result["entities"])
    print("근거 chunk 수:", result["supporting_chunks"])
    print("KG 관계 수:", result["kg_relationships"])


# =========================
# 9. 실행
# =========================

def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    reset_output_if_requested(OUTPUT_PATH)
    processed_chunk_ids = load_processed_chunk_ids(OUTPUT_PATH)

    chunks_to_process = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in processed_chunk_ids
    ]

    if KG_CHUNK_LIMIT > 0:
        chunks_to_process = chunks_to_process[:KG_CHUNK_LIMIT]

    print("전체 chunk 수:", len(chunks))
    print("이미 처리한 chunk 수:", len(processed_chunk_ids))
    print("이번에 처리할 chunk 수:", len(chunks_to_process))

    graph = get_graph()
    create_constraints(graph)

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(KGGraph, method="json_schema")

    total_nodes = 0
    total_relationships = 0

    for i, chunk in enumerate(chunks_to_process, start=1):
        print("=" * 80)
        print(
            f"[{i}/{len(chunks_to_process)}] "
            f"chunk_index={chunk['chunk_index']} page={chunk.get('page_number')}"
        )
        print(chunk["text"][:180].replace("\n", " "))

        kg = normalize_kg(extract_kg_from_chunk(structured_llm, chunk))

        print("추출 노드 수:", len(kg.nodes))
        print("추출 관계 수:", len(kg.relationships))

        save_kg_to_neo4j(graph, kg, chunk)
        append_result_to_jsonl(OUTPUT_PATH, chunk, kg)

        total_nodes += len(kg.nodes)
        total_relationships += len(kg.relationships)

    graph.refresh_schema()

    print("\n완료")
    print("이번 실행에서 추출한 노드 수:", total_nodes)
    print("이번 실행에서 추출한 관계 수:", total_relationships)
    print("결과 저장 경로:", OUTPUT_PATH)
    print_summary(graph)

    print("\nGraph schema:")
    print(graph.schema)


if __name__ == "__main__":
    main()
