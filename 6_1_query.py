import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI


# =========================
# 1. 환경 변수 / Neo4j 연결
# =========================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")


graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

graph.refresh_schema()
print(graph.schema)

# =========================
# 2. 현재 KG 상태 확인용 쿼리
# =========================

def print_kg_summary() -> None:
    summary = graph.query(
        """
        MATCH (e:KGEntity)
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
        OPTIONAL MATCH (e)-[r]->(target:KGEntity)
        RETURN
            count(DISTINCT e) AS entities,
            count(DISTINCT c) AS chunks,
            count(DISTINCT r) AS relationships
        """
    )[0]

    print("KG 엔티티 수:", summary["entities"])
    print("근거 chunk 수:", summary["chunks"])
    print("KG 관계 수:", summary["relationships"])


def print_sample_entities() -> None:
    rows = graph.query(
        """
        MATCH (e:KGEntity)
        RETURN e.type AS type, e.name AS name
        ORDER BY e.type, e.name
        LIMIT 20
        """
    )

    print("\n엔티티 샘플")
    print("=" * 80)
    for row in rows:
        print(f"[{row['type']}] {row['name']}")


# =========================
# 3. 자연어 질문
# =========================

questions = [
    "사업자등록을 하지 않으면 어떤 불이익이 있나요?",
    "간이과세자가 되려면 어떤 조건이 필요한가요?",
    "부가가치세 신고 납부기한은 어떻게 되나요?",
    "세금을 체납하면 어떤 제재를 받을 수 있나요?",
    "억울한 세금이 있을 때 어떤 권리구제 절차를 이용할 수 있나요?",
    "국세와 지방세에는 어떤 세금들이 포함되나요?",
]


# 질문을 바꿔보고 싶으면 아래 인덱스만 변경하면 됩니다.
QUESTION_INDEX = 0
question = questions[QUESTION_INDEX]


# =========================
# 4. Cypher 생성 프롬프트
# =========================

CYPHER_GENERATION_TEMPLATE = """
당신은 Neo4j Cypher 전문가입니다.
사용자의 질문을 보고, 아래 그래프 스키마에 맞는 Cypher 쿼리만 생성하세요.

스키마:
{schema}

중요 규칙:
- 설명하지 말고 Cypher 쿼리만 반환하세요.
- 읽기 전용 MATCH 쿼리만 작성하세요.
- KG 지식은 (:KGEntity)와 그 하위 라벨(Tax, TaxEvent, Deadline, Penalty 등)에 들어 있습니다.
- 원문 근거는 관계의 evidence 속성 또는 (:KGEntity)-[:SUPPORTED_BY]->(:Chunk)의 Chunk.text에 있습니다.
- 어떤 개념을 찾을 때는 e.name CONTAINS "검색어" 패턴을 우선 사용하세요.
- 여러 관계 타입을 variable length로 나열할 때는 콜론을 한 번만 사용하세요.
  올바른 예: -[:HAS_REQUIREMENT|RELATED_TO|TRIGGERS*1..3]->
  잘못된 예: -[:HAS_REQUIREMENT|:RELATED_TO|:TRIGGERS*1..3]->
- 이 예제 코드에서는 variable length 관계를 사용하지 마세요. 즉 *1..2, *1..3 같은 문법을 쓰지 마세요.
- 관계 변수 r을 반환할 때는 반드시 단일 관계 패턴만 사용하세요.
  올바른 예: MATCH (e:KGEntity)-[r:HAS_PENALTY|RELATED_TO]->(target:KGEntity)
  잘못된 예: MATCH (e:KGEntity)-[r:HAS_PENALTY|RELATED_TO*1..2]->(target:KGEntity)
- 가능하면 evidence를 함께 반환하세요.
- 결과는 20개 이하로 제한하세요.

자주 쓰는 패턴:
1. 특정 개념 주변 관계:
MATCH (e:KGEntity)-[r]-(target:KGEntity)
WHERE e.name CONTAINS "사업자등록"
RETURN e.name AS source, type(r) AS relationship, target.name AS target, r.evidence AS evidence
LIMIT 20

2. 특정 개념을 언급한 원문 chunk:
MATCH (c:Chunk)-[:MENTIONS]->(e:KGEntity)
WHERE e.name CONTAINS "사업자등록"
RETURN e.name AS entity, c.page_number AS page, c.title AS title, c.text AS text
LIMIT 5

질문:
{question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)


# =========================
# 5. GraphCypherQAChain 실행
# =========================

chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(model=OPENAI_MODEL, temperature=0),
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    validate_cypher=True,
    allow_dangerous_requests=True,
)


def main() -> None:
    print_kg_summary()
    print_sample_entities()

    print("\n질문")
    print("=" * 80)
    print(question)

    result = chain.invoke({"query": question})

    print("\n답변")
    print("=" * 80)
    print(result["result"])


if __name__ == "__main__":
    main()
