import os
from typing import Any

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
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

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# =========================
# 2. 질문 세트
# =========================

questions = [
    {
        "question": "사업자등록을 하지 않으면 어떤 불이익이 있나요?",
        "keywords": ["사업자등록", "가산세", "매입세액"],
    },
    {
        "question": "간이과세자가 되려면 어떤 조건이 필요한가요?",
        "keywords": ["간이과세", "간이과세자", "1억 400만 원", "4,800만 원"],
    },
    {
        "question": "부가가치세 신고 납부기한은 어떻게 되나요?",
        "keywords": ["부가가치세", "신고", "납부기한"],
    },
    {
        "question": "세금을 체납하면 어떤 제재를 받을 수 있나요?",
        "keywords": ["체납", "가산세", "강제징수", "출국금지"],
    },
    {
        "question": "억울한 세금이 있을 때 어떤 권리구제 절차를 이용할 수 있나요?",
        "keywords": ["권리구제", "이의신청", "심사청구", "심판청구", "행정소송"],
    },
    {
        "question": "국세와 지방세에는 어떤 세금들이 포함되나요?",
        "keywords": ["국세", "지방세", "소득세", "부가가치세", "취득세"],
    },
]


# 질문을 바꿔보고 싶으면 아래 인덱스만 변경하면 됩니다.
QUESTION_INDEX = 1


# =========================
# 3. 고정 Cypher 조회
# =========================

def search_chunks_by_keywords(keywords: list[str], limit: int = 6) -> list[dict[str, Any]]:
    return graph.query(
        """
        MATCH (c:Chunk)
        WHERE any(keyword IN $keywords WHERE c.text CONTAINS keyword)
        WITH c,
             size([keyword IN $keywords WHERE c.text CONTAINS keyword]) AS score
        RETURN
            c.chunk_index AS chunk_index,
            c.page_number AS page,
            c.title AS title,
            c.text AS text,
            score
        ORDER BY score DESC, c.chunk_index ASC
        LIMIT $limit
        """,
        params={"keywords": keywords, "limit": limit},
    )


def search_kg_relations_by_keywords(keywords: list[str], limit: int = 20) -> list[dict[str, Any]]:
    return graph.query(
        """
        MATCH (source:KGEntity)-[r]->(target:KGEntity)
        WHERE type(r) <> 'SUPPORTED_BY'
          AND (
            any(keyword IN $keywords WHERE source.name CONTAINS keyword)
            OR any(keyword IN $keywords WHERE target.name CONTAINS keyword)
            OR any(keyword IN $keywords WHERE r.evidence CONTAINS keyword)
          )
        RETURN
            source.name AS source,
            source.type AS source_type,
            type(r) AS relationship,
            target.name AS target,
            target.type AS target_type,
            r.evidence AS evidence,
            r.chunk_ids AS chunk_ids
        LIMIT $limit
        """,
        params={"keywords": keywords, "limit": limit},
    )


# =========================
# 4. 답변 생성
# =========================

def format_context(chunks: list[dict[str, Any]], relations: list[dict[str, Any]]) -> str:
    chunk_lines = []
    for row in chunks:
        chunk_lines.append(
            "\n".join(
                [
                    f"- page: {row.get('page')}",
                    f"  title: {row.get('title')}",
                    f"  chunk_index: {row.get('chunk_index')}",
                    f"  text: {row.get('text')}",
                ]
            )
        )

    relation_lines = []
    for row in relations:
        relation_lines.append(
            (
                f"- {row.get('source')} ({row.get('source_type')}) "
                f"-[{row.get('relationship')}]-> "
                f"{row.get('target')} ({row.get('target_type')})\n"
                f"  evidence: {row.get('evidence')}"
            )
        )

    return f"""
[관련 원문 chunk]
{chr(10).join(chunk_lines) if chunk_lines else "검색된 원문 chunk가 없습니다."}

[관련 KG 관계]
{chr(10).join(relation_lines) if relation_lines else "검색된 KG 관계가 없습니다."}
"""


def answer_question(question: str, keywords: list[str]) -> str:
    chunks = search_chunks_by_keywords(keywords)
    relations = search_kg_relations_by_keywords(keywords)
    context = format_context(chunks, relations)

    prompt = f"""
당신은 국세청 세금절약 가이드 문서를 바탕으로 답변하는 세금 안내 도우미입니다.

규칙:
- 아래 제공된 원문 chunk와 KG 관계에 근거해서만 답변하세요.
- 근거가 부족하면 부족하다고 말하세요.
- 답변은 한국어로 간결하게 작성하세요.
- 가능한 경우 PDF page 번호를 함께 언급하세요.
- 법률·세무 자문처럼 단정하지 말고, 문서에 적힌 안내 범위에서 설명하세요.

질문:
{question}

근거:
{context}
"""

    response = llm.invoke(prompt)
    return response.content


# =========================
# 5. 실행
# =========================

def main() -> None:
    selected = questions[QUESTION_INDEX]
    question = selected["question"]
    keywords = selected["keywords"]

    print("질문")
    print("=" * 80)
    print(question)
    print("검색 키워드:", ", ".join(keywords))

    chunks = search_chunks_by_keywords(keywords)
    relations = search_kg_relations_by_keywords(keywords)

    print("\n조회된 원문 chunk")
    print("=" * 80)
    for row in chunks:
        print(
            f"page={row['page']} chunk_index={row['chunk_index']} "
            f"score={row['score']} title={row['title']}"
        )

    print("\n조회된 KG 관계")
    print("=" * 80)
    for row in relations[:10]:
        print(
            f"{row['source']} -[{row['relationship']}]-> {row['target']} "
            f"/ 근거: {row['evidence']}"
        )

    answer = answer_question(question, keywords)

    print("\n답변")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
