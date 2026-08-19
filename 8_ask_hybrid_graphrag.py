import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ============================================================
# 1. 환경 변수 / 인덱스 설정
# ============================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

DOCUMENT_ID = "tax_saving_guide_2026"
VECTOR_INDEX_NAME = "tax_chunk_vector_index"
KEYWORD_INDEX_NAME = "tax_chunk_keyword_index"


def require_env(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"필수 환경 변수가 없습니다: {name}")
    return value


# ============================================================
# 2. Neo4j / Embedding / LLM 준비
# ============================================================

graph = Neo4jGraph(
    url=require_env("NEO4J_URI", NEO4J_URI),
    username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
    password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
    database=NEO4J_DATABASE,
)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=require_env("NEO4J_URI", NEO4J_URI),
    username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
    password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
    database=NEO4J_DATABASE,
    index_name=VECTOR_INDEX_NAME,
    keyword_index_name=KEYWORD_INDEX_NAME,
    search_type="hybrid",
)


# ============================================================
# 3. 질문 세트
# ============================================================

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
        "keywords": ["부가가치세", "신고", "납부기한", "확정신고"],
    },
    {
        "question": "세금을 체납하면 어떤 제재를 받을 수 있나요?",
        "keywords": ["체납", "가산세", "강제징수", "출국금지", "명단공개"],
    },
    {
        "question": "억울한 세금이 있을 때 어떤 권리구제 절차를 이용할 수 있나요?",
        "keywords": ["권리구제", "이의신청", "심사청구", "심판청구", "행정소송"],
    },
]


# 질문을 바꿔보고 싶으면 아래 인덱스만 변경하면 됩니다.
QUESTION_INDEX = 3


# ============================================================
# 4. Vector + Keyword 검색
# ============================================================

def search_vector_keyword(question: str, k: int = 5):
    return vector_store.similarity_search_with_score(question, k=k)


def format_vector_keyword_context(results) -> str:
    if not results:
        return "Vector + Keyword 검색 결과가 없습니다."

    formatted = []

    for i, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata
        formatted.append(
            f"""
[Hybrid 검색 결과 {i}]
score: {score}
page: {metadata.get("page_number")}
chunk_index: {metadata.get("chunk_index")}
title: {metadata.get("title")}
source: {metadata.get("source")}
text:
{doc.page_content}
"""
        )

    return "\n".join(formatted)


# ============================================================
# 5. 고정 Cypher 기반 Graph 검색
# ============================================================

def search_graph_relations_by_keywords(
    keywords: list[str],
    limit: int = 25,
) -> list[dict[str, Any]]:
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


def search_graph_chunks_by_keywords(
    keywords: list[str],
    limit: int = 6,
) -> list[dict[str, Any]]:
    return graph.query(
        """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE any(keyword IN $keywords WHERE c.text CONTAINS keyword)
        WITH c,
             size([keyword IN $keywords WHERE c.text CONTAINS keyword]) AS score
        RETURN
            c.page_number AS page,
            c.chunk_index AS chunk_index,
            c.title AS title,
            c.text AS text,
            score
        ORDER BY score DESC, c.chunk_index ASC
        LIMIT $limit
        """,
        params={
            "document_id": DOCUMENT_ID,
            "keywords": keywords,
            "limit": limit,
        },
    )


def format_graph_context(
    graph_chunks: list[dict[str, Any]],
    graph_relations: list[dict[str, Any]],
) -> str:
    chunk_lines = []
    for row in graph_chunks:
        chunk_lines.append(
            f"""
[Graph 원문 근거]
page: {row.get("page")}
chunk_index: {row.get("chunk_index")}
score: {row.get("score")}
title: {row.get("title")}
text:
{row.get("text")}
"""
        )

    relation_lines = []
    for row in graph_relations:
        relation_lines.append(
            (
                f"- {row.get('source')} ({row.get('source_type')}) "
                f"-[{row.get('relationship')}]-> "
                f"{row.get('target')} ({row.get('target_type')})\n"
                f"  evidence: {row.get('evidence')}"
            )
        )

    return f"""
[Graph 원문 chunk 검색 결과]
{chr(10).join(chunk_lines) if chunk_lines else "검색된 원문 chunk가 없습니다."}

[Graph KG 관계 검색 결과]
{chr(10).join(relation_lines) if relation_lines else "검색된 KG 관계가 없습니다."}
"""


# ============================================================
# 6. 최종 답변 생성 프롬프트
# ============================================================

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
너는 국세청 세금절약 가이드 기반 Hybrid GraphRAG assistant다.

너에게는 두 종류의 검색 결과가 제공된다.

1. Vector + Keyword 검색 결과
   - 질문과 의미적으로 유사하거나 키워드가 일치하는 Chunk
   - 원문 근거 확인에 유용하다.

2. Graph 검색 결과
   - 고정 Cypher로 조회한 관련 Chunk, KGEntity 관계, evidence
   - 세금 개념 사이의 관계와 원문 근거 확인에 유용하다.

답변 규칙:
- 반드시 제공된 검색 결과에 근거해서만 답변하라.
- 검색 결과에 없는 내용은 추측하지 말고 근거가 부족하다고 말하라.
- 한국어로 답변하라.
- 사용자가 이해하기 쉽게 핵심부터 말하라.
- 가능한 경우 PDF page 번호를 함께 제시하라.
- 세무 대리나 법률 자문처럼 단정하지 말고, 문서 안내 기준으로 설명하라.
""",
        ),
        (
            "human",
            """
질문:
{question}

[1] Vector + Keyword 검색 context:
{vector_keyword_context}

[2] Graph 검색 context:
{graph_context}
""",
        ),
    ]
)


# ============================================================
# 7. Hybrid GraphRAG 질의응답 함수
# ============================================================

def answer_question(
    question: str,
    keywords: list[str],
    hybrid_k: int = 5,
    show_context: bool = False,
) -> str:
    vector_keyword_results = search_vector_keyword(question, k=hybrid_k)
    vector_keyword_context = format_vector_keyword_context(vector_keyword_results)

    graph_chunks = search_graph_chunks_by_keywords(keywords)
    graph_relations = search_graph_relations_by_keywords(keywords)
    graph_context = format_graph_context(graph_chunks, graph_relations)

    if show_context:
        print("\n" + "-" * 80)
        print("[Vector + Keyword 검색 context]")
        print("-" * 80)
        print(vector_keyword_context)

        print("\n" + "-" * 80)
        print("[Graph 검색 context]")
        print("-" * 80)
        print(graph_context)

    messages = final_prompt.format_messages(
        question=question,
        vector_keyword_context=vector_keyword_context,
        graph_context=graph_context,
    )

    response = llm.invoke(messages)
    return response.content


# ============================================================
# 8. 실행
# ============================================================

def main() -> None:
    selected = questions[QUESTION_INDEX]
    question = selected["question"]
    keywords = selected["keywords"]

    print("=" * 80)
    print("질문:", question)
    print("검색 키워드:", ", ".join(keywords))

    answer = answer_question(
        question=question,
        keywords=keywords,
        hybrid_k=5,
        show_context=False,
    )

    print("\n최종 답변:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
