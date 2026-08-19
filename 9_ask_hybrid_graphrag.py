import os

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


# ============================================================
# 3. Graph Expansion Retrieval Query
# ------------------------------------------------------------
# 8번 파일과 다른 핵심입니다.
#
# 8번:
#   Vector+Keyword 검색과 Graph 검색을 코드에서 따로 실행한 뒤 합칩니다.
#
# 9번:
#   Neo4jVector가 Hybrid Search로 찾은 Chunk를 node 변수로 넘겨주면,
#   retrieval_query 안에서 그 Chunk 주변 그래프를 바로 확장합니다.
#
# 확장하는 정보:
#   - 검색된 Chunk 원문
#   - 이전/다음 Chunk
#   - Chunk가 언급한 KGEntity
#   - KGEntity 주변 관계와 evidence
#
# Neo4jVector의 retrieval_query는 반드시 text, score, metadata를 반환해야 합니다.
# ============================================================

retrieval_query = """
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(node)
OPTIONAL MATCH (node)-[:NEXT_CHUNK]->(next:Chunk)
OPTIONAL MATCH (node)-[:MENTIONS]->(entity:KGEntity)
OPTIONAL MATCH (entity)-[rel]->(target:KGEntity)
WHERE type(rel) <> 'SUPPORTED_BY'

WITH
    node,
    score,
    prev,
    next,
    collect(DISTINCT {
        name: entity.name,
        type: entity.type
    }) AS entities,
    collect(DISTINCT {
        source: entity.name,
        source_type: entity.type,
        relationship: type(rel),
        target: target.name,
        target_type: target.type,
        evidence: rel.evidence
    }) AS relationships

RETURN
    {
        chunk_text: node.text,
        previous_chunk_text: prev.text,
        next_chunk_text: next.text,
        entities: entities,
        relationships: relationships
    } AS text,
    score,
    {
        chunk_id: node.id,
        chunk_index: node.chunk_index,
        page_number: node.page_number,
        title: node.title,
        source: node.source,
        document_id: node.document_id
    } AS metadata
"""


# ============================================================
# 4. Hybrid GraphRAG VectorStore 불러오기
# ============================================================

vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=require_env("NEO4J_URI", NEO4J_URI),
    username=require_env("NEO4J_USERNAME", NEO4J_USERNAME),
    password=require_env("NEO4J_PASSWORD", NEO4J_PASSWORD),
    database=NEO4J_DATABASE,
    index_name=VECTOR_INDEX_NAME,
    keyword_index_name=KEYWORD_INDEX_NAME,
    search_type="hybrid",
    retrieval_query=retrieval_query,
)


# ============================================================
# 5. 질문 세트
# ============================================================

questions = [
    "사업자등록을 하지 않으면 어떤 불이익이 있나요?",
    "간이과세자가 되려면 어떤 조건이 필요한가요?",
    "부가가치세 신고 납부기한은 어떻게 되나요?",
    "세금을 체납하면 어떤 제재를 받을 수 있나요?",
    "억울한 세금이 있을 때 어떤 권리구제 절차를 이용할 수 있나요?",
]


# 질문을 바꿔보고 싶으면 아래 인덱스만 변경하면 됩니다.
QUESTION_INDEX = 4


# ============================================================
# 6. Hybrid GraphRAG 검색
# ============================================================

def search_hybrid_graphrag(question: str, k: int = 5):
    return vector_store.similarity_search_with_score(question, k=k)


def format_context(results) -> str:
    if not results:
        return "검색된 문맥이 없습니다."

    formatted = []

    for i, (doc, score) in enumerate(results, start=1):
        formatted.append(
            f"""
[검색 결과 {i}]
score: {score}
metadata: {doc.metadata}

검색된 Chunk 및 주변 Graph Context:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)


# ============================================================
# 7. 답변 생성 프롬프트
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
너는 국세청 세금절약 가이드 기반 Hybrid GraphRAG assistant다.

너는 다음 검색 결과를 바탕으로 답변한다.
1. Vector Search로 찾은 의미적으로 유사한 Chunk
2. Keyword Search로 찾은 키워드 일치 Chunk
3. 검색된 Chunk 주변의 Knowledge Graph 엔티티와 관계
4. 검색된 Chunk의 이전/다음 Chunk 문맥

답변 규칙:
- 반드시 제공된 context에 근거해서 답변하라.
- context에 없는 내용은 추측하지 말고 근거가 부족하다고 말하라.
- 검색된 이전/다음 Chunk 문맥은 질문과 직접 관련될 때만 사용하라.
- 질문이 사업자등록 미이행을 묻는 경우, 폐업신고·명의대여·사업자등록증 진위확인처럼 다른 주제는 직접 묻지 않는 한 핵심 답변에 포함하지 말라.
- 한국어로 답변하라.
- 가능한 경우 PDF page 번호를 함께 제시하라.
- 세무 대리나 법률 자문처럼 단정하지 말고, 문서 안내 기준으로 설명하라.
""",
        ),
        (
            "human",
            """
질문:
{question}

검색된 context:
{context}
""",
        ),
    ]
)


# ============================================================
# 8. 최종 질의응답 함수
# ============================================================

def answer_question(question: str, k: int = 5, show_context: bool = False) -> str:
    search_results = search_hybrid_graphrag(question, k=k)
    context = format_context(search_results)

    if show_context:
        print("\n" + "-" * 80)
        print("[검색 context]")
        print("-" * 80)
        print(context)

    messages = prompt.format_messages(
        question=question,
        context=context,
    )

    response = llm.invoke(messages)
    return response.content


# ============================================================
# 9. 실행
# ============================================================

def main() -> None:
    question = questions[QUESTION_INDEX]

    print("=" * 80)
    print("질문:", question)

    answer = answer_question(
        question,
        k=5,
        show_context=False,
    )

    print("\n답변:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
