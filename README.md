# Graph Mini PJT - Neo4j Hybrid GraphRAG

국세청 `2026 세금절약 가이드 I` PDF를 대상으로 구축한 Neo4j 기반 Hybrid GraphRAG 실습 프로젝트입니다.

```text
PDF
  -> 페이지 단위 텍스트 로드/전처리
  -> LLM context window에 맞춘 chunk 분할
  -> Neo4j에 Document/Page/Chunk 원문 그래프 저장
  -> LLM으로 세금 도메인 Knowledge Graph 추출
  -> Neo4j에 KGEntity와 관계 저장
  -> Chunk embedding + full-text keyword index 생성
  -> Vector + Keyword + Graph context 기반 질의응답
```

## 데이터

```text
data/tax_saving_guide_2026.pdf
```

PDF 처리 후 생성되는 중간 산출물:

```text
outputs/parsed_docs.jsonl
outputs/chunks.jsonl
outputs/extracted_kg.jsonl
```

현재 파이프라인 기준 처리 결과:

```text
parsed documents: 152
chunks: 327
KG processed chunks: 40
Chunk embeddings: 327
```

## 환경 설정

가상환경을 만들고 패키지를 설치합니다.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일을 만듭니다.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

PDF_PATH=data/tax_saving_guide_2026.pdf
```

`.env`와 `.venv/`는 `.gitignore`에 포함되어 있어 Git에 올리지 않습니다.

## Neo4j 준비

Neo4j Desktop 또는 로컬 Neo4j 서버를 실행합니다.

```text
Browser: http://localhost:7474
Bolt: bolt://localhost:7687
```

DB를 초기화하고 싶다면 Neo4j Browser에서 실행합니다.

```cypher
MATCH (n)
DETACH DELETE n;
```

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `1_test_connection.py` | Neo4j 접속 확인 |
| `2_load_pdfplumber.py` | PDF 텍스트 추출, 페이지 단위 전처리, `parsed_docs.jsonl` 저장 |
| `3_split.py` | 페이지 문서를 chunk로 분할, `chunks.jsonl` 저장 |
| `4_ingest_chunks_to_neo4j_.py` | `Document`, `Page`, `Chunk` 원문 그래프를 Neo4j에 저장 |
| `5_build_kg_from_chunks.py` | LLM structured output으로 세금 KG 추출 후 Neo4j 저장 |
| `6_1_query.py` | `GraphCypherQAChain` 실험용 자연어 그래프 질의 |
| `6_2_stable_query.py` | 고정 Cypher + LLM 답변 생성 방식의 안정형 질의 |
| `7_build_vector_index.py` | `Chunk.embedding`, vector index, full-text keyword index 생성 |
| `8_ask_hybrid_graphrag.py` | Vector/Keyword 검색 + 고정 Graph 검색을 따로 실행해 합치는 Hybrid GraphRAG |
| `9_ask_hybrid_graphrag.py` | `retrieval_query`로 검색된 Chunk 주변 그래프를 바로 확장하는 Hybrid GraphRAG |

## 실행 순서

### 1. Neo4j 연결 확인

```bash
python 1_test_connection.py
```

### 2. PDF 로드 및 전처리

```bash
python 2_load_pdfplumber.py
```

생성 파일:

```text
outputs/parsed_docs.jsonl
```

하는 일:

```text
data/tax_saving_guide_2026.pdf 로드
표지/목차성 페이지 일부 제외
페이지 번호, 머리말/꼬리말, 불필요 문자 정리
페이지 단위 LangChain Document 저장
```

### 3. Chunk 생성

```bash
python 3_split.py
```

생성 파일:

```text
outputs/chunks.jsonl
```

현재 설정:

```text
chunk_size: 1000
chunk_overlap: 150
```

세금 문서의 `Guide`, `관련 법규`, `Q`, 번호 목록, bullet 구조를 고려해 분할합니다. 너무 짧은 chunk는 같은 페이지 안에서 병합합니다.

### 4. Chunk 원문 그래프 저장

```bash
python 4_ingest_chunks_to_neo4j_.py
```

생성되는 기본 그래프:

```text
(:Document:TaxGuideDocument)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
```

문서 정보:

```text
DOC_ID = tax_saving_guide_2026
DOC_TITLE = 2026 세금절약 가이드 I
DOC_PUBLISHER = 국세청
DOC_YEAR = 2026
```

### 5. Knowledge Graph 추출

```bash
python 5_build_kg_from_chunks.py
```

이 단계는 OpenAI API를 호출합니다. 전체 chunk를 처리하기 전에 일부만 테스트하는 것을 권장합니다.

```powershell
$env:KG_CHUNK_LIMIT="1"
$env:KG_RESET_OUTPUT="true"
python 5_build_kg_from_chunks.py
```

40개만 처리하려면:

```powershell
$env:KG_CHUNK_LIMIT="40"
$env:KG_RESET_OUTPUT="true"
python 5_build_kg_from_chunks.py
```

생성 파일:

```text
outputs/extracted_kg.jsonl
```

생성되는 KG 구조:

```text
(:Chunk)-[:MENTIONS]->(:KGEntity)
(:KGEntity)-[:SUPPORTED_BY]->(:Chunk)
(:KGEntity)-[:HAS_DEADLINE]->(:KGEntity)
(:KGEntity)-[:HAS_REQUIREMENT]->(:KGEntity)
(:KGEntity)-[:REQUIRES_DOCUMENT]->(:KGEntity)
(:KGEntity)-[:HAS_PENALTY]->(:KGEntity)
...
```

대표 노드 타입:

```text
Tax
TaxType
Taxpayer
BusinessType
IncomeType
TaxEvent
Deadline
Requirement
Procedure
Deduction
Credit
Exemption
Penalty
SupportProgram
Institution
Law
RequiredDocument
Amount
Rate
Condition
Concept
```

대표 관계 타입:

```text
APPLIES_TO
TRIGGERS
HAS_DEADLINE
HAS_REQUIREMENT
HAS_PROCEDURE
REQUIRES_DOCUMENT
HAS_PENALTY
QUALIFIES_FOR
REDUCES_TAX
EXEMPTS_FROM
CALCULATED_BY
HAS_RATE
HAS_AMOUNT
ADMINISTERED_BY
BASED_ON_LAW
DEFINES
RELATED_TO
```

### 6. 그래프 질의

#### 6-1. GraphCypherQAChain 실험

```bash
python 6_1_query.py
```

`GraphCypherQAChain`은 자연어 질문을 매번 Cypher로 변환합니다. 실험에는 유용하지만, 같은 질문에서도 생성 Cypher가 달라질 수 있어 답변이 흔들릴 수 있습니다.

#### 6-2. 안정형 고정 Cypher 질의

```bash
python 6_2_stable_query.py
```

이 파일은 질문별 키워드를 기준으로 고정 Cypher를 실행하고, 조회 결과만 LLM에 전달합니다.

```text
질문
  -> keywords
  -> 고정 Cypher로 Chunk/KG 관계 조회
  -> LLM 답변 생성
```

`QUESTION_INDEX`를 바꿔 다른 질문을 실행할 수 있습니다.

### 7. Vector / Keyword Hybrid Index 생성

```bash
python 7_build_vector_index.py
```

하는 일:

```text
Chunk.text 임베딩 생성
Chunk.embedding 속성 저장
Neo4j vector index 생성
Neo4j full-text keyword index 생성
Hybrid search 테스트
```

생성되는 인덱스:

```text
tax_chunk_vector_index
tax_chunk_keyword_index
```

현재 실행 결과:

```text
total_chunks: 327
embedded_chunks: 327
```

### 8. Hybrid GraphRAG - 분리형 검색

```bash
python 8_ask_hybrid_graphrag.py
```

8번 방식은 검색을 두 갈래로 나눠 실행한 뒤 마지막에 합칩니다.

```text
1. Neo4jVector hybrid search
   - Vector search
   - Keyword search

2. 고정 Cypher graph search
   - 관련 Chunk
   - KGEntity 관계
   - evidence

3. 두 context를 LLM에 전달해 최종 답변 생성
```

장점:

```text
검색 흐름이 명확하고 디버깅하기 쉽다.
GraphCypherQAChain보다 답변이 안정적이다.
```

### 9. Hybrid GraphRAG - retrieval_query 그래프 확장

```bash
python 9_ask_hybrid_graphrag.py
```

9번 방식은 `Neo4jVector.from_existing_index(..., retrieval_query=...)`를 사용합니다.

```text
질문
  -> Vector + Keyword hybrid search
  -> 검색된 Chunk가 node 변수로 retrieval_query에 전달됨
  -> node 주변 그래프 확장
     - 이전 Chunk
     - 다음 Chunk
     - MENTIONS된 KGEntity
     - KGEntity 주변 관계/evidence
  -> 확장된 context를 LLM에 전달
  -> 최종 답변 생성
```

8번과 9번 차이:

| 구분 | 8번 | 9번 |
| --- | --- | --- |
| 검색 방식 | Vector/Keyword 검색과 Graph 검색을 따로 실행 | Vector/Keyword 검색 결과에서 바로 그래프 확장 |
| Graph context | 고정 Cypher 함수로 별도 조회 | `retrieval_query` 안에서 확장 |
| 장점 | 흐름이 단순하고 안정적 | 검색된 chunk 주변 문맥을 자연스럽게 포함 |
| 주의점 | graph 검색 키워드를 별도로 관리해야 함 | 앞뒤 chunk 때문에 질문과 먼 문맥이 섞일 수 있음 |

## 예시 질문

```text
사업자등록을 하지 않으면 어떤 불이익이 있나요?
간이과세자가 되려면 어떤 조건이 필요한가요?
부가가치세 신고 납부기한은 어떻게 되나요?
세금을 체납하면 어떤 제재를 받을 수 있나요?
억울한 세금이 있을 때 어떤 권리구제 절차를 이용할 수 있나요?
```

예시 답변 요지:

```text
사업자등록을 하지 않으면 가산세가 부과되고, 매입세액 공제를 받을 수 없습니다.
개인: 공급가액의 1%
간이과세자: 매출액의 0.5%와 5만 원 중 큰 금액
법인: 공급가액의 1%
근거: PDF page 25, page 33
```

## Neo4j에서 확인하기 좋은 Cypher

전체 문서 구조:

```cypher
MATCH path = (:Document {id: "tax_saving_guide_2026"})-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk)
RETURN path
LIMIT 50;
```

Chunk와 KGEntity:

```cypher
MATCH path = (c:Chunk)-[:MENTIONS]->(e:KGEntity)
RETURN path
LIMIT 100;
```

사업자등록 관련 원문:

```cypher
MATCH (c:Chunk)
WHERE c.text CONTAINS "사업자등록을 하지 않으면"
RETURN c.page_number AS page, c.title AS title, c.text AS text
LIMIT 5;
```

KG 관계와 evidence:

```cypher
MATCH (source:KGEntity)-[r]->(target:KGEntity)
WHERE type(r) <> "SUPPORTED_BY"
RETURN
  source.name AS source,
  type(r) AS relationship,
  target.name AS target,
  r.evidence AS evidence
LIMIT 50;
```

Vector / full-text index 확인:

```cypher
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties
WHERE name CONTAINS "tax_chunk"
RETURN name, type, labelsOrTypes, properties;
```

Embedding 저장 확인:

```cypher
MATCH (c:Chunk)
RETURN count(c) AS total_chunks, count(c.embedding) AS embedded_chunks;
```

