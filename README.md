# dual-timeline-extraction

Code for extracting, from serialized fiction, the chapter in which each fact is revealed to the reader and the chapter in which each character learns it, and for evaluating the extraction against gold data.

연재 소설에서 사실마다 독자에게 공개되는 회차(공개 시점)와 각 인물이 그 사실을 알게 되는 회차(인지 시점)를 추출하고, 정답 데이터와 대조해 평가하는 코드입니다.

## 구성

- `src/sfqa/ontology.py`: 인물·사실·인지 시점 스키마, 독자 위치 N과 인물 시점 M에 따른 허용 집합, 숨겨진 별칭 처리
- `src/sfqa/onto_extract.py`: 회차 단위 순차 추출(누적 기록, 추가 전용 갱신, 인용문 검증, 인물 병합)
- `src/sfqa/onto_eval.py`: 정답 사실과의 매칭 판정과 공개·인지 시점 채점
- `src/sfqa/fantom.py`: FANToM 보조 실험
- `src/sfqa/` 나머지: 모델 호출(`llm.py`, `claude.py`), 설정, 본문 읽기, BM25 등 보조 모듈
- `docs/ontology-extract/`: 수치 집계, 민감도 분석, 사람 검증 집계 스크립트
- `tests/`: 단위 테스트

실행 방법은 각 모듈과 스크립트 맨 위 설명에 있습니다.

## 실행 환경

- Python 3.12 이상. 예: `uv sync` 후 `uv run pytest`
- 모델을 부르는 실행에는 환경 변수 `ANTHROPIC_API_KEY`가 필요합니다.
- 데이터(합성 웹소설, 정답 데이터, 추출 결과, 사람 검증표)는 이 저장소에 없습니다. 분석 스크립트와 일부 테스트는 `data/`, `runs/`, `eval/` 아래 자료가 있어야 동작하며, 자료가 없으면 해당 테스트는 건너뜁니다.

## 출처

원래 프로젝트에서 추출·평가에 쓰는 모듈만 옮겼습니다. `gold.py`에는 본문을 읽는 `load_texts`만 남겼습니다.
