"""작품 본문을 회차별로 읽는다.

원래 프로젝트의 gold.py에는 Q&A 정답지 검사 코드도 있지만, 이 저장소의 추출·평가 코드는 load_texts만 쓰므로 그것만 둔다.
"""

from __future__ import annotations

from sfqa.works import Work


def load_texts(work: Work) -> dict[int, str]:
    return {c: work.clean_path(c).read_text(encoding="utf-8") for c in work.chapter_numbers}
