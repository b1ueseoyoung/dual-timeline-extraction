from __future__ import annotations

from sfqa.works import Work


def load_texts(work: Work) -> dict[int, str]:
    return {c: work.clean_path(c).read_text(encoding="utf-8") for c in work.chapter_numbers}
