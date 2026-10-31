"""정답 온톨로지만으로, 독자 축만 막을 때 인물 보기에 더 들어가는 '인물이 모르는 사실'의 비율을 센다. 모델을 부르지 않는다.

채팅 기본값(인물 시점 = 독자 위치 N)에서 1~마지막 장의 N마다, N장까지 등장한 인물마다 split_facts로 나눈다.
- 앎: 인물이 N장까지 알게 된 사실(상식 포함, 숨은 정체 규칙 적용)
- 독자만: 독자는 읽었지만 인물은 모르는 사실. 독자 축만 쓰면 이것도 인물 보기에 들어간다.
칸(장 × 인물 × 사실)을 모두 더한 뒤 '독자만'의 비율을 낸다(docs/paper-plan.md).

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/reader_only_gap.py [작품 id ...]
"""

import sys

from sfqa.ontology import from_synthetic, split_facts

WORKS = ["wn01-rofan", "wn02-hyeonpan", "wn03-muhyeop", "wn04-romance"]


def gap(work: str) -> tuple[int, int, int]:
    """(장 수, 앎 칸, 독자만 칸)"""
    onto = from_synthetic(work)
    known = reader_only = 0
    for n in range(1, onto.chapters + 1):
        for c in onto.characters:
            if c.first_ch is None or c.first_ch > n:
                continue
            k, r, _ = split_facts(onto, c.id, n)
            known += len(k)
            reader_only += len(r)
    return onto.chapters, known, reader_only


def main(works: list[str]) -> None:
    total_k = total_r = 0
    for w in works:
        n, k, r = gap(w)
        total_k, total_r = total_k + k, total_r + r
        print(f"{w}({n}장): 앎 {k}칸, 독자만 {r}칸, 독자만 비율 {r / (k + r):.3f}")
    print(f"합계: 앎 {total_k}칸, 독자만 {total_r}칸, 독자만 비율 {total_r / (total_k + total_r):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or WORKS)
