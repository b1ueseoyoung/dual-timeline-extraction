"""추출과 단순 기준선 둘을 같은 짝·같은 지표로 비교한다. 모델을 부르지 않는다(docs/paper-plan.md F2).

- 추출: 짝지어진 추출 사실의 아는 인물(onto_eval.score와 같은 계산)
- 독자 축만: 짝지어진 정답 사실을 범위 안의 모든 인물이 공개 장부터 안다고 본다(인물 축이 없을 때)
- 장면 동석: 짝지어진 추출 사실이 나온 장의 장면에 있던 인물(추출한 장면 기록)만 그 장부터 안다고 본다
  (LLM의 인물별 앎 판단 없이 '그 장에 있었으면 안다'만 쓸 때)
지표는 둘이다. 아는 인물 쌍 정밀도·재현율(범위 끝 기준, score와 같은 계산)과, 채팅 기본값 보기 칸의
정밀도·재현율(인물 시점 = 독자 위치, 1장부터 칸 합, '하나라도' 기준. compare_runs.view_cells)이다.

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/baselines.py <실행 폴더>:<작품 id> ...
"""

import json
import sys
from pathlib import Path

from compare_runs import load, pair_sets, view_cells

from sfqa.gold import load_texts
from sfqa.onto_eval import id_map
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.works import load_work


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def compare(run: str, work: str) -> dict:
    gold = from_synthetic(work)
    _, ev, parts = load(run, gold)
    upto, comparable = ev["upto"], set(ev["comparable"])
    d = Path(run)
    onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
    judgments = json.loads((d / "judgments.json").read_text(encoding="utf-8"))
    m, facts = id_map(onto, gold), {f.id: f for f in onto.facts}
    golds = {g.id: g for g in gold.facts}
    present = {s.chapter: {m[p] for p in s.present if p in m} for s in onto.scenes}
    reader = {gid: [{c: golds[gid].reveal_ch for c in comparable}] for gid, p in parts.items() if p}
    scene = {gid: [{c: facts[e].reveal_ch for c in present.get(facts[e].reveal_ch, set()) & comparable} for e in j["ids"]]
             for gid, j in judgments["match"].items()}
    texts = load_texts(load_work(work, SYNTHETIC_ROOT))
    first = {c.id: next(n for n in range(1, upto + 1) if any(f in texts[n] for f in c.forms))
             for c in gold.characters if c.id in comparable}
    out = {}
    for label, p in (("추출", parts), ("독자 축만", reader), ("장면 동석", scene)):
        _, tp, fp, fn = pair_sets(gold, p, upto, comparable)
        cells = view_cells(gold, p, upto, first, whole=False)
        vt, vn = cells["맞게 앎"], cells["모름"]
        vf = cells["더 앎(정답은 뒤에 앎)"] + cells["더 앎(정답은 범위 끝까지 모름)"]
        out[label] = {"쌍 정밀도": ratio(len(tp), len(tp) + len(fp)), "쌍 재현율": ratio(len(tp), len(tp) + len(fn)),
                      "보기 정밀도": ratio(vt, vt + vf), "보기 재현율": ratio(vt, vt + vn), "보기 더 앎 칸": vf,
                      "보기 모름 칸": vn}
    return out


def main(pairs: list[str]) -> None:
    for item in pairs:
        run, work = item.split(":")
        print(f"== {run} ({work})")
        for label, row in compare(run, work).items():
            print(f"  {label}: " + ", ".join(f"{k} {v}" for k, v in row.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
