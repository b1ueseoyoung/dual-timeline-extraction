"""제2 검증자(B)가 수용하지 않은 판정을 표본 120개에 적용해 수치 변화를 잰다. 모델을 부르지 않고 실행 결과를 고치지 않는다.

eval/human-check/match-rescore.csv가 B가 '아니오'로 둔 26행의 처리표다. B의 메모를 읽고 나눴다(논문 6.5절).
- 부분: 세부 내용이 빠진 완전 매칭을 부분 매칭으로 바꾼다.
- 다름: 내용이 다르다고 본 매칭. 완화는 부분 매칭, 엄격은 매칭 없음으로 바꾼다.
- 제외: drop의 추출 사실만 짝에서 뺀다.
- 제외·부분: drop을 빼고 부분 매칭으로 바꾼다.
- 없음: 매칭 없음으로 바꾼다.
- 완전: 부분 매칭을 완전 매칭으로 바꾼다.

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/judge_sensitivity.py
"""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

from human_check import OUT, RUNS, read_rows

from sfqa.gold import load_texts
from sfqa.onto_eval import score
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.works import load_work


def rescore(judgments: dict, plan: list[dict], strict: bool) -> dict:
    out = copy.deepcopy(judgments)
    for r in plan:
        e, treat = out["match"][r["gold_id"]], r["treat"]
        if treat.startswith("제외"):
            e["ids"] = [i for i in e["ids"] if i not in r["drop"].split()]
        if treat in ("부분", "제외·부분") or (treat == "다름" and not strict):
            e["partial"] = True
        elif treat in ("없음", "다름"):
            e["ids"], e["partial"] = [], False
        elif treat == "완전":
            e["partial"] = False
    return out


def main() -> None:
    sheet, plan = read_rows(OUT / "match-check-B.csv"), read_rows(OUT / "match-rescore.csv")
    if {int(r["row"]) for r in plan} != {n for n, r in enumerate(sheet, 2) if r["human_ok"].strip() == "아니오"}:
        sys.exit("처리표가 B의 아니오 행과 다름")
    total = {name: Counter() for name in ("현재", "완화", "엄격")}
    for work, run in RUNS.items():
        d = Path("runs/onto-extract") / run
        gold = from_synthetic(work)
        ids = {r["gold_id"] for r in sheet if r["work"] == work}
        sample = gold.model_copy(update={"facts": [f for f in gold.facts if f.id in ids]})
        onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
        judgments = json.loads((d / "judgments.json").read_text(encoding="utf-8"))
        upto = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))["upto"]
        texts = load_texts(load_work(work, SYNTHETIC_ROOT))
        rows = [r for r in plan if r["work"] == work]
        for name, j in (("현재", judgments), ("완화", rescore(judgments, rows, False)), ("엄격", rescore(judgments, rows, True))):
            s = score(sample, onto, upto, j, texts)
            total[name].update({k: s[k] for k in ("full", "partial", "missed", "tp", "fp", "fn")} | s["reveal"])
    for name, t in total.items():
        matched = t["full"] + t["partial"]
        print(f"{name}: 완전 {t['full']} 부분 {t['partial']} 없음 {t['missed']} | 공개 시점 일치 {t['same']}/{matched} "
              f"({t['same'] / matched:.1%}), 이름 {t['early']}, 늦음 {t['late']} | 인지 쌍 정밀도 {t['tp'] / (t['tp'] + t['fp']):.3f}, "
              f"재현율 {t['tp'] / (t['tp'] + t['fn']):.3f}")


if __name__ == "__main__":
    main()
