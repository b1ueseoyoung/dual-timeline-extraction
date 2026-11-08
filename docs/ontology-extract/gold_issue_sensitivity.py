"""WN1 정답에 남은 문제 9건(data/synthetic/wn01-rofan/checks/known-issues.md)을 정답에 적용해 수치 변화를 잰다. 모델을 부르지 않는다(논문 5.3절).

고치는 방법이 둘인 항목이 있어 두 경우를 본다.
- 최소: 인물을 더하거나 빼는 것이 분명한 항목만 고친다(추가 6쌍, 제거 4쌍).
- 최대: 사실을 나누는 수정까지 인물 변경으로 친다. f147의 48화 인물을 빼고, f064에 f127·f176의 인물을 더하고, 모르데인 16화 '짐작'을 20화로 늦춘다.

실행(worktree 루트): PYTHONPATH=src:docs/ontology-extract uv run --no-sync python docs/ontology-extract/gold_issue_sensitivity.py
"""

import json
from collections import Counter
from pathlib import Path

from sfqa.gold import load_texts
from sfqa.onto_eval import score
from sfqa.ontology import SYNTHETIC_ROOT, Knowing, Ontology, from_synthetic
from sfqa.works import load_work

WORK, RUN = "wn01-rofan", Path("runs/onto-extract/wn01-opus-50")
OTHERS = ["wn02-opus-50", "wn03-opus-50", "wn04-opus-50"]
ADD = [("f121", "ondre", 0), ("f121", "raphael", 47), ("f046", "ondre", 0), ("f097", "ondre", 46), ("f065", "marie", 37), ("f065", "ondre", 46)]
REMOVE = [("f139", "marie"), ("f160", "noel"), ("f160", "rowan"), ("f160", "rosetta")]
SHIFT = [("f034", "mordain", 20), ("f175", "mordain", 20)]


def changed(gold: Ontology, add: list, remove: list, shift: list = ()) -> tuple[Ontology, Counter]:
    """고친 정답과 실제로 바뀐 쌍의 수."""
    new = gold.model_copy(deep=True)
    facts, n = {f.id: f for f in new.facts}, Counter()
    for fid, who, ch in add:
        if all(k.character != who for k in facts[fid].known_by):
            facts[fid].known_by.append(Knowing(character=who, from_ch=ch))
            n["추가"] += 1
    for fid, who in remove:
        n["제거"] += sum(k.character == who for k in facts[fid].known_by)
        facts[fid].known_by = [k for k in facts[fid].known_by if k.character != who]
    for fid, who, ch in shift:
        for k in facts[fid].known_by:
            if k.character == who and k.from_ch != ch:
                k.from_ch = ch
                n["시점 변경"] += 1
    return new, n


def main() -> None:
    gold = from_synthetic(WORK)
    facts = {f.id: f for f in gold.facts}
    late = [("f147", k.character) for k in facts["f147"].known_by if k.from_ch == 48]
    more = [("f064", k.character, k.from_ch) for fid in ("f127", "f176") for k in sorted(facts[fid].known_by, key=lambda x: x.from_ch)]
    variants = {"현재": (gold, Counter()), "최소": changed(gold, ADD, REMOVE), "최대": changed(gold, ADD + more, REMOVE + late, SHIFT)}
    onto = Ontology.model_validate_json((RUN / "ontology.json").read_text(encoding="utf-8"))
    judgments = json.loads((RUN / "judgments.json").read_text(encoding="utf-8"))
    upto = json.loads((RUN / "evaluation.json").read_text(encoding="utf-8"))["upto"]
    texts = load_texts(load_work(WORK, SYNTHETIC_ROOT))
    others = Counter()
    for run in OTHERS:
        ev = json.loads((RUN.parent / run / "evaluation.json").read_text(encoding="utf-8"))
        others.update({k: ev[k] for k in ("tp", "fp", "fn")})
    for name, (g, n) in variants.items():
        s = score(g, onto, upto, judgments, texts)
        t = Counter({k: s[k] for k in ("tp", "fp", "fn")}) + others
        print(f"{name}: 바뀐 쌍 {sum(n.values())} {dict(n)} | WN1 정밀도 {s['precision']} 재현율 {s['recall']} 공개 시점 {s['reveal']}"
              f" | 4편 정밀도 {t['tp'] / (t['tp'] + t['fp']):.3f} 재현율 {t['tp'] / (t['tp'] + t['fn']):.3f}")


if __name__ == "__main__":
    main()
