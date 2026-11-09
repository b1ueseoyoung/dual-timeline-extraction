"""4편 정답 재검증 결과(checks/gold-reverify-20261109.json)의 쌍 단위 비율과 평가 수치 영향을 잰다. 모델을 부르지 않는다(논문 5.3절).

- 표본 쌍: 표본 사실의 known_by 쌍에 에이전트가 더하라고 한 쌍을 합친 수
- 표시 쌍: 표본 사실에서 에이전트가 고치라고 한 쌍. 유형은 현재 정답 기준으로 나눈다.
- 영향: 표시한 쌍을 모두 정답에 반영해 onto_eval.score를 다시 계산한다. 전체와 표본 사실만 따로 본다.

논문 5.3절의 재대조 수치(사실 74개 중 44개, 표본 쌍 686개 중 47개)는 정답을 고치기 전(커밋 a86b4db의 data/synthetic) 기준이다.
지금 정답으로 돌리면 이미 고친 14쌍만큼 달라진다. 최종 수치는 final_metrics.py로 낸다.

실행(worktree 루트): PYTHONPATH=src:docs/ontology-extract uv run --no-sync python docs/ontology-extract/gold_reverify_impact.py
"""

import json
from collections import Counter
from pathlib import Path

from gold_issue_sensitivity import ADD, REMOVE, SHIFT

from sfqa.gold import load_texts
from sfqa.onto_eval import score
from sfqa.ontology import SYNTHETIC_ROOT, Knowing, Ontology, from_synthetic
from sfqa.works import load_work

WORKS = {"wn01-rofan": "wn01-opus-50", "wn02-hyeonpan": "wn02-opus-50", "wn03-muhyeop": "wn03-opus-50", "wn04-romance": "wn04-opus-50"}
KEYS = ("tp", "fp", "fn")


def unique_pairs(verdict: dict) -> list[dict]:
    """같은 (사실, 인물, 변경)이 여러 문제에 나오면 한 번만 센다."""
    seen, out = set(), []
    for u in verdict["unresolved"]:
        for p in u.get("pairs", []):
            key = (p["fact"], p["character"], p["change"])
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def fixed(gold: Ontology, pairs: list[dict]) -> Ontology:
    new = gold.model_copy(deep=True)
    facts = {f.id: f for f in new.facts}
    for p in pairs:
        f = facts[p["fact"]]
        if p["change"] == "add" and all(k.character != p["character"] for k in f.known_by):
            f.known_by.append(Knowing(character=p["character"], from_ch=int(p["from_ep"])))
        elif p["change"] == "remove":
            f.known_by = [k for k in f.known_by if k.character != p["character"]]
        elif p["change"] == "shift":
            for k in f.known_by:
                if k.character == p["character"]:
                    k.from_ch = int(p["from_ep"])
    return new


def earlier_issues(gold: Ontology) -> list[dict]:
    """생성 당시 WN1에 남긴 9건을 최대로 잡은 쌍(gold_issue_sensitivity의 최대)."""
    facts = {f.id: f for f in gold.facts}
    out = [{"fact": f, "character": c, "change": "add", "from_ep": e} for f, c, e in ADD]
    out += [{"fact": f, "character": c, "change": "remove", "from_ep": None} for f, c in REMOVE]
    out += [{"fact": f, "character": c, "change": "shift", "from_ep": e} for f, c, e in SHIFT]
    out += [{"fact": "f147", "character": k.character, "change": "remove", "from_ep": None} for k in facts["f147"].known_by if k.from_ch == 48]
    out += [{"fact": "f064", "character": k.character, "change": "add", "from_ep": k.from_ch}
            for fid in ("f127", "f176") for k in sorted(facts[fid].known_by, key=lambda x: x.from_ch)]
    return out


def pr(c: Counter) -> tuple[float, float]:
    return round(c["tp"] / (c["tp"] + c["fp"]), 3), round(c["tp"] / (c["tp"] + c["fn"]), 3)


def main() -> None:
    total = {name: Counter() for name in ("현재", "수정", "수정+이전 9건", "표본 현재", "표본 수정")}
    kinds, sizes, sev = Counter(), Counter(), Counter()
    for work, run in WORKS.items():
        d = Path("runs/onto-extract") / run
        gold = from_synthetic(work)
        onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
        judgments = json.loads((d / "judgments.json").read_text(encoding="utf-8"))
        upto = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))["upto"]
        loaded = load_work(work, SYNTHETIC_ROOT)
        texts = load_texts(loaded)
        now = score(gold, onto, upto, judgments, texts)
        total["현재"].update({k: now[k] for k in KEYS})
        path = loaded.dir / "checks" / "gold-reverify-20261109.json"
        v = json.loads(path.read_text(encoding="utf-8"))
        sev.update(u["severity"] for u in v["unresolved"])
        pairs, sampled = unique_pairs(v), {s["fact"] for s in v["sampled"]}
        new = fixed(gold, pairs)
        after = score(new, onto, upto, judgments, texts)
        total["수정"].update({k: after[k] for k in KEYS})
        more = fixed(gold, pairs + earlier_issues(gold)) if work == "wn01-rofan" else new
        wider = score(more, onto, upto, judgments, texts)
        total["수정+이전 9건"].update({k: wider[k] for k in KEYS})
        a = score(gold.model_copy(update={"facts": [f for f in gold.facts if f.id in sampled]}), onto, upto, judgments, texts)
        b = score(new.model_copy(update={"facts": [f for f in new.facts if f.id in sampled]}), onto, upto, judgments, texts)
        total["표본 현재"].update({k: a[k] for k in KEYS})
        total["표본 수정"].update({k: b[k] for k in KEYS})
        facts = {f.id: f for f in gold.facts}
        outline = json.loads((loaded.dir / "outline.json").read_text(encoding="utf-8"))
        off = {(k["fact_id"], k["character"]) for e in outline for k in e.get("knowledge_events", []) if k.get("offstage")}
        hit = [p for p in pairs if p["fact"] in sampled]
        for p in hit:
            cur = {k.character: k.from_ch for k in facts[p["fact"]].known_by}.get(p["character"])
            kinds["누락" if p["change"] == "add" else "0인데 본문 근거 없음" if cur == 0
                  else "본문 밖 인지의 시점 근거 없음" if (p["fact"], p["character"]) in off else "본문 안 시점 차이"] += 1
        sizes.update({"표본 사실": len(sampled), "일치": v["sampled_ok"], "표본 쌍": sum(len(facts[f].known_by) for f in sampled) + sum(p["change"] == "add" for p in hit),
                      "표시 쌍": len(hit), "전체 표시 쌍": len(pairs)})
        print(f"{work}: 표본 {v['sampled_ok']}/{v['sampled_total']}, 정밀도 {now['precision']}→{after['precision']}, 재현율 {now['recall']}→{after['recall']}, 공개 시점 {after['reveal']}")
    print(dict(sizes), f"표시 비율 {sizes['표시 쌍'] / sizes['표본 쌍']:.3f}", dict(kinds), dict(sev))
    for name, c in total.items():
        print(name, dict(c), "정밀도·재현율", pr(c))


if __name__ == "__main__":
    main()
