"""논문에 쓰는 수치를 지금 정답과 저장된 추출 결과로 한 번에 다시 낸다. 모델을 부르지 않는다.

- 정답의 basis가 latest_by인 쌍은 인지 시점의 상한이다(eval/gold-fix/README.md). onto_eval.score가 이른 인지와 시점 일치에서 뺀다.
- 과잉 지식과 허용 지식은 기록된 회차를 기준으로 계산한다. 상한 쌍을 뺀 값을 민감도로 함께 낸다.
- 유보 민감도는 eval/gold-fix/decisions.csv의 판단 유보 쌍을 정답에 반영한 값이다. 철회한 제안은 대안 정답이 아니므로 뺀다.

결과는 eval/gold-fix/final-metrics.json에 쓴다. --write를 주면 실행 폴더의 evaluation.json도 새로 쓴다.
실행(worktree 루트): PYTHONPATH=src:docs/ontology-extract uv run --no-sync python docs/ontology-extract/final_metrics.py [--write]
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from baselines import compare
from compare_runs import load, pair_sets
from reader_only_gap import WORKS as WORK_IDS

from sfqa.gold import load_texts
from sfqa.onto_eval import score
from sfqa.ontology import SYNTHETIC_ROOT, Knowing, Ontology, from_synthetic, split_facts
from sfqa.works import load_work

RUNS = Path("runs/onto-extract")
OUT = Path("eval/gold-fix/final-metrics.json")
DECISIONS = Path("eval/gold-fix/decisions.csv")
KEYS = ("tp", "fp", "fn", "full", "partial", "missed", "since_late", "since_same", "since_late_bound", "since_bound_unknown",
        "after_reveal", "after_reveal_hit")


def run_dir(work: str, model: str) -> Path:
    return RUNS / f"{work.split('-')[0]}-{model}-50"


def bound_pairs(gold: Ontology) -> frozenset:
    return frozenset((f.id, k.character) for f in gold.facts for k in f.known_by if k.basis == "latest_by")


def with_held(gold: Ontology, tag: str, rows: list[dict]) -> Ontology:
    """철회를 뺀 판단 유보 쌍을 반영한 정답(민감도)."""
    new = gold.model_copy(deep=True)
    facts = {f.id: f for f in new.facts}
    for r in (x for x in rows if x["work"] == tag and x["decision"] == "판단 유보" and x["kind"] != "철회"):
        f, who = facts[r["fact"]], r["character"]
        if r["change"] == "add" and all(k.character != who for k in f.known_by):
            f.known_by.append(Knowing(character=who, from_ch=int(r["new"])))
        elif r["change"] == "remove":
            f.known_by = [k for k in f.known_by if k.character != who]
        elif r["change"] == "shift":
            for k in f.known_by:
                if k.character == who:
                    k.from_ch = int(r["new"])
    return new


def gap(gold: Ontology) -> tuple[int, int]:
    """회차·인물·사실 칸 중 (인물이 아는 칸, 독자만 아는 칸). 인물 시점 = 독자 위치, 등장한 인물만 센다."""
    known = only = 0
    for n in range(1, gold.chapters + 1):
        for c in gold.characters:
            if c.first_ch is None or c.first_ch > n:
                continue
            k, r, _ = split_facts(gold, c.id, n)
            known += len(k)
            only += len(r)
    return known, only


def recall_by_kind(work: str, gold: Ontology) -> Counter:
    run = run_dir(work, "opus")
    _, ev, parts = load(str(run), gold)
    _, tp, _, fn = pair_sets(gold, parts, ev["upto"], set(ev["comparable"]))
    outline = json.loads((load_work(work, SYNTHETIC_ROOT).dir / "outline.json").read_text(encoding="utf-8"))
    off = {(k["fact_id"], k["character"]) for e in outline for k in e.get("knowledge_events", []) if k.get("offstage")}
    since = {(f.id, k.character): k.from_ch for f in gold.facts for k in f.known_by}
    out = Counter()
    for label, pairs in (("tp", tp), ("fn", fn)):
        for p in pairs:
            out[("0" if since[p] == 0 else "본문 밖" if p in off else "본문 안", label)] += 1
    return out


def view(row: dict) -> dict:
    return {"독자 규칙": row["독자 축만"]["보기 더 앎 칸"], "추출": row["추출"]["보기 더 앎 칸"], "허용 재현율": row["추출"]["보기 재현율"]}


def reduction(v: dict) -> float:
    return round(1 - v["추출"] / v["독자 규칙"], 3)


def main(write: bool) -> None:
    rows = list(csv.DictReader(DECISIONS.open(encoding="utf-8-sig")))
    out: dict = {"작품": {}, "모델": {}}
    total = {m: Counter() for m in ("opus", "sonnet", "haiku")}
    held_total, kinds = Counter(), Counter()
    for work in WORK_IDS:
        tag = work.split("-")[0]
        gold = from_synthetic(work)
        loaded = load_work(work, SYNTHETIC_ROOT)
        texts = load_texts(loaded)
        outline = json.loads((loaded.dir / "outline.json").read_text(encoding="utf-8"))
        bound = bound_pairs(gold)
        known, only = gap(gold)
        info = {
            "인물": len(gold.characters), "사실": len(gold.facts), "반전": sum(f.is_twist for f in gold.facts),
            "인지 쌍": sum(len(f.known_by) for f in gold.facts),
            "공개 후 인지": sum(k.from_ch > f.reveal_ch and k.basis != "latest_by" for f in gold.facts for k in f.known_by),
            "본문 밖 인지 사건": sum(bool(k.get("offstage")) for e in outline for k in e.get("knowledge_events", [])),
            "상한 쌍": len(bound), "모르는 칸 비율": round(only / (known + only), 4), "아는 칸": known, "모르는 칸": only,
        }
        scores = {}
        for model in total:
            d = run_dir(work, model)
            onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
            judgments = json.loads((d / "judgments.json").read_text(encoding="utf-8"))
            s = score(gold, onto, judgments["upto"], judgments, texts)
            if write:
                (d / "evaluation.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
            scores[model] = s
            total[model].update({k: s[k] for k in KEYS})
            total[model].update({"early": len(s["since_early"]), "early_partial": len(s["since_early_partial"])})
            total[model].update({f"reveal_{k}": v for k, v in s["reveal"].items()})
            total[model].update({f"twist_{k}": v for k, v in s["twists"].items()})
            out["모델"].setdefault(model, {})[tag] = {"정밀도": s["precision"], "재현율": s["recall"], "완전": s["full"], "부분": s["partial"], "누락": s["missed"],
                                                     "공개 일치": s["reveal"].get("same", 0), "공개 빠름": s["reveal"].get("early", 0), "공개 늦음": s["reveal"].get("late", 0),
                                                     "반전 완전": s["twists"].get("full", 0), "반전 전체": sum(s["twists"].values())}
        kinds.update(recall_by_kind(work, gold))
        base = compare(str(run_dir(work, "opus")), work)
        skipped = compare(str(run_dir(work, "opus")), work, skip=bound)
        info["비교"] = {name: {k: base[name][k] for k in ("쌍 정밀도", "쌍 재현율")} for name in base}
        info["보기"], info["보기(상한 쌍 제외)"] = view(base), view(skipped)
        # 유보 민감도
        held = with_held(gold, tag, rows)
        d = run_dir(work, "opus")
        onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
        judgments = json.loads((d / "judgments.json").read_text(encoding="utf-8"))
        h = score(held, onto, judgments["upto"], judgments, texts)
        held_total.update({k: h[k] for k in KEYS})
        held_total.update({"early": len(h["since_early"])})
        hv = view(compare(str(d), work, gold=held))
        held_known, held_only = gap(held)
        held_total.update({"더 앎 독자": hv["독자 규칙"], "더 앎 추출": hv["추출"], "아는 칸": held_known, "모르는 칸": held_only})
        info["유보 반영"] = {"정밀도": h["precision"], "재현율": h["recall"], "감소율": reduction(hv), "허용 재현율": hv["허용 재현율"], "모르는 칸 비율": round(held_only / (held_known + held_only), 4)}
        out["작품"][tag] = info
        print(tag, {k: info[k] for k in ("인물", "사실", "반전", "인지 쌍", "공개 후 인지", "본문 밖 인지 사건", "상한 쌍", "모르는 칸 비율")})
        print("   Opus", out["모델"]["opus"][tag])
        print("   비교", info["비교"])
        print("   보기", info["보기"], "감소", reduction(info["보기"]), "| 상한 쌍 제외", info["보기(상한 쌍 제외)"], "감소", reduction(info["보기(상한 쌍 제외)"]), "| 유보 반영", info["유보 반영"])
    def pr(c: Counter) -> tuple[float, float]:
        return round(c["tp"] / (c["tp"] + c["fp"]), 3), round(c["tp"] / (c["tp"] + c["fn"]), 3)
    infos = out["작품"].values()
    out["합계"] = {
        "표 1": {k: sum(i[k] for i in infos) for k in ("인물", "사실", "반전", "인지 쌍", "공개 후 인지", "본문 밖 인지 사건", "상한 쌍")},
        "모르는 칸 비율": round(sum(i["모르는 칸"] for i in infos) / sum(i["아는 칸"] + i["모르는 칸"] for i in infos), 4),
        "Opus": {**dict(total["opus"]), "정밀도·재현율": pr(total["opus"])},
        "재현율 유형별": {k: {"tp": kinds[(k, "tp")], "fn": kinds[(k, "fn")], "재현율": round(kinds[(k, "tp")] / (kinds[(k, "tp")] + kinds[(k, "fn")]), 3)} for k in ("본문 안", "0", "본문 밖")},
        "보기": {k: sum(i["보기"][k] for i in infos) for k in ("독자 규칙", "추출")},
        "보기(상한 쌍 제외)": {k: sum(i["보기(상한 쌍 제외)"][k] for i in infos) for k in ("독자 규칙", "추출")},
        "표 3 평균": {name: {k: round(sum(i["비교"][name][k] for i in infos) / 4, 3) for k in ("쌍 정밀도", "쌍 재현율")} for name in ("독자 축만", "장면 동석", "추출")},
        "모델 평균": {m: {"정밀도": round(sum(v["정밀도"] for v in out["모델"][m].values()) / 4, 3), "재현율": round(sum(v["재현율"] for v in out["모델"][m].values()) / 4, 3), "완전": sum(v["완전"] for v in out["모델"][m].values())} for m in total},
        "유보 반영": {**dict(held_total), "정밀도·재현율": pr(held_total), "감소율": round(1 - held_total["더 앎 추출"] / held_total["더 앎 독자"], 3), "모르는 칸 비율": round(held_total["모르는 칸"] / (held_total["아는 칸"] + held_total["모르는 칸"]), 4)},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: list(x)), encoding="utf-8")
    print(json.dumps(out["합계"], ensure_ascii=False, indent=0, default=lambda x: list(x)))


if __name__ == "__main__":
    main("--write" in sys.argv[1:])
