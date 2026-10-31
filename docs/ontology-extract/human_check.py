"""사람 검증용 확인표를 만들고, 채운 표를 집계한다. 모델을 부르지 않는다(docs/paper-plan.md 7절 한계, eval/human-check/README.md).

확인표 셋(합성 4편, Opus 추출 실행 기준):
- match-check.csv: LLM 짝 판정 확인. 실행마다 onto_eval이 무작위로 뽑아 둔 30행(review-matches.csv)을 모은 120행.
- early-check.csv: 공개 시점이 정답보다 이르게 잡힌 23건이 실제로 뒷내용을 흘리는지 확인.
- gold-check.csv: 정답 장부 확인. 반전 사실 전부(34개)와 인물별 인지 시점 기록 표본 50건(1화 이후에 알게 된 것만).
  기록마다 본문에서 찾은 근거 후보 줄을 붙인다(사실 키워드나 인물 이름이 든 줄). 판단은 본문 기준으로 한다.

검증자는 표를 복사해 각자 채운다(예: match-check-A.csv, match-check-B.csv). 둘이 채우면 일치도(카파)도 낸다.

실행(worktree 루트):
  PYTHONPATH=src uv run --no-sync python docs/ontology-extract/human_check.py make
  PYTHONPATH=src uv run --no-sync python docs/ontology-extract/human_check.py score eval/human-check/match-check-A.csv [...-B.csv]
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from sfqa.gold import load_texts
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.works import load_work

RUNS = {"wn01-rofan": "wn01-opus-50", "wn02-hyeonpan": "wn02-opus-50", "wn03-muhyeop": "wn03-opus-50",
        "wn04-romance": "wn04-opus-50"}
OUT = Path("eval/human-check")
SEED = 20261102
KNOWN_SAMPLE = 50  # 작품마다 13·13·12·12건
YES_NO = {"예", "아니오"}


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def lines_with(text: str, words: list[str], limit: int = 3) -> str:
    """words 가운데 하나라도 든 줄을 앞에서부터 limit개(줄 번호와 함께)."""
    found = [f"{n}: {line.strip()}" for n, line in enumerate(text.splitlines(), 1)
             if line.strip() and any(w and w in line for w in words)]
    return " / ".join(found[:limit])


def match_rows() -> list[dict]:
    rows = []
    for work, run in RUNS.items():
        with (Path("runs/onto-extract") / run / "review-matches.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows.append({"work": work, "gold_id": r["gold_id"], "gold_ch": r["gold_ch"],
                             "gold_statement": r["gold_statement"], "judged": r["judged"], "partial": r["partial"],
                             "matched": r["matched"], "candidates": r["candidates"], "reason": r["reason"],
                             "human_ok": "", "human_note": ""})
    return rows


def early_rows() -> list[dict]:
    rows = []
    for work, run in RUNS.items():
        d = Path("runs/onto-extract") / run
        ev = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
        onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
        facts, texts = {f.id: f for f in onto.facts}, load_texts(load_work(work, SYNTHETIC_ROOT))
        for e in ev["reveal_early"]:
            f = facts[e["extracted"]]
            ch = texts[e["extracted_ch"]].splitlines()
            cited = " / ".join(f"{n}: {ch[n - 1].strip()}" for ev_ in f.evidence for n in ev_.lines)
            rows.append({"work": work, "gold_id": e["gold"], "gold_ch": e["gold_ch"], "gold_statement": e["gold_statement"],
                         "extracted_id": e["extracted"], "extracted_ch": e["extracted_ch"],
                         "extracted_statement": e["extracted_statement"], "cited_lines": cited,
                         "human_leak": "", "human_note": ""})
    return rows


def gold_rows() -> list[dict]:
    rng, rows = random.Random(SEED), []
    quota = dict(zip(RUNS, (13, 13, 12, 12)))
    for work in RUNS:
        gold, texts = from_synthetic(work), load_texts(load_work(work, SYNTHETIC_ROOT))
        names = {c.id: c for c in gold.characters}
        raw = {json.loads(line)["id"]: json.loads(line)
               for line in (SYNTHETIC_ROOT / work / "facts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
        outline = {e["ep"]: e for e in json.loads((SYNTHETIC_ROOT / work / "outline.json").read_text(encoding="utf-8"))}
        for f in gold.facts:
            if f.is_twist:
                keys = raw[f.id].get("evidence_keywords", [])
                rows.append({"kind": "반전 공개 시점", "work": work, "fact_id": f.id, "statement": f.statement,
                             "character": "", "chapter": f.reveal_ch,
                             "question": f"{f.reveal_ch}화 본문에서 이 사실이 처음 분명해지는가",
                             "hint_lines": lines_with(texts[f.reveal_ch], keys), "hint_note": "",
                             "human_ok": "", "human_note": ""})
        pool = [(f, k) for f in gold.facts for k in f.known_by if k.from_ch >= 1 and k.character in names]
        for f, k in rng.sample(pool, quota[work]):
            c = names[k.character]
            keys = raw[f.id].get("evidence_keywords", [])
            note = next((x.get("note", "") for x in raw[f.id]["known_by"] if x["character"] == k.character), "")
            present = k.character in outline.get(k.from_ch, {}).get("characters_present", [])
            rows.append({"kind": "인지 시점", "work": work, "fact_id": f.id, "statement": f.statement,
                         "character": c.name, "chapter": k.from_ch,
                         "question": f"{c.name}(이)가 {k.from_ch}화에서 이 사실을 알게 되는가(직접 겪음, 그 자리에 있음, 들음)",
                         "hint_lines": lines_with(texts[k.from_ch], keys) or lines_with(texts[k.from_ch], c.forms, 2),
                         "hint_note": ("설정상 그 화에 등장" if present else "설정상 그 화 등장인물 명단에 없음")
                                      + (f" · 설정 메모: {note}" if note else ""),
                         "human_ok": "", "human_note": ""})
    return rows


def make() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m, e, g = match_rows(), early_rows(), gold_rows()
    write(OUT / "match-check.csv", list(m[0]), m)
    write(OUT / "early-check.csv", list(e[0]), e)
    write(OUT / "gold-check.csv", list(g[0]), g)
    kinds = Counter(r["kind"] for r in g)
    print(f"{OUT}: match-check {len(m)}행, early-check {len(e)}행, gold-check {len(g)}행({dict(kinds)})")


def kappa(a: list[str], b: list[str]) -> float | None:
    """두 검증자의 예·아니오 일치도(Cohen's kappa)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(v) / n) * (b.count(v) / n) for v in YES_NO)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def read_answers(path: Path) -> tuple[str, list[str]]:
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    col = "human_leak" if "human_leak" in rows[0] else "human_ok"
    return col, [r[col].strip() for r in rows]


def score(paths: list[str]) -> None:
    sheets = [(Path(p), *read_answers(Path(p))) for p in paths]
    for path, col, ans in sheets:
        filled = [a for a in ans if a in YES_NO]
        bad = sum(1 for a in ans if a and a not in YES_NO)
        yes = filled.count("예")
        print(f"{path.name}: {len(filled)}/{len(ans)}행 채움, 예 {yes}, 아니오 {len(filled) - yes}"
              + (f", '예·아니오' 밖의 값 {bad}행" if bad else "")
              + (f" → {col} '예' 비율 {yes / len(filled):.3f}" if filled else ""))
    if len(sheets) == 2:
        (_, _, a), (_, _, b) = sheets
        both = [(x, y) for x, y in zip(a, b) if x in YES_NO and y in YES_NO]
        if both:
            xs, ys = [x for x, _ in both], [y for _, y in both]
            agree = sum(x == y for x, y in both)
            print(f"두 검증자: 함께 채운 {len(both)}행, 일치 {agree}행({agree / len(both):.3f}), 카파 {kappa(xs, ys)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["make"]:
        make()
    elif sys.argv[1:2] == ["score"] and sys.argv[2:]:
        score(sys.argv[2:])
    else:
        sys.exit(__doc__)
