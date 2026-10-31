"""FANToM 외부 검증: 인물 축(누가 무엇을 아는가)을 다른 팀이 만든 정답으로 잰다(docs/paper-plan.md 8절).

FANToM(Kim et al., EMNLP 2023)은 여러 사람의 대화에서 인물이 자리를 비운 사이에 오간 정보를 누가 아는지 묻는다.
대화는 GPT-4가 만들고 사람이 검증했다. 저장소는 MIT이고 평가 용도로만 쓴다. 데이터는 data/external/fantom/에 받는다.

- prepare: 대화를 무작위로 골라 실행 폴더에 표본(sample.json)을 만든다. 모델 호출 없음.
- extract: 대화를 발화 SEGMENT개씩 잘라 장으로 보고 본문 추출기(onto_extract)를 그대로 돌린다.
  누가 나갔는지 같은 정답 정보는 쓰지 않는다. 지시문은 첫 줄만 대화용으로 바꾼다(CONV_SYSTEM).
- answer: 묶음마다 정보 문장과 같은 추출 사실을 판정 모델이 고르고, 그 사실을 아는 인물을 모아 목록 질문에 답한다.
- direct: 같은 모델에 대화와 질문을 FANToM 형식 그대로 주고 답을 받는다(기준선).
- score: 두 방식을 FANToM 채점 그대로 센다. 목록 문항은 아는 인물을 모두 넣고 모르는 인물을 하나도 넣지 않아야 정답이다.
  입력은 두 방식 모두 대화 전체(full)라서, 모르는 인물이 있는 목록 문항은 비대칭(inaccessible)으로 센다(FANToM 규칙).

실행: uv run python -m sfqa.fantom prepare --run runs/fantom/pilot --n 10 --seed 20261101
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from sfqa.onto_extract import (
    LONG_MAX_TOKENS,
    SETTINGS,
    SYSTEM,
    Budget,
    Extractor,
    State,
    build_request,
    cost_usd,
    parse,
)
from sfqa.ontology import Ontology, check
from sfqa.paths import DATA, ROOT

DATA_FILE = DATA / "external" / "fantom" / "fantom_v1.json"
CASSETTE = ROOT / "runs" / "fantom" / "cassette.jsonl"
SEGMENT = 10  # 한 장으로 볼 발화 수. 대화당 평균 39발화라 약 4장이 된다
DIRECT_MAX_TOKENS = 4000  # thinking을 끌 수 없는 모델도 답이 잘리지 않게
# 목록 질문의 답 형식. FANToM 채점은 답 안에 이름이 있는지만 봐서, 요즘 모델이 붙이는 설명("Paul은 자리에 없었다")의
# 이름까지 답으로 센다(11/01 시험: Opus 직접 질의의 첫 줄에 모르는 인물이 든 것은 27개 중 2개인데 답 전체로는 24개).
# 그래서 모든 방법에 이름 목록 JSON을 강제한다. 질문 문장은 FANToM 그대로다.
ANSWER_SCHEMA = {"type": "object", "properties": {"characters": {"type": "array", "items": {"type": "string"}}},
                 "required": ["characters"], "additionalProperties": False}
JUDGE_MAX_TOKENS = 2000
# (질문 종류, 묶음의 칸). 두 목록 질문의 정답 목록은 870묶음 모두 같다(11/01 확인).
LIST_QS = [("tom:info_accessibility:list", "infoAccessibilityQA_list"),
           ("tom:answerability:list", "answerabilityQA_list")]

_NOVEL_LEAD = "너는 한국어 장편소설을 한 장씩 읽으며 '장 시간축 온톨로지'를 쌓는 추출기다."
_CONV_LEAD = ("너는 여러 사람이 나누는 영어 대화를 한 구간씩 읽으며 '장 시간축 온톨로지'를 쌓는 추출기다. "
              "여기서 '장'은 대화의 한 구간(발화 몇 개)이고 '본문'은 그 구간의 발화다. "
              "대화에서 '그 장면에 있었다'는 그 말이 오갈 때 대화 자리에 있었다는 뜻이고, 자리를 비운 사이에 오간 말은 듣지 못한 것이다. "
              "statement는 본문과 같은 언어(영어)로 쓴다.")
CONV_SYSTEM = SYSTEM.replace(_NOVEL_LEAD, _CONV_LEAD, 1)  # v1: 11/01 시험 실행(runs/fantom/pilot*)
# v2: 시험 실행에서 대화 속 의견·제안을 사실로 뽑지 않는 것을 보고 한 문장을 더했다(docs/paper-plan.md 8절).
CONV_SYSTEM_V2 = SYSTEM.replace(_NOVEL_LEAD, _CONV_LEAD + " 대화에서는 누가 어떤 경험, 의견, 제안, 생각을 말했는지도 새 사실이다.", 1)
PROMPTS = {"v1": CONV_SYSTEM, "v2": CONV_SYSTEM_V2}

JUDGE_SYSTEM = """너는 대화에서 뽑은 사실 목록과 정보 문장 하나를 받는다. 정보 문장과 같은 내용을 말하는 사실의 키를 모두 고른다.
- 정보 문장의 일부만 담은 사실도 고른다.
- 같은 화제라도 내용이 다르면 고르지 않는다.
- 맞는 사실이 없으면 빈 배열을 낸다."""
JUDGE_SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}},
                                                 "reason": {"type": "string"}},
                "required": ["ids", "reason"], "additionalProperties": False}


# ---------- 데이터 ----------


def load_sets(data: Path = DATA_FILE) -> list[dict]:
    return json.loads(data.read_text(encoding="utf-8"))


def conversations(sets: list[dict]) -> dict[str, dict]:
    """대화 id → {id, context, sets}. 같은 full_context의 묶음을 모은다.

    conv_id 10은 서로 다른 대화 둘에 붙어 있어서(11/01 확인), 뒤에 나온 대화는 c10-2로 부른다.
    """
    by_text: dict[str, dict] = {}
    used: set[str] = set()
    for s in sets:
        conv = by_text.get(s["full_context"])
        if conv is None:
            cid, n = f"c{s['conv_id']}", 2
            while cid in used:
                cid, n = f"c{s['conv_id']}-{n}", n + 1
            used.add(cid)
            conv = by_text[s["full_context"]] = {"id": cid, "context": s["full_context"], "sets": []}
        conv["sets"].append(s)
    return {c["id"]: c for c in by_text.values()}


def segments(context: str, size: int = SEGMENT) -> list[str]:
    """빈 줄을 뺀 발화를 size개씩 묶는다. 누가 나가고 들어오는지는 보지 않는다(정답 정보를 쓰지 않게)."""
    lines = [line for line in context.strip().split("\n") if line.strip()]
    return ["\n".join(lines[i:i + size]) for i in range(0, len(lines), size)]


def prepare(run_dir: Path, n: int, seed: int, data: Path = DATA_FILE) -> dict:
    convs = conversations(load_sets(data))
    ids = sorted(convs)
    pick = sorted(random.Random(seed).sample(ids, n)) if n < len(ids) else ids
    sample = {"data": str(data), "seed": seed, "segment": SEGMENT, "conversations": pick,
              "sets": [s["set_id"] for cid in pick for s in convs[cid]["sets"]]}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    return sample


def load_sample(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    """(표본, 표본에 든 대화)."""
    sample = json.loads((run_dir / "sample.json").read_text(encoding="utf-8"))
    convs = conversations(load_sets(Path(sample["data"])))
    return sample, {cid: convs[cid] for cid in sample["conversations"]}


# ---------- 추출 ----------


def extract_conversation(conv: dict, model: str, ex: Extractor, out_dir: Path, prompt: str = "v2") -> dict:
    """구간을 1장부터 차례로 추출한다. onto_extract.run과 같은 순서다(잘리면 출력 상한을 늘려 한 번 더, 두 번 실패하면 멈춤)."""
    segs = segments(conv["context"])
    state = State(work=conv["id"], title=f"FANToM {conv['id']}", chapters=len(segs))
    report = {"conversation": conv["id"], "segments": len(segs), "prompt": prompt, "done": 0, "stopped": None,
              "chapters": []}
    calls, spent = ex.calls, ex.spent
    for c, text in enumerate(segs, 1):
        req = build_request(model, state.blocks, c, text)
        req["system"][0]["text"] = PROMPTS[prompt]  # 소설 실행의 요청은 그대로 두고 여기서만 바꾼다
        tries, out = [], None
        try:
            for attempt in (0, 1):
                cut = bool(tries) and tries[-1]["error"] == "finish_reason=length"
                resp, fresh = ex.complete({**req, "max_tokens": LONG_MAX_TOKENS} if cut else req, attempt)
                out, err = parse(resp)
                tries.append({"attempt": attempt, "fresh": fresh, "error": err, "cost": round(cost_usd(resp), 5),
                              "input": resp.input_tokens, "cache_read": resp.extra.get("cache_read", 0),
                              "cache_write": resp.extra.get("cache_write", 0), "output": resp.output_tokens,
                              "raw": resp.raw_text if err else None})
                if out is not None:
                    break
        except Budget as e:
            report["stopped"] = str(e)
            report["chapters"].append({"chapter": c, "tries": tries})
            break
        if out is None:
            report["stopped"] = f"{c}구간 두 번 실패"
            report["chapters"].append({"chapter": c, "tries": tries})
            break
        summary = state.apply(c, out, text)
        report["chapters"].append({"chapter": c, "tries": tries, **summary, "output": out.model_dump()})
        report["done"] = c
    onto = state.ontology(source=f"fantom-extract:{model}")
    report["problems"] = check(onto)
    report["calls"], report["spent_usd"] = ex.calls - calls, round(ex.spent - spent, 4)
    report["rejected"], report["notes"] = state.rejected, state.notes
    d = out_dir / conv["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "ontology.json").write_text(onto.model_dump_json(indent=1), encoding="utf-8")
    (d / "record.txt").write_text("\n\n".join(state.blocks), encoding="utf-8")
    (d / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def run_extract(run_dir: Path, model: str, *, max_usd: float, max_calls: int = 300, sdk=None,
                cassette: Path | None = CASSETTE, prompt: str = "v2") -> dict:
    if model not in SETTINGS:
        raise ValueError(f"모르는 모델: {model}")
    _, convs = load_sample(run_dir)
    ex = Extractor(cassette, sdk=sdk, max_calls=max_calls, max_usd=max_usd)
    summary = {"model": model, "prompt": prompt, "conversations": [], "stopped": None}
    for conv in convs.values():
        r = extract_conversation(conv, model, ex, run_dir / "extract", prompt)
        summary["conversations"].append({k: r[k] for k in ("conversation", "segments", "done", "calls", "spent_usd",
                                                             "stopped")})
        if r["stopped"]:
            summary["stopped"] = f"{conv['id']}: {r['stopped']}"
            break
    summary["calls"], summary["spent_usd"] = ex.calls, round(ex.spent, 4)
    (run_dir / "extract.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# ---------- 추출 결과로 답하기 ----------


def info_text(s: dict) -> str:
    """FANToM이 정보 질문에 붙이는 정보 문장(사실 질문 + 정답)."""
    return f"{s['factQA']['question']} {s['factQA']['correct_answer']}"


def judge_request(model: str, onto: Ontology, info: str) -> dict:
    thinking, effort = SETTINGS[model]
    listing = "\n".join(f"{f.id}: {f.statement}" for f in onto.facts)
    return {"model": model, "max_tokens": JUDGE_MAX_TOKENS,
            "system": [{"type": "text", "text": JUDGE_SYSTEM}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": f"<사실>\n{listing}\n</사실>\n\n정보: {info}"}]}],
            "thinking": thinking,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}}}


def knowers(onto: Ontology, ids: list[str]) -> tuple[list[str], list[str]]:
    """대화 끝에서 (하나라도: 고른 사실 가운데 하나라도 아는 인물, 모두: 고른 사실을 모두 아는 인물)의 이름.

    상식으로 표시된 사실은 모든 인물이 안다고 본다(ontology.split_facts와 같은 규칙).
    """
    names = {c.id: c.name for c in onto.characters}
    facts = {f.id: f for f in onto.facts}
    sets = [set(names) if facts[i].common else {k.character for k in facts[i].known_by if k.character in names}
            for i in ids]
    if not sets:
        return [], []
    return sorted(names[c] for c in set().union(*sets)), sorted(names[c] for c in set.intersection(*sets))


def run_answer(run_dir: Path, model: str, *, max_usd: float, max_calls: int = 1000, sdk=None,
               cassette: Path | None = CASSETTE) -> dict:
    if model not in SETTINGS:
        raise ValueError(f"모르는 모델: {model}")
    _, convs = load_sample(run_dir)
    done = {c["conversation"]: c for c in json.loads((run_dir / "extract.json").read_text(encoding="utf-8"))["conversations"]}
    ex = Extractor(cassette, sdk=sdk, max_calls=max_calls, max_usd=max_usd)
    answers, stopped = [], None
    for cid, conv in convs.items():
        row = done.get(cid)
        if row is None or row["done"] < row["segments"]:
            why = "추출 안 함" if row is None else "추출 미완"
            answers += [{"set_id": s["set_id"], "conversation": cid, "skipped": why} for s in conv["sets"]]
            continue
        onto = Ontology.model_validate_json((run_dir / "extract" / cid / "ontology.json").read_text(encoding="utf-8"))
        for s in conv["sets"]:
            ids, note = [], None
            if onto.facts:
                try:
                    resp, _ = ex.complete(judge_request(model, onto, info_text(s)))
                except Budget as e:
                    stopped = str(e)
                    break
                try:
                    picked = json.loads(resp.raw_text or "")["ids"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    picked, note = [], f"판정 해석 실패: finish_reason={resp.finish_reason}"
                known = {f.id for f in onto.facts}
                ids = [i for i in picked if i in known]
                if len(ids) < len(picked):
                    note = f"없는 사실 키: {sorted(set(picked) - known)}"
            any_names, all_names = knowers(onto, ids)
            answers.append({"set_id": s["set_id"], "conversation": cid, "ids": ids, "any": any_names, "all": all_names,
                            "note": note})
        if stopped:
            break
    result = {"model": model, "stopped": stopped, "calls": ex.calls, "spent_usd": round(ex.spent, 4), "answers": answers}
    (run_dir / "answers.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ---------- 직접 질의(기준선) ----------


def direct_parts(s: dict, kind: str) -> tuple[str, str]:
    """(대화, 뒤에 붙는 질문). 둘을 이으면 FANToM eval_fantom.setup_fantom의 full 입력과 같다."""
    fq, fa = s["factQA"]["question"], s["factQA"]["correct_answer"]
    if kind == "tom:info_accessibility:list":
        tail = f"\n\nInformation: {fq} {fa}\nQuestion: {s['infoAccessibilityQA_list']['question']}\nAnswer:"
    else:
        tail = f"\n\nTarget: {fq}\nQuestion: {s['answerabilityQA_list']['question']}\nAnswer:"
    return s["full_context"].strip(), tail


def direct_request(model: str, context: str, tail: str, fmt: str = "json") -> dict:
    """한 사용자 메시지(FANToM 방식). 대화 부분에 캐시 지점을 둬서 같은 대화의 다른 질문이 캐시를 읽는다.

    FANToM은 temperature 0을 쓰지만, thinking을 끌 수 없는 모델이 있어 추출과 같은 thinking·effort 설정을 쓴다.
    fmt가 json이면 답을 이름 목록(ANSWER_SCHEMA)으로 받는다. text는 11/01 시험 실행의 자유 형식이다.
    """
    thinking, effort = SETTINGS[model]
    output = {"effort": effort}
    if fmt == "json":
        output["format"] = {"type": "json_schema", "schema": ANSWER_SCHEMA}
    return {"model": model, "max_tokens": DIRECT_MAX_TOKENS,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": tail}]}],
            "thinking": thinking, "output_config": output}


def parse_answer(text: str | None) -> str:
    """FANToM parse_response와 같다: 'Answer:'가 있으면 그 뒤만 쓴다."""
    text = text or ""
    return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()


def parse_names(text: str | None) -> str:
    """JSON 답의 이름을 쉼표로 잇는다. 형식이 깨지면 자유 형식처럼 읽는다(채점에서 틀릴 수 있다)."""
    try:
        return ", ".join(json.loads(text or "")["characters"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return parse_answer(text)


def run_direct(run_dir: Path, model: str, *, max_usd: float, max_calls: int = 1000, sdk=None,
               cassette: Path | None = CASSETTE, fmt: str = "json") -> dict:
    if model not in SETTINGS:
        raise ValueError(f"모르는 모델: {model}")
    _, convs = load_sample(run_dir)
    ex = Extractor(cassette, sdk=sdk, max_calls=max_calls, max_usd=max_usd)
    read = parse_names if fmt == "json" else parse_answer
    responses, stopped = [], None
    for conv in convs.values():
        for s in conv["sets"]:
            for kind, _ in LIST_QS:
                try:
                    resp, _ = ex.complete(direct_request(model, *direct_parts(s, kind), fmt))
                except Budget as e:
                    stopped = str(e)
                    break
                responses.append({"set_id": s["set_id"], "kind": kind, "answer": read(resp.raw_text),
                                  "finish_reason": resp.finish_reason, "output": resp.output_tokens})
            if stopped:
                break
        if stopped:
            break
    result = {"model": model, "format": fmt, "stopped": stopped, "calls": ex.calls, "spent_usd": round(ex.spent, 4),
              "responses": responses}
    (run_dir / "direct.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ---------- 추출 기록으로 답하기(구조 질의) ----------

STRUCT_LEAD = ("Below are facts extracted from a conversation, in the order they came up. "
               "After each fact are the characters who know it by the end of the conversation.\n\n")


def struct_record(onto: Ontology) -> str:
    """원문 대신 줄 추출 기록. 상식으로 표시된 사실은 모든 인물이 안다(knowers와 같은 규칙)."""
    names = {c.id: c.name for c in onto.characters}
    lines = []
    for f in onto.facts:
        who = "everyone" if f.common else ", ".join(names[k.character] for k in f.known_by if k.character in names)
        lines.append(f"- {f.statement} (known by: {who or 'no one'})")
    return STRUCT_LEAD + "\n".join(lines)


def run_struct(run_dir: Path, model: str, *, max_usd: float, max_calls: int = 1000, sdk=None,
               cassette: Path | None = CASSETTE, fmt: str = "json") -> dict:
    """같은 모델이 원문 대신 추출 기록을 읽고 FANToM 질문에 답한다. 질문 부분과 답 형식은 직접 질의와 같다."""
    if model not in SETTINGS:
        raise ValueError(f"모르는 모델: {model}")
    _, convs = load_sample(run_dir)
    done = {c["conversation"]: c for c in json.loads((run_dir / "extract.json").read_text(encoding="utf-8"))["conversations"]}
    ex = Extractor(cassette, sdk=sdk, max_calls=max_calls, max_usd=max_usd)
    read = parse_names if fmt == "json" else parse_answer
    responses, skipped, stopped = [], [], None
    for cid, conv in convs.items():
        row = done.get(cid)
        if row is None or row["done"] < row["segments"]:
            skipped += [s["set_id"] for s in conv["sets"]]
            continue
        record = struct_record(Ontology.model_validate_json(
            (run_dir / "extract" / cid / "ontology.json").read_text(encoding="utf-8")))
        for s in conv["sets"]:
            for kind, _ in LIST_QS:
                try:
                    resp, _ = ex.complete(direct_request(model, record, direct_parts(s, kind)[1], fmt))
                except Budget as e:
                    stopped = str(e)
                    break
                responses.append({"set_id": s["set_id"], "kind": kind, "answer": read(resp.raw_text),
                                  "finish_reason": resp.finish_reason, "output": resp.output_tokens})
            if stopped:
                break
        if stopped:
            break
    result = {"model": model, "format": fmt, "stopped": stopped, "calls": ex.calls, "spent_usd": round(ex.spent, 4),
              "skipped": skipped, "responses": responses}
    (run_dir / "struct.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ---------- 채점 ----------


def list_result(correct: list[str], wrong: list[str], answer: str) -> tuple[bool, bool, bool]:
    """FANToM evaluate_list_q 그대로: (정답, 아는 인물을 뺌, 모르는 인물을 넣음). 이름이 답 안에 있는지만 본다."""
    text = answer.lower()
    excluded = any(c.lower() not in text for c in correct)
    included = any(w.lower() in text for w in wrong)
    return not (excluded or included), excluded, included


def scenario(qa: dict) -> str:
    """full 입력 규칙(eval_fantom.setup_fantom): 모르는 인물이 있는 목록 문항은 inaccessible."""
    return "inaccessible" if qa["wrong_answer"] else qa["missed_info_accessibility"]


COUNTS = ("n", "correct", "excluded", "included", "tp", "fp", "fn")
PAIRED = "struct-vs-direct"  # 같은 문항에서 구조 질의만 맞힘(a_only)·직접 질의만 맞힘(b_only)


def outcomes(sets: list[dict], answers: dict[tuple[str, str], str]) -> dict[tuple[str, str], dict]:
    """answers: (set_id, 질문 종류) → 답 문자열. 문항마다 칸(종류:상황), 정답 여부, 틀린 이유, 인물 쌍 개수."""
    out = {}
    for s in sets:
        for kind, field in LIST_QS:
            if (s["set_id"], kind) not in answers:
                continue
            qa, answer = s[field], answers[(s["set_id"], kind)]
            ok, excluded, included = list_result(qa["correct_answer"], qa["wrong_answer"], answer)
            named = {n for n in qa["correct_answer"] + qa["wrong_answer"] if n.lower() in answer.lower()}
            out[(s["set_id"], kind)] = {
                "cell": f"{kind.split(':')[1]}:{scenario(qa)}", "n": 1, "correct": ok, "excluded": excluded,
                "included": included, "tp": len(named & set(qa["correct_answer"])),
                "fp": len(named & set(qa["wrong_answer"])), "fn": len(set(qa["correct_answer"]) - named)}
    return out


def _ratios(cell: dict) -> dict:
    cell["accuracy"] = round(cell["correct"] / cell["n"], 4)
    cell["precision"] = round(cell["tp"] / (cell["tp"] + cell["fp"]), 4) if cell["tp"] + cell["fp"] else None
    cell["recall"] = round(cell["tp"] / (cell["tp"] + cell["fn"]), 4) if cell["tp"] + cell["fn"] else None
    return cell


def score(sets: list[dict], answers: dict[tuple[str, str], str]) -> dict:
    """종류 × 상황마다 정답률, 틀린 이유, 인물 쌍 정밀도·재현율."""
    cells: dict[str, dict] = {}
    for o in outcomes(sets, answers).values():
        cell = cells.setdefault(o["cell"], dict.fromkeys(COUNTS, 0))
        for k in COUNTS:
            cell[k] += o[k]
    return {name: _ratios(cell) for name, cell in sorted(cells.items())}


def mcnemar(b: int, c: int) -> float:
    """정확 McNemar(양측). 엇갈린 문항 b, c가 반반으로 나뉜다는 가정의 이항 검정."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(sets: list[dict], a: dict[tuple[str, str], str], b: dict[tuple[str, str], str]) -> dict:
    """같은 문항에서 a만 맞힌 수, b만 맞힌 수, 정확 McNemar p."""
    oa, ob = outcomes(sets, a), outcomes(sets, b)
    cells: dict[str, dict] = {}
    for key in sorted(oa.keys() & ob.keys()):
        cell = cells.setdefault(oa[key]["cell"], {"n": 0, "a_only": 0, "b_only": 0})
        cell["n"] += 1
        cell["a_only"] += oa[key]["correct"] and not ob[key]["correct"]
        cell["b_only"] += ob[key]["correct"] and not oa[key]["correct"]
    for cell in cells.values():
        cell["p"] = round(mcnemar(cell["a_only"], cell["b_only"]), 4)
    return dict(sorted(cells.items()))


def run_score(run_dir: Path) -> dict:
    sample, convs = load_sample(run_dir)
    sets = [s for conv in convs.values() for s in conv["sets"]]
    result, answers = {}, {}
    if (run_dir / "answers.json").exists():
        rows = [a for a in json.loads((run_dir / "answers.json").read_text(encoding="utf-8"))["answers"]
                if "skipped" not in a]
        for basis in ("any", "all"):
            result[f"ours:{basis}"] = score(sets, {(a["set_id"], kind): ", ".join(a[basis])
                                                   for a in rows for kind, _ in LIST_QS})
    for name, method in (("struct.json", "ours:struct"), ("direct.json", "direct")):
        if (run_dir / name).exists():
            rows = json.loads((run_dir / name).read_text(encoding="utf-8"))["responses"]
            answers[method] = {(r["set_id"], r["kind"]): r["answer"] for r in rows}
            result[method] = score(sets, answers[method])
    if len(answers) == 2:
        result[PAIRED] = paired(sets, answers["ours:struct"], answers["direct"])
    (run_dir / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def combine(results: list[dict]) -> dict:
    """샤드마다 낸 run_score 결과를 합친다. 개수는 더하고 비율과 p는 다시 낸다."""
    total: dict[str, dict] = {}
    for result in results:
        for method, cells in result.items():
            keys = ("n", "a_only", "b_only") if method == PAIRED else COUNTS
            for name, cell in cells.items():
                t = total.setdefault(method, {}).setdefault(name, dict.fromkeys(keys, 0))
                for k in keys:
                    t[k] += cell[k]
    for method, cells in total.items():
        for cell in cells.values():
            if method == PAIRED:
                cell["p"] = round(mcnemar(cell["a_only"], cell["b_only"]), 4)
            else:
                _ratios(cell)
        total[method] = dict(sorted(cells.items()))
    return total


def split(run_dir: Path, k: int) -> list[Path]:
    """표본을 대화 단위로 번갈아 k개로 나눠 run_dir/s1..sk에 둔다(동시에 돌리려고). 샤드마다 카세트를 따로 쓴다."""
    sample = json.loads((run_dir / "sample.json").read_text(encoding="utf-8"))
    convs = conversations(load_sets(Path(sample["data"])))
    dirs = []
    for i in range(k):
        ids = sample["conversations"][i::k]
        d = run_dir / f"s{i + 1}"
        d.mkdir(parents=True, exist_ok=True)
        part = {**sample, "shard": f"{i + 1}/{k}", "conversations": ids,
                "sets": [s["set_id"] for cid in ids for s in convs[cid]["sets"]]}
        (d / "sample.json").write_text(json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
        dirs.append(d)
    return dirs


def _print_score(result: dict) -> None:
    for method, cells in result.items():
        print(method)
        for name, c in cells.items():
            if method == PAIRED:
                print(f"  {name}: {c['n']}문항, 구조 질의만 맞힘 {c['a_only']}, 직접 질의만 맞힘 {c['b_only']}, McNemar p {c['p']}")
            else:
                print(f"  {name}: {c['correct']}/{c['n']} ({c['accuracy']:.3f}), 아는 인물 뺌 {c['excluded']}, "
                      f"모르는 인물 넣음 {c['included']}, 쌍 정밀도 {c['precision']} 재현율 {c['recall']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sfqa.fantom", description="FANToM으로 인물 축을 잰다")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="대화를 무작위로 골라 표본을 만든다(모델 호출 없음)")
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    for name, help_text in (("extract", "구간마다 추출한다(유료)"),
                            ("answer", "판정 모델이 고른 추출 사실의 아는 인물로 답한다(유료)"),
                            ("struct", "같은 모델이 추출 기록을 읽고 답한다(유료)"),
                            ("direct", "같은 모델에 원문으로 직접 묻는다(유료)")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run", required=True)
        p.add_argument("--model", required=True, choices=list(SETTINGS))
        p.add_argument("--max-usd", type=float, required=True, help="이 실행에서 새로 쓸 수 있는 최대 금액")
        p.add_argument("--max-calls", type=int, default=1000)
        p.add_argument("--cassette", help=f"카세트 경로(기본 {CASSETTE}). iCloud 밖에 두면 덧붙인 줄이 사라지지 않는다")
        if name == "extract":
            p.add_argument("--prompt", choices=list(PROMPTS), default="v2", help="추출 지시문(v1은 11/01 시험 실행)")
        if name in ("struct", "direct"):
            p.add_argument("--format", choices=["json", "text"], default="json",
                           help="답 형식(text는 11/01 시험 실행의 자유 형식, 설명 속 이름까지 채점된다)")
    p = sub.add_parser("split", help="표본을 샤드 k개로 나눈다(모델 호출 없음)")
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, required=True)
    p = sub.add_parser("score", help="FANToM 채점(모델 호출 없음). 샤드 여러 개를 주면 합쳐서 센다")
    p.add_argument("--run", required=True, nargs="+")
    p.add_argument("--out", help="합친 결과 파일(샤드 여러 개일 때, 기본은 첫 샤드의 부모 폴더 score.json)")
    args = parser.parse_args(argv)

    if args.cmd == "score":
        dirs = [Path(r) for r in args.run]
        if len(dirs) == 1:
            _print_score(run_score(dirs[0]))
            return 0
        result = combine([run_score(d) for d in dirs])
        out = Path(args.out) if args.out else dirs[0].parent / "score.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        _print_score(result)
        print(f"합친 결과: {out}")
        return 0
    run_dir = Path(args.run)
    if args.cmd == "prepare":
        sample = prepare(run_dir, args.n, args.seed)
        print(f"{run_dir}: 대화 {len(sample['conversations'])}개, 묶음 {len(sample['sets'])}개")
        return 0
    if args.cmd == "split":
        for d in split(run_dir, args.k):
            s = json.loads((d / "sample.json").read_text(encoding="utf-8"))
            print(f"{d}: 대화 {len(s['conversations'])}개, 묶음 {len(s['sets'])}개")
        return 0
    run = {"extract": run_extract, "answer": run_answer, "struct": run_struct, "direct": run_direct}[args.cmd]
    extra = ({"prompt": args.prompt} if args.cmd == "extract" else
             {"fmt": args.format} if args.cmd in ("struct", "direct") else {})
    result = run(run_dir, args.model, max_usd=args.max_usd, max_calls=args.max_calls,
                 cassette=Path(args.cassette) if args.cassette else CASSETTE, **extra)
    print(f"{args.cmd} {args.model}: 새 호출 {result['calls']}회, ${result['spent_usd']:.4f}"
          + (f", 멈춤: {result['stopped']}" if result["stopped"] else ""))
    if args.cmd == "extract":
        for c in result["conversations"]:
            print(f"  {c['conversation']}: {c['done']}/{c['segments']}구간, 새 호출 {c['calls']}회, ${c['spent_usd']:.4f}")
    return 1 if result["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
