import json
from types import SimpleNamespace

import pytest

from sfqa.fantom import (
    _NOVEL_LEAD,
    ANSWER_SCHEMA,
    PAIRED,
    combine,
    mcnemar,
    split,
    CONV_SYSTEM,
    CONV_SYSTEM_V2,
    STRUCT_LEAD,
    conversations,
    direct_parts,
    knowers,
    list_result,
    parse_answer,
    parse_names,
    prepare,
    run_answer,
    run_direct,
    run_extract,
    run_score,
    run_struct,
    scenario,
    segments,
    struct_record,
)
from sfqa.onto_extract import SYSTEM, build_request
from sfqa.ontology import Character, Fact, Knowing, Ontology

CONV_A = "\n".join([
    "Ann: Hi Bob, hi Cy, nice to meet you both.",
    "Bob: Nice to meet you too, Ann.",
    "Cy: Sorry, I need to grab a coffee. Be right back!",
    "Ann: Sure. So Bob, do you have any pets?",
    "Bob: Yes, my dog is named Rex and he loves the beach.",
    "Ann: Rex sounds adorable.",
    "Bob: He really is.",
    "Ann: I have a cat myself.",
    "Bob: Cats are great.",
    "Ann: Indeed.",
    "Cy: I'm back! What did I miss?",
    "Bob: Not much, just small talk.",
])
CONV_B = "Dee: Hi Eve, I just moved here from Lisbon.\nEve: Welcome! I grew up in Lisbon too.\nDee: What a coincidence."


def a_set(set_id, conv_id, context, fact_q, fact_a, correct, wrong, accessibility) -> dict:
    def qa(question, qtype):
        return {"question": question, "question_type": qtype, "correct_answer": correct, "wrong_answer": wrong,
                "missed_info_accessibility": accessibility}

    return {"set_id": set_id, "part_id": f"{conv_id}-0", "conv_id": conv_id, "full_context": context,
            "short_context": context, "missed_info": "", "joining_speaker": "",
            "factQA": {"question": fact_q, "question_type": "fact", "correct_answer": fact_a, "wrong_answer": ""},
            "beliefQAs": [],
            "infoAccessibilityQA_list": qa("List all the characters who know this information.",
                                           "tom:info_accessibility:list"),
            "answerabilityQA_list": qa("List all the characters who know the precise correct answer to this question.",
                                       "tom:answerability:list"),
            "infoAccessibilityQAs_binary": [], "answerabilityQAs_binary": []}


SETS = [a_set("0-0-0", 0, CONV_A, "What is the name of Bob's dog?", "Bob's dog is named Rex.", ["Ann", "Bob"], ["Cy"],
              "inaccessible"),
        a_set("1-0-0", 1, CONV_B, "Where did Dee move from?", "Dee moved here from Lisbon.", ["Dee", "Eve"], [],
              "accessible")]


def seg_out(characters=(), facts=(), present=()) -> dict:
    return {"characters": [{"id": c.lower(), "name": c, "appeared": True, "aliases": []} for c in characters],
            "aliases": [], "merges": [], "learned": [],
            "facts": [{"statement": s, "category": "event", "quote": q, "common": False,
                       "knowers": [{"character": c, "since": n} for c, n in k]} for s, q, k in facts],
            "scene": {"pov": None, "location": "party", "present": list(present)}}


EXTRACT = [
    seg_out(["Ann", "Bob", "Cy"], [("Bob's dog is named Rex.", "my dog is named Rex and he loves the beach",
                                    [("bob", 0), ("ann", 1)])], ["ann", "bob", "cy"]),
    seg_out(present=["cy", "bob"]),
    seg_out(["Dee", "Eve"], [("Dee moved here from Lisbon.", "I just moved here from Lisbon", [("dee", 0), ("eve", 1)])],
            ["dee", "eve"]),
]


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeSDK:
    """요청을 기록하고, 준비한 응답을 차례로 돌려준다. 응답이 모자라면 실패한다(카세트 재생 확인용)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def stream(self, **req):
        self.requests.append(req)
        return FakeStream(self.replies.pop(0))


def msg(payload) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason="end_turn", usage=usage, _request_id="req_test")


@pytest.fixture
def run_dir(tmp_path):
    data = tmp_path / "fantom_v1.json"
    data.write_text(json.dumps(SETS), encoding="utf-8")
    d = tmp_path / "run"
    prepare(d, n=5, seed=1, data=data)
    return d


def test_conv_system_changes_only_the_lead_line():
    assert SYSTEM.split("\n", 1)[0] == _NOVEL_LEAD
    assert CONV_SYSTEM.split("\n", 1)[1] == SYSTEM.split("\n", 1)[1]
    assert _NOVEL_LEAD not in CONV_SYSTEM and "자리를 비운 사이에 오간 말은 듣지 못한 것" in CONV_SYSTEM
    # 소설 추출 요청은 그대로다(본 실험 카세트 키가 바뀌지 않게)
    assert build_request("claude-opus-5-5", [], 1, "본문")["system"][0]["text"] == SYSTEM
    # v2는 v1 첫 줄에 대화 속 의견·제안도 사실이라는 한 문장만 더했다
    v1_lead, v2_lead = CONV_SYSTEM.split("\n", 1)[0], CONV_SYSTEM_V2.split("\n", 1)[0]
    assert v2_lead == v1_lead + " 대화에서는 누가 어떤 경험, 의견, 제안, 생각을 말했는지도 새 사실이다."
    assert CONV_SYSTEM_V2.split("\n", 1)[1] == SYSTEM.split("\n", 1)[1]


def test_segments_cut_every_ten_utterances_and_skip_blank_lines():
    text = "\n".join(f"P{i % 3}: line {i}" for i in range(25)) + "\n\n"
    assert [len(s.split("\n")) for s in segments(text)] == [10, 10, 5]
    assert segments(CONV_A)[1] == "Cy: I'm back! What did I miss?\nBob: Not much, just small talk."


def test_conversations_split_a_reused_conv_id():
    other = {**SETS[1], "conv_id": 0, "set_id": "0-1-0"}
    assert sorted(conversations([SETS[0], other, SETS[0]])) == ["c0", "c0-2"]
    assert len(conversations(SETS)["c0"]["sets"]) == 1


def test_list_result_follows_fantom():
    assert list_result(["Ann", "Bob"], ["Cy"], "Ann and Bob") == (True, False, False)
    assert list_result(["Ann", "Bob"], ["Cy"], "Ann, Bob, and Cy") == (False, False, True)
    assert list_result(["Ann", "Bob"], ["Cy"], "Ann") == (False, True, False)
    assert list_result(["Ann", "Bob"], ["Cy"], "") == (False, True, False)


def test_scenario_uses_the_full_input_rule():
    assert scenario({"wrong_answer": ["Cy"], "missed_info_accessibility": "accessible"}) == "inaccessible"
    assert scenario({"wrong_answer": [], "missed_info_accessibility": "accessible"}) == "accessible"


def test_direct_parts_rebuild_the_fantom_input():
    s = SETS[0]
    context, tail = direct_parts(s, "tom:info_accessibility:list")
    assert context + tail == (f"{CONV_A}\n\nInformation: What is the name of Bob's dog? Bob's dog is named Rex.\n"
                              "Question: List all the characters who know this information.\nAnswer:")
    context, tail = direct_parts(s, "tom:answerability:list")
    assert tail == ("\n\nTarget: What is the name of Bob's dog?\n"
                    "Question: List all the characters who know the precise correct answer to this question.\nAnswer:")
    assert parse_answer("Let me think.\nAnswer: Ann and Bob") == "Ann and Bob"
    assert parse_answer(None) == ""
    # 설명 속 이름이 채점에 섞이지 않게 이름 목록 JSON을 읽는다. 깨지면 자유 형식처럼 읽는다
    assert parse_names('{"characters": ["Ann", "Bob"]}') == "Ann, Bob"
    assert parse_names("Answer: Ann and Bob") == "Ann and Bob"


def test_knowers_any_and_all_with_common_facts():
    onto = Ontology(work="t", title="t", chapters=2, source="test",
                    characters=[Character(id=c, name=c.title(), first_ch=1) for c in ("ann", "bob", "cy")],
                    facts=[Fact(id="c1-1", statement="a", reveal_ch=1,
                                known_by=[Knowing(character="ann", from_ch=1), Knowing(character="bob", from_ch=0)]),
                           Fact(id="c1-2", statement="b", reveal_ch=1, known_by=[Knowing(character="bob", from_ch=1)]),
                           Fact(id="c2-1", statement="c", reveal_ch=2, common=True)])
    assert knowers(onto, ["c1-1", "c1-2"]) == (["Ann", "Bob"], ["Bob"])
    assert knowers(onto, ["c2-1"]) == (["Ann", "Bob", "Cy"], ["Ann", "Bob", "Cy"])
    assert knowers(onto, []) == ([], [])


def test_pipeline_extract_answer_direct_score(run_dir, tmp_path):
    cassette = tmp_path / "cassette.jsonl"
    sdk = FakeSDK([msg(o) for o in EXTRACT])
    summary = run_extract(run_dir, "claude-opus-5-5", max_usd=1.0, sdk=sdk, cassette=cassette)
    assert summary["stopped"] is None and summary["calls"] == 3 and summary["prompt"] == "v2"
    assert [(c["conversation"], c["done"], c["segments"]) for c in summary["conversations"]] == [("c0", 2, 2), ("c1", 1, 1)]
    assert all(r["system"][0]["text"] == CONV_SYSTEM_V2 for r in sdk.requests)
    onto = Ontology.model_validate_json((run_dir / "extract" / "c0" / "ontology.json").read_text(encoding="utf-8"))
    assert {k.character: k.from_ch for k in onto.facts[0].known_by} == {"bob": 0, "ann": 1}

    # 다시 돌리면 카세트에서 재생하고 새로 부르지 않는다
    assert run_extract(run_dir, "claude-opus-5-5", max_usd=1.0, sdk=FakeSDK([]), cassette=cassette)["calls"] == 0

    judge = FakeSDK([msg({"ids": ["c1-1", "c9-9"], "reason": "같은 내용"}), msg({"ids": ["c1-1"], "reason": "같은 내용"})])
    answers = run_answer(run_dir, "claude-sonnet-5-5", max_usd=1.0, sdk=judge, cassette=cassette)["answers"]
    assert [(a["set_id"], a["ids"], a["any"]) for a in answers] == [("0-0-0", ["c1-1"], ["Ann", "Bob"]),
                                                                     ("1-0-0", ["c1-1"], ["Dee", "Eve"])]
    assert answers[0]["note"] == "없는 사실 키: ['c9-9']"
    assert "정보: What is the name of Bob's dog? Bob's dog is named Rex." in judge.requests[0]["messages"][0]["content"][0]["text"]

    direct = FakeSDK([msg({"characters": ["Ann", "Bob", "Cy"]}), msg("Ann and Bob"), msg({"characters": ["Dee", "Eve"]}),
                      msg("Dee")])
    run_direct(run_dir, "claude-opus-5-5", max_usd=1.0, sdk=direct, cassette=cassette)
    first = direct.requests[0]["messages"][0]["content"]
    assert first[0]["text"] == CONV_A and first[0]["cache_control"] == {"type": "ephemeral"}
    assert direct.requests[0]["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}

    record = struct_record(onto)
    assert record == STRUCT_LEAD + "- Bob's dog is named Rex. (known by: Bob, Ann)"
    struct = FakeSDK([msg("Ann and Bob"), msg("Ann and Bob"), msg("Dee, Eve"), msg("Dee and Eve")])
    run_struct(run_dir, "claude-opus-5-5", max_usd=1.0, sdk=struct, cassette=cassette)
    first = struct.requests[0]["messages"][0]["content"]
    assert first[0]["text"] == record and first[1]["text"] == direct_parts(SETS[0], "tom:info_accessibility:list")[1]

    result = run_score(run_dir)
    assert all(c["correct"] == c["n"] for c in result["ours:struct"].values())
    assert {k: (v["a_only"], v["b_only"]) for k, v in result[PAIRED].items()} == {
        "answerability:accessible": (1, 0), "answerability:inaccessible": (0, 0),
        "info_accessibility:accessible": (0, 0), "info_accessibility:inaccessible": (1, 0)}
    ours = result["ours:any"]
    assert {k: (v["correct"], v["n"]) for k, v in ours.items()} == {
        "answerability:accessible": (1, 1), "answerability:inaccessible": (1, 1),
        "info_accessibility:accessible": (1, 1), "info_accessibility:inaccessible": (1, 1)}
    d = result["direct"]
    assert (d["info_accessibility:inaccessible"]["correct"], d["info_accessibility:inaccessible"]["included"]) == (0, 1)
    assert (d["info_accessibility:inaccessible"]["precision"], d["info_accessibility:inaccessible"]["recall"]) == (0.6667, 1.0)
    assert (d["answerability:accessible"]["correct"], d["answerability:accessible"]["excluded"]) == (0, 1)
    assert d["answerability:inaccessible"]["correct"] == 1 and d["info_accessibility:accessible"]["correct"] == 1


def test_split_shards_and_combine_scores(run_dir):
    dirs = split(run_dir, 2)
    samples = [json.loads((d / "sample.json").read_text(encoding="utf-8")) for d in dirs]
    assert [(s["shard"], s["conversations"], s["sets"]) for s in samples] == [("1/2", ["c0"], ["0-0-0"]),
                                                                              ("2/2", ["c1"], ["1-0-0"])]
    a = {"direct": {"x": {"n": 1, "correct": 1, "excluded": 0, "included": 0, "tp": 2, "fp": 0, "fn": 0}},
         PAIRED: {"x": {"n": 1, "a_only": 1, "b_only": 0, "p": 1.0}}}
    b = {"direct": {"x": {"n": 1, "correct": 0, "excluded": 0, "included": 1, "tp": 2, "fp": 1, "fn": 0}},
         PAIRED: {"x": {"n": 1, "a_only": 1, "b_only": 0, "p": 1.0}}}
    total = combine([a, b])
    assert (total["direct"]["x"]["accuracy"], total["direct"]["x"]["precision"]) == (0.5, 0.8)
    assert total[PAIRED]["x"] == {"n": 2, "a_only": 2, "b_only": 0, "p": 0.5}


def test_mcnemar_is_exact_and_two_sided():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 0) == 0.0625
    assert mcnemar(3, 3) == 1.0


def test_extract_stops_at_the_budget_and_answer_skips_unfinished(run_dir, tmp_path):
    cassette = tmp_path / "cassette.jsonl"
    summary = run_extract(run_dir, "claude-opus-5-5", max_usd=1.0, max_calls=1, sdk=FakeSDK([msg(EXTRACT[0])]),
                          cassette=cassette)
    assert summary["stopped"].startswith("c0: 호출 상한")
    answers = run_answer(run_dir, "claude-sonnet-5-5", max_usd=1.0, sdk=FakeSDK([]), cassette=cassette)["answers"]
    assert [a.get("skipped") for a in answers] == ["추출 미완", "추출 안 함"]
