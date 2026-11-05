"""사람 검증 화면(이 컴퓨터 안에서만 열림). 확인표 CSV 세 개를 한 행씩 보여 주고, 답을 누를 때마다 같은 CSV에 바로 쓴다.

실행(worktree 루트):
  python3 -I docs/ontology-extract/human_check_ui.py --dir eval/human-check --who A --port 8765
그다음 브라우저에서 http://127.0.0.1:8765 를 연다.

- 쓰는 칸: <dir>/{gold,match,early}-check-<who>.csv의 답 칸(human_ok·human_leak)과 human_note만.
  누를 때마다 <dir>/human-check-<who>-log.jsonl에 한 줄씩 남긴다.
- 나머지 칸은 바이트까지 그대로 둔다. 시작할 때 다시 쓴 결과가 원본과 바이트까지 같은지 보고, 다르면 쓰지 않고 멈춘다
  (엑셀 등으로 저장해 형식이 바뀐 표에는 쓰지 않는다).
- gold-check 장면은 <dir>/gold-check-scenes.txt(human_check.py scenes), 화 전체는 data/synthetic/<작품>/clean/chNN.txt에서 읽는다.
- 판단 기준 문구는 eval/human-check/README.md 4절과 검증자 안내(human-check-C.zip의 00_GUIDE.txt 2절)를 옮겼다.

서버 없이 쓰는 HTML 한 파일(파이썬이 없는 검증자용):
  python3 -I docs/ontology-extract/human_check_ui.py --standalone eval/human-check/gold-check-C.html --who C --sheets gold
- 빈 원본 표, 장면 모음, 필요한 화 본문만 넣는다. 다른 검증자의 답은 넣지 않는다.
- 답은 그 브라우저(localStorage)에 저장하고, 'CSV 내려받기'로 <표>-<who>.csv를 받는다. 이 CSV는 human_check.py가 쓰는 꼴과 바이트까지 같다.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SHEETS = {"gold": "gold-check", "match": "match-check", "early": "early-check"}
ANSWERS = {"", "예", "아니오"}


def parse(raw: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def serialize(rows: list[list[str]]) -> bytes:
    buf = io.StringIO(newline="")
    csv.writer(buf).writerows(rows)  # excel 방언: 필요할 때만 따옴표, 줄 끝 \r\n. human_check.py가 쓴 방식과 같다
    return buf.getvalue().encode("utf-8-sig")


def load_scenes(path: Path) -> dict[int, list]:
    out = {}
    for block in re.split(r"^={60}\n", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
        n = int(re.match(r"\[행 (\d+)\]", block).group(1))
        lines = []
        for line in block.split("-" * 60 + "\n", 1)[1].splitlines():
            m = re.match(r"^([▶ ])\s*(\d+)  (.*)$", line)
            if m:
                lines.append([m.group(1) == "▶", int(m.group(2)), m.group(3)])
            elif line.strip() == "…":
                lines.append(None)
        out[n] = lines
    return out


class Store:
    def __init__(self, folder: Path, who: str, texts: Path):
        self.lock, self.texts = threading.Lock(), texts
        self.log = folder / f"human-check-{who}-log.jsonl"
        self.sheets = {}
        for key, stem in SHEETS.items():
            path = folder / f"{stem}-{who}.csv"
            raw = path.read_bytes()
            rows = parse(raw)
            if serialize(rows) != raw:
                sys.exit(f"{path}: 다시 쓰면 바이트가 달라짐(엑셀 등으로 저장해 형식이 바뀐 듯). 쓰지 않고 멈춤.")
            header = rows[0]
            ans = "human_leak" if "human_leak" in header else "human_ok"
            self.sheets[key] = {"path": path, "rows": rows, "ans": header.index(ans), "note": header.index("human_note")}
        self.scenes = load_scenes(folder / "gold-check-scenes.txt")
        if sorted(self.scenes) != list(range(2, len(self.sheets["gold"]["rows"]) + 1)):
            sys.exit("장면 모음의 행 번호가 gold-check와 맞지 않음")
        self.works = {r[1] for s in self.sheets.values() for r in s["rows"][1:]}

    def state(self) -> dict:
        return {"sheets": {k: {"file": s["path"].name, "header": s["rows"][0], "rows": s["rows"][1:],
                               "ans": s["ans"], "note": s["note"]} for k, s in self.sheets.items()},
                "scenes": self.scenes}

    def chapter(self, work: str, ch: int) -> list:
        if work not in self.works or not 1 <= ch <= 99:
            raise ValueError("작품이나 화가 범위 밖")
        text = (self.texts / work / "clean" / f"ch{ch:02d}.txt").read_text(encoding="utf-8")
        return [[n, line] for n, line in enumerate(text.splitlines(), 1) if line.strip()]

    def set(self, sheet: str, row: int, field: str, value: str) -> str:
        s = self.sheets[sheet]
        if not 2 <= row <= len(s["rows"]):
            raise ValueError("행 번호가 범위 밖")
        if field == "answer":
            if value not in ANSWERS:
                raise ValueError("예·아니오·빈칸만 씀")
            col = s["ans"]
        elif field == "note":
            value, col = " ".join(value.split()), s["note"]  # 줄바꿈은 한 칸으로
        else:
            raise ValueError("field는 answer나 note")
        with self.lock:
            s["rows"][row - 1][col] = value
            tmp = s["path"].with_name("." + s["path"].name + ".tmp")
            tmp.write_bytes(serialize(s["rows"]))
            os.replace(tmp, s["path"])
            with self.log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "sheet": sheet, "row": row,
                                     "field": field, "value": value}, ensure_ascii=False) + "\n")
        return time.strftime("%H:%M:%S")


def make_handler(store: Store, page: bytes, port: int):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code: int, obj) -> None:
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if self.headers.get("Host") not in hosts:
                return self.send(403, b"forbidden", "text/plain")
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self.send(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self.send_json(200, store.state())
            elif url.path == "/api/chapter":
                q = parse_qs(url.query)
                try:
                    self.send_json(200, {"lines": store.chapter(q["work"][0], int(q["ch"][0]))})
                except (KeyError, ValueError, OSError) as e:
                    self.send_json(400, {"error": str(e)})
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.headers.get("Host") not in hosts or self.headers.get("X-Check") != "1" or self.path != "/api/set":
                return self.send(403, b"forbidden", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                at = store.set(req["sheet"], int(req["row"]), req["field"], str(req["value"]))
                self.send_json(200, {"ok": True, "at": at})
            except (KeyError, ValueError, TypeError) as e:
                self.send_json(400, {"ok": False, "error": str(e)})

        def log_message(self, fmt, *args):
            pass

    return Handler


PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>사람 검증 __WHO__</title>
<style>
:root{--bg:#f6f6f3;--card:#fff;--text:#1c1c1e;--muted:#6e6e73;--line:#e2e2de;--accent:#2f5fd0;--accent-bg:#e8eefc;
  --yes:#16794a;--yes-bg:#e3f3ea;--no:#b3261e;--no-bg:#fbe9e7;--hint:#fff1b8;--chip:#efefeb;--empty:#deded9}
@media (prefers-color-scheme: dark){:root{--bg:#18181a;--card:#232326;--text:#ececee;--muted:#a1a1a8;--line:#38383d;
  --accent:#8fb0ff;--accent-bg:#26304a;--yes:#5fd39a;--yes-bg:#1d3a2b;--no:#ff8a80;--no-bg:#432522;--hint:#4b4118;--chip:#2f2f33;--empty:#45454b}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text)}
body{font:17px/1.75 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Noto Sans KR","Malgun Gothic",sans-serif;padding-bottom:170px}
header{position:sticky;top:0;z-index:10;background:var(--card);border-bottom:1px solid var(--line)}
.top{max-width:920px;margin:0 auto;padding:10px 16px 8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.tab{border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:999px;padding:5px 14px;font:inherit;font-size:15px;cursor:pointer}
.tab.on{background:var(--accent-bg);border-color:var(--accent);color:var(--accent);font-weight:600}
.tab .n{color:var(--muted);font-weight:400;margin-left:6px;font-variant-numeric:tabular-nums}
.spacer{flex:1}
.ghost{border:1px solid var(--line);background:transparent;color:var(--text);border-radius:8px;padding:5px 10px;font:inherit;font-size:14px;cursor:pointer}
.bar{height:4px;background:var(--line)}.bar>div{height:100%;background:var(--accent);width:0;transition:width .2s}
.grid{display:none;max-width:920px;margin:0 auto;padding:4px 16px 10px;gap:4px;flex-wrap:wrap}
.grid.show{display:flex}
.cell{width:40px;height:28px;border-radius:6px;border:none;font-size:12px;cursor:pointer;background:var(--empty);color:var(--text);font-variant-numeric:tabular-nums}
.cell.yes{background:var(--yes-bg);color:var(--yes);font-weight:700}.cell.no{background:var(--no-bg);color:var(--no);font-weight:700}
.cell.cur{outline:2px solid var(--accent);outline-offset:1px}
main{max-width:920px;margin:0 auto;padding:16px}
.rules{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 16px;margin-bottom:14px;font-size:15px;color:var(--muted)}
.rules summary{cursor:pointer;color:var(--text);font-weight:600}
.rules ul{margin:6px 0 4px;padding-left:20px}.rules li{margin:2px 0}
.keys{font-size:13px;margin:4px 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px}
.meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:14px;color:var(--muted);margin-bottom:6px}
.rowno{font-weight:700;color:var(--text);font-size:16px;margin-right:4px}
.chip{background:var(--chip);border-radius:6px;padding:1px 8px}.chip.kind{background:var(--accent-bg);color:var(--accent);font-weight:600}
.label{font-size:13px;color:var(--muted);margin:16px 0 2px;font-weight:600}
.fact{font-size:20px;line-height:1.65;font-weight:600}.fact.second{font-weight:500}
.small{font-size:14px;color:var(--muted)}
.lines{margin-top:4px}
.ln{display:grid;grid-template-columns:46px 1fr;gap:10px;padding:3px 6px;border-radius:6px}
.ln .num{color:var(--muted);font-size:13px;text-align:right;padding-top:3px;font-variant-numeric:tabular-nums}
.ln.hint{background:var(--hint)}
.gap{color:var(--muted);text-align:center;font-size:14px}
.box{border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin-top:4px}
.item{padding:5px 0;border-top:1px dashed var(--line)}.item:first-child{border-top:none}
.full{margin-top:14px}.full summary{cursor:pointer;color:var(--accent);font-size:15px}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);z-index:10}
.foot{max-width:920px;margin:0 auto;padding:10px 16px 8px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ans{border:2px solid var(--line);background:var(--card);color:var(--text);border-radius:10px;padding:7px 14px;font:inherit;font-weight:700;cursor:pointer}
.ans small{font-weight:400;color:var(--muted);margin-left:5px;font-size:13px}
.ans.yes.on{background:var(--yes-bg);border-color:var(--yes);color:var(--yes)}.ans.no.on{background:var(--no-bg);border-color:var(--no);color:var(--no)}
#note{flex:1;min-width:200px;font:inherit;font-size:15px;padding:8px 12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--text)}
.nav{border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:10px;padding:7px 12px;font:inherit;cursor:pointer}
.status{font-size:13px;color:var(--muted);width:100%;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.status .err{color:var(--no);font-weight:700}
.link{background:none;border:none;color:var(--muted);text-decoration:underline;cursor:pointer;font:inherit;font-size:13px;padding:0}
kbd{font:12px/1.4 ui-monospace,Menlo,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;color:var(--muted)}
@media (max-width:640px){body{font-size:16px}.fact{font-size:18px}.ln{grid-template-columns:36px 1fr}.card{padding:14px}}
</style>
</head>
<body>
<header>
  <div class="top"><span id="tabs"></span><span class="spacer"></span><span id="fileBtns"></span>
    <button class="ghost" id="gridBtn">행 목록</button><button class="ghost" id="emptyBtn">다음 빈 행 <kbd>E</kbd></button></div>
  <div class="grid" id="grid"></div>
  <div class="bar"><div id="prog"></div></div>
</header>
<main>
  <div id="intro"></div>
  <details class="rules" id="rules" open><summary>판단 기준</summary><ul id="rulesList"></ul>
    <div class="keys"><kbd>1</kbd> 예 · <kbd>2</kbd> 아니오(메모 칸으로 이동) · <kbd>←</kbd><kbd>→</kbd> 이전·다음 행 · <kbd>E</kbd> 다음 빈 행 · <kbd>M</kbd> 메모 · 메모에서 <kbd>Enter</kbd> 저장하고 다음 행. 답을 고른 뒤 <kbd>→</kbd>로 다음 행</div></details>
  <div class="card" id="card">불러오는 중…</div>
</main>
<footer><div class="foot">
  <button class="nav" id="prev" title="이전 행">◀</button>
  <button class="ans yes" id="yes"></button><button class="ans no" id="no"></button>
  <input id="note" placeholder="메모 (아니오면 이유) · Enter 저장하고 다음 행" autocomplete="off">
  <button class="nav" id="next" title="다음 행">▶</button>
  <div class="status"><span id="pos"></span><span><span id="saved"></span> · <button class="link" id="clear">이 행 답 지우기</button></span></div>
</div></footer>
<!--DATA-->
<script>
const DATA_EL = document.getElementById("data");
const MODE = DATA_EL ? "standalone" : "server";  // standalone: 서버 없이 연 HTML 한 파일. 답은 브라우저에 두고 CSV로 내려받는다
const KEY = "hc-answers-__WHO__";
let ORDER = ["gold", "match", "early"], sinceExport = 0, storageOK = true;
const NAMES = {gold: "정답 확인", match: "짝 확인", early: "앞당김 확인"};
const RULES = {
  gold: ["반전 공개 시점이면 그 화 본문에서 그 사실이 처음 분명해지는지 봄.",
         "인지 시점이면 그 인물이 그 화에서 그 사실을 알게 되는지 봄. 직접 겪거나, 그 자리에 있었거나, 전해 들은 경우가 해당함.",
         "▶ 줄(힌트 줄)은 키워드나 인물 이름으로 찾은 근거 후보 줄임. 후보일 뿐이므로 필요하면 본문을 직접 확인함.",
         "힌트 줄만 보고 판단하지 않음. 대사 줄에는 말한 사람이 적혀 있지 않은 경우가 많으므로, 앞뒤 줄의 서술로 말한 사람과 그 자리에 있는 인물을 확인함.",
         "장면만으로 부족하면 아래 '그 화 전체 보기'를 엶. 본문에 그 장면이 아예 없으면(설정 메모에만 있는 경우 등) 아니오로 쓰고 메모에 이유를 적음.",
         "맞으면 예, 다르면 아니오로 쓰고 메모에 맞는 화나 이유를 적음."],
  match: ["판정이 '같음'이면 짝지은 추출 사실이 정답 사실과 같은 내용인지 봄. 같으면 예.",
          "'일부만 담음(partial)' 표시가 있으면 정답의 일부만 담았다는 판정임. 일부만 담은 것이 맞으면 예.",
          "판정이 '없음'이면 후보 중 정답과 같은 내용이 정말 없는지 봄. 없으면 예. (이번 표본은 모두 '같음')"],
  early: ["질문: 추출 화에 나온 추출 사실이, 정답 화에서야 공개되는 정보를 미리 드러내는가.",
          "근거 줄은 추출 사실의 근거가 된 본문 줄임.",
          "미리 드러내면 예, 그 화 본문에 이미 있는 내용만 담았으면 아니오."],
};
const COMMON = ["예 또는 아니오만 고름. 애매하면 아니오로 두고 메모에 이유를 적음.",
                "LLM에 묻지 않고 본문만 보고 판단함. 다른 검증자의 답은 보지 않음."];
let S = null, sheet = "gold", idx = 0, noteDirty = null, noteTimer = null, pending = 0;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const sh = () => S.sheets[sheet];
const cur = () => sh().rows[idx];
const get = (r, name) => r[sh().header.indexOf(name)];

function chips(...xs) { return xs.filter(Boolean).join(""); }
function chip(t, cls = "") { return `<span class="chip ${cls}">${esc(t)}</span>`; }
function lineRows(list, marks = new Set()) {
  return list.map(l => l === null ? `<div class="gap">⋯</div>` :
    `<div class="ln ${l[0] || marks.has(l[1]) ? "hint" : ""}"><span class="num">${l[1]}</span><span>${esc(l[2])}</span></div>`).join("");
}
function fullChapter(work, ch, marks) {
  return `<details class="full" data-work="${esc(work)}" data-ch="${esc(ch)}" data-marks="${[...marks].join(",")}"><summary>${esc(ch)}화 전체 보기</summary><div class="lines">불러오는 중…</div></details>`;
}
function items(s) {
  const out = [], re = /(c\d+-\d+)\[(\d+)장\] (.*?)(?= \/ c\d+-\d+\[\d+장\] |$)/g; let m;
  while ((m = re.exec(s))) out.push(m);
  return out;
}
function itemBox(s) {
  const it = items(s);
  return it.length ? it.map(m => `<div class="item">${chip(m[2] + "장")} ${esc(m[3])} <span class="small">${esc(m[1])}</span></div>`).join("") : esc(s);
}
function goldCard(r, n) {
  const ch = get(r, "chapter"), sc = S.scenes[n] || [], marks = new Set(sc.filter(l => l && l[0]).map(l => l[1]));
  return `<div class="meta"><span class="rowno">${n}행</span>${chips(chip(get(r, "kind"), "kind"), chip(get(r, "work")), chip(ch + "화"),
      get(r, "character") && chip(get(r, "character")))}<span class="small">${esc(get(r, "fact_id"))}</span></div>
    <div class="label">사실</div><div class="fact">${esc(get(r, "statement"))}</div>
    <div class="label">질문</div><div>${esc(get(r, "question"))}</div>
    ${get(r, "hint_note") ? `<div class="label">설정 메모</div><div class="small">${esc(get(r, "hint_note"))}</div>` : ""}
    <div class="label">${esc(ch)}화 본문 · ▶ 힌트 줄 앞 20줄, 뒤 10줄 (노란 줄이 힌트 줄)</div><div class="lines">${lineRows(sc)}</div>
    ${fullChapter(get(r, "work"), ch, marks)}`;
}
function matchCard(r, n) {
  const partial = /^true$/i.test(get(r, "partial")), k = items(get(r, "matched")).length;
  return `<div class="meta"><span class="rowno">${n}행</span>${chips(chip(get(r, "work")), chip("정답 " + get(r, "gold_ch") + "화"),
      chip("판정: " + get(r, "judged"), "kind"), partial && chip("일부만 담음(partial)", "kind"))}<span class="small">${esc(get(r, "gold_id"))}</span></div>
    <div class="label">정답 사실</div><div class="fact">${esc(get(r, "gold_statement"))}</div>
    <div class="label">짝지은 추출 사실${k > 1 ? ` (${k}개)` : ""}</div><div class="box">${itemBox(get(r, "matched"))}</div>
    <div class="label">LLM 판정 이유</div><div class="small">${esc(get(r, "reason"))}</div>
    <details class="full"><summary>후보 목록 (판정이 '없음'일 때만 봄)</summary><div class="box">${itemBox(get(r, "candidates"))}</div></details>
    ${fullChapter(get(r, "work"), get(r, "gold_ch"), new Set())}`;
}
function earlyCard(r, n) {
  const ec = get(r, "extracted_ch"), gc = get(r, "gold_ch"), cited = [], re = /(\d+): (.*?)(?= \/ \d+: |$)/g; let m;
  while ((m = re.exec(get(r, "cited_lines")))) cited.push([true, +m[1], m[2]]);
  return `<div class="meta"><span class="rowno">${n}행</span>${chips(chip(get(r, "work")), chip("정답 " + gc + "화 공개", "kind"),
      chip("추출 " + ec + "화"))}<span class="small">${esc(get(r, "gold_id"))} · ${esc(get(r, "extracted_id"))}</span></div>
    <div class="label">정답 사실 (${esc(gc)}화에서야 공개)</div><div class="fact">${esc(get(r, "gold_statement"))}</div>
    <div class="label">추출 사실 (${esc(ec)}화에서 뽑음)</div><div class="fact second">${esc(get(r, "extracted_statement"))}</div>
    <div class="label">추출 사실의 근거 줄 (${esc(ec)}화)</div><div class="lines">${cited.length ? lineRows(cited) : esc(get(r, "cited_lines"))}</div>
    <div class="label">질문</div><div>${esc(ec)}화의 추출 사실이 ${esc(gc)}화에서야 공개되는 정보를 미리 드러내는가</div>
    ${fullChapter(get(r, "work"), ec, new Set(cited.map(c => c[1])))}`;
}
function render(keepNote = false) {
  const s = sh(), r = cur(), n = idx + 2, a = r[s.ans];
  $("tabs").innerHTML = ORDER.map(k => {
    const t = S.sheets[k], done = t.rows.filter(x => x[t.ans]).length;
    return `<button class="tab ${k === sheet ? "on" : ""}" data-k="${k}">${NAMES[k]}<span class="n">${done}/${t.rows.length}</span></button>`;
  }).join(" ");
  $("prog").style.width = (100 * s.rows.filter(x => x[s.ans]).length / s.rows.length) + "%";
  $("rulesList").innerHTML = RULES[sheet].concat(COMMON).map(t => `<li>${esc(t)}</li>`).join("");
  $("card").innerHTML = sheet === "gold" ? goldCard(r, n) : sheet === "match" ? matchCard(r, n) : earlyCard(r, n);
  $("yes").innerHTML = `예<small>${sheet === "early" ? "미리 드러냄" : "맞음"}</small> <kbd>1</kbd>`;
  $("no").innerHTML = `아니오<small>${sheet === "early" ? "그 화 내용만" : "다름"}</small> <kbd>2</kbd>`;
  $("yes").classList.toggle("on", a === "예");
  $("no").classList.toggle("on", a === "아니오");
  if (!keepNote) $("note").value = r[s.note];
  $("pos").textContent = `${NAMES[sheet]} ${idx + 1} / ${s.rows.length} · 시트 ${n}행 · ${s.file}`;
  renderGrid();
  try { localStorage.setItem("hc-pos-__WHO__", JSON.stringify({sheet, idx})); } catch (e) {}
}
function renderGrid() {
  const g = $("grid");
  if (!g.classList.contains("show")) return;
  const s = sh();
  g.innerHTML = s.rows.map((x, i) => `<button class="cell ${x[s.ans] === "예" ? "yes" : x[s.ans] === "아니오" ? "no" : ""} ${i === idx ? "cur" : ""}" data-i="${i}">${i + 2}</button>`).join("");
}
function showSaved(t, err = false) { $("saved").innerHTML = err ? `<span class="err">${esc(t)}</span>` : esc(t); }
async function saveAt(k, i, field, value) {
  const s = S.sheets[k];
  s.rows[i][field === "answer" ? s.ans : s.note] = value;
  if (MODE === "standalone") {
    sinceExport++;
    const ok = saveLocal();
    showSaved(ok ? `브라우저에 저장됨 ${new Date().toTimeString().slice(0, 8)}` : "브라우저에 저장하지 못함. 'CSV 내려받기'를 자주 눌러 주세요", !ok);
    renderFileBtns();
    return;
  }
  pending++; showSaved("저장 중…");
  try {
    const res = await fetch("/api/set", {method: "POST", headers: {"Content-Type": "application/json", "X-Check": "1"},
      body: JSON.stringify({sheet: k, row: i + 2, field, value})});
    const j = await res.json();
    if (!res.ok || !j.ok) throw new Error(j.error || res.status);
    showSaved(`저장됨 ${j.at}`);
  } catch (e) {
    showSaved(`저장 실패(${e.message}). 서버가 꺼졌으면 다시 켠 뒤 이 행을 다시 누르세요`, true);
  } finally { pending--; }
}
function flushNote() {
  if (!noteDirty) return;
  const {k, i, v} = noteDirty;
  noteDirty = null; clearTimeout(noteTimer);
  saveAt(k, i, "note", v);
}
function answer(v) {
  flushNote();
  saveAt(sheet, idx, "answer", v);
  render(true);
  if (v === "아니오") $("note").focus();  // 예를 눌러도 다음 행으로 넘기지 않는다(11/05: 빠르게 누르기만 해서 다시 봄)
}
function go(d) { flushNote(); idx = Math.max(0, Math.min(sh().rows.length - 1, idx + d)); render(); window.scrollTo(0, 0); }
function jump(k, i) { flushNote(); sheet = k; idx = i; render(); window.scrollTo(0, 0); }
function nextEmpty() {
  const s = sh(), N = s.rows.length;
  for (let d = 1; d <= N; d++) { const j = (idx + d) % N; if (!s.rows[j][s.ans]) return jump(sheet, j); }
  showSaved("이 표는 다 채웠음");
}
document.addEventListener("click", e => {
  const t = e.target.closest("button"); if (!t) return;
  if (t.classList.contains("tab")) { const k = t.dataset.k, s = S.sheets[k], j = s.rows.findIndex(x => !x[s.ans]); jump(k, j < 0 ? 0 : j); }
  else if (t.classList.contains("cell")) jump(sheet, +t.dataset.i);
  else if (t.id === "yes") answer("예");
  else if (t.id === "no") answer("아니오");
  else if (t.id === "prev") go(-1);
  else if (t.id === "next") go(1);
  else if (t.id === "emptyBtn") nextEmpty();
  else if (t.id === "gridBtn") { $("grid").classList.toggle("show"); renderGrid(); }
  else if (t.id === "clear") { saveAt(sheet, idx, "answer", ""); render(true); }
  else if (t.id === "exportBtn") {
    flushNote();
    ORDER.forEach(download);
    sinceExport = 0; renderFileBtns();
    showSaved(`내려받음: ${ORDER.map(k => S.sheets[k].file).join(", ")}`);
  }
  else if (t.id === "importBtn") $("importFile").click();
});
document.addEventListener("change", async e => {
  if (e.target.id !== "importFile" || !e.target.files.length) return;
  try {
    const [k, n] = importCSV(await e.target.files[0].text());
    sheet = k; idx = 0; render();
    showSaved(`불러옴: ${S.sheets[k].file}에서 답 ${n}행`);
  } catch (err) { showSaved(`불러오지 못함: ${err.message}`, true); }
  e.target.value = "";
});
document.addEventListener("toggle", async e => {
  const d = e.target;
  if (!d.classList || !d.classList.contains("full") || !d.open || !d.dataset.work || d.dataset.loaded) return;
  d.dataset.loaded = "1";
  const marks = new Set((d.dataset.marks || "").split(",").filter(Boolean).map(Number));
  try {
    let lines;
    if (MODE === "standalone") {
      lines = (S.chapters || {})[`${d.dataset.work}/${d.dataset.ch}`];
      if (!lines) throw new Error("이 파일에 없는 화");
    } else {
      const res = await fetch(`/api/chapter?work=${encodeURIComponent(d.dataset.work)}&ch=${encodeURIComponent(d.dataset.ch)}`);
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || res.status);
      lines = j.lines;
    }
    d.querySelector(".lines").innerHTML = lineRows(lines.map(([num, text]) => [false, num, text]), marks);
  } catch (err) { d.querySelector(".lines").textContent = `불러오지 못함: ${err.message}`; d.dataset.loaded = ""; }
}, true);
$("note").addEventListener("input", () => {
  noteDirty = {k: sheet, i: idx, v: $("note").value};
  clearTimeout(noteTimer); noteTimer = setTimeout(flushNote, 700);
});
$("note").addEventListener("blur", flushNote);
document.addEventListener("keydown", e => {
  if (e.target === $("note")) {
    if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); flushNote(); $("note").blur(); go(1); }
    else if (e.key === "Escape") $("note").blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey || e.repeat || !S) return;  // 키를 누르고 있어도 한 번만
  const c = e.code;
  if (c === "Digit1" || c === "Numpad1") { e.preventDefault(); answer("예"); }
  else if (c === "Digit2" || c === "Numpad2") { e.preventDefault(); answer("아니오"); }
  else if (c === "ArrowRight") { e.preventDefault(); go(1); }
  else if (c === "ArrowLeft") { e.preventDefault(); go(-1); }
  else if (c === "KeyE") { e.preventDefault(); nextEmpty(); }
  else if (c === "KeyM") { e.preventDefault(); $("note").focus(); }
});
window.addEventListener("beforeunload", e => {
  flushNote();
  if (pending > 0 || (MODE === "standalone" && !storageOK && sinceExport > 0)) { e.preventDefault(); e.returnValue = ""; }
});
function saveLocal() {
  const out = {};
  for (const k of ORDER) {
    const s = S.sheets[k], m = {};
    s.rows.forEach((r, i) => { if (r[s.ans] || r[s.note]) m[i + 2] = [r[s.ans], r[s.note]]; });
    out[k] = m;
  }
  try { localStorage.setItem(KEY, JSON.stringify(out)); storageOK = true; } catch (e) { storageOK = false; }
  return storageOK;
}
function loadLocal() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(KEY) || "null"); localStorage.setItem(KEY + "-probe", "1"); localStorage.removeItem(KEY + "-probe"); }
  catch (e) { storageOK = false; }
  let n = 0;
  for (const k of ORDER) for (const [row, v] of Object.entries((saved || {})[k] || {})) {
    const s = S.sheets[k], r = s.rows[+row - 2];
    if (r && Array.isArray(v)) { r[s.ans] = v[0] || ""; r[s.note] = v[1] || ""; n++; }
  }
  return n;
}
const csvField = v => { const s = String(v ?? ""); return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
function toCSV(k) {  // human_check.py와 같은 꼴: BOM, 필요할 때만 따옴표, 줄 끝 \r\n
  const s = S.sheets[k];
  return "﻿" + [s.header, ...s.rows].map(r => r.map(csvField).join(",")).join("\r\n") + "\r\n";
}
function parseCSV(text) {
  text = text.replace(/^﻿/, "");
  const rows = []; let row = [], f = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(f); f = ""; }
    else if (c === "\r" || c === "\n") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(f); rows.push(row); row = []; f = ""; }
    else f += c;
  }
  if (f !== "" || row.length) { row.push(f); rows.push(row); }
  return rows;
}
function download(k) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([toCSV(k)], {type: "text/csv;charset=utf-8"}));
  a.download = S.sheets[k].file;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
}
function importCSV(text) {  // 엑셀·구글 시트에서 채우던 CSV를 이어받는다. 답 칸 밖이 원본과 다르면 받지 않는다
  const rows = parseCSV(text), same = (x, y) => x === y || (/^(true|false)$/i.test(x) && x.toLowerCase() === y.toLowerCase());
  const k = ORDER.find(k => S.sheets[k].header.join("\u0001") === (rows[0] || []).join("\u0001"));
  if (!k) throw new Error("머리줄이 이 확인표와 다름(CSV UTF-8로 저장했는지 확인)");
  const s = S.sheets[k];
  if (rows.length - 1 !== s.rows.length) throw new Error(`행 수가 다름(${rows.length - 1}/${s.rows.length})`);
  s.rows.forEach((r, i) => {
    r.forEach((v, j) => { if (j !== s.ans && j !== s.note && !same(rows[i + 1][j] ?? "", v)) throw new Error(`${i + 2}행 '${s.header[j]}' 칸이 원본과 다름`); });
    const a = (rows[i + 1][s.ans] || "").trim();
    if (a && a !== "예" && a !== "아니오") throw new Error(`${i + 2}행 답이 '${a}'`);
  });
  let n = 0;
  s.rows.forEach((r, i) => { r[s.ans] = (rows[i + 1][s.ans] || "").trim(); r[s.note] = rows[i + 1][s.note] || ""; if (r[s.ans]) n++; });
  saveLocal();
  return [k, n];
}
function renderFileBtns() {
  if (MODE !== "standalone") return;
  $("fileBtns").innerHTML = `<button class="ghost" id="exportBtn">CSV 내려받기${sinceExport ? ` (새 답 ${sinceExport})` : ""}</button> <button class="ghost" id="importBtn">CSV 불러오기</button><input type="file" id="importFile" accept=".csv,text/csv" hidden>`;
}
(async function load() {
  if (MODE === "standalone") {
    S = JSON.parse(DATA_EL.textContent);
    ORDER = ORDER.filter(k => S.sheets[k]);
    const n = loadLocal();
    renderFileBtns();
    const files = ORDER.map(k => S.sheets[k].file).join(", ");
    $("intro").innerHTML = `<div class="rules" style="color:var(--text)"><b>이 화면 쓰는 법</b><ul>
      <li>답은 이 브라우저에 바로 저장됩니다. 창을 닫았다 다시 열어도 같은 브라우저면 이어서 할 수 있습니다${n ? ` (지금 이어받은 답 ${n}행)` : ""}.</li>
      <li>다 끝나면 위쪽 <b>CSV 내려받기</b>를 눌러 받은 <b>${esc(files)}</b>를 보내 주세요. 중간중간 내려받아 두면 더 안전합니다.</li>
      <li>시크릿(개인 정보 보호) 창에서는 저장되지 않습니다. 엑셀에서 채우던 CSV가 있으면 <b>CSV 불러오기</b>로 이어서 합니다.</li></ul>
      ${storageOK ? "" : `<div class="err" style="color:var(--no);font-weight:700">이 브라우저에서는 자동 저장이 되지 않습니다. CSV 내려받기를 자주 눌러 주세요.</div>`}</div>`;
  } else {
    try { S = await (await fetch("/api/state")).json(); }
    catch (e) { $("card").textContent = "서버에 연결하지 못함. 서버를 켠 뒤 새로고침하세요."; return; }
  }
  ORDER = ORDER.filter(k => S.sheets[k]);
  let pos = null;
  try { pos = JSON.parse(localStorage.getItem("hc-pos-__WHO__")); } catch (e) {}
  if (pos && S.sheets[pos.sheet] && pos.idx >= 0 && pos.idx < S.sheets[pos.sheet].rows.length) { sheet = pos.sheet; idx = pos.idx; }
  else { const s = S.sheets[ORDER[0]], j = s.rows.findIndex(x => !x[s.ans]); sheet = ORDER[0]; idx = j < 0 ? 0 : j; }
  render();
})();
</script>
</body>
</html>
"""


def standalone(folder: Path, who: str, keys: list[str], texts: Path, out: Path) -> None:
    """서버 없이 더블클릭으로 여는 HTML 한 파일. 빈 원본 표, 장면, 필요한 화 본문을 넣는다(다른 검증자의 답은 넣지 않는다)."""
    data, need = {"sheets": {}, "scenes": {}, "chapters": {}}, set()
    for key in keys:
        stem = SHEETS[key]
        rows = parse((folder / f"{stem}.csv").read_bytes())
        header = rows[0]
        ans, note = header.index("human_leak" if "human_leak" in header else "human_ok"), header.index("human_note")
        if any(r[ans] or r[note] for r in rows[1:]):
            sys.exit(f"{stem}.csv에 답이 들어 있음. 빈 원본으로만 만든다.")
        data["sheets"][key] = {"file": f"{stem}-{who}.csv", "header": header, "rows": rows[1:], "ans": ans, "note": note}
        col = header.index({"gold": "chapter", "match": "gold_ch", "early": "extracted_ch"}[key])
        need |= {(r[header.index("work")], int(r[col])) for r in rows[1:]}
    if "gold" in keys:
        data["scenes"] = load_scenes(folder / "gold-check-scenes.txt")
    for work, ch in sorted(need):
        text = (texts / work / "clean" / f"ch{ch:02d}.txt").read_text(encoding="utf-8")
        data["chapters"][f"{work}/{ch}"] = [[n, line] for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(PAGE.replace("__WHO__", who).replace("<!--DATA-->", f'<script id="data" type="application/json">{blob}</script>'),
                   encoding="utf-8")
    print(f"{out}: 표 {', '.join(keys)}, 본문 {len(need)}화, {out.stat().st_size / 1e6:.1f}MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="eval/human-check")
    ap.add_argument("--who", default="A")
    ap.add_argument("--texts", default="data/synthetic")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--standalone", help="서버 없이 여는 HTML 한 파일을 이 경로에 만든다(답은 브라우저에 두고 CSV로 내려받음)")
    ap.add_argument("--sheets", default="gold,match,early", help="--standalone에 넣을 표(쉼표로 구분)")
    a = ap.parse_args()
    if a.standalone:
        return standalone(Path(a.dir), a.who, a.sheets.split(","), Path(a.texts), Path(a.standalone))
    store = Store(Path(a.dir), a.who, Path(a.texts))
    page = PAGE.replace("__WHO__", a.who).replace("<!--DATA-->", "").encode("utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store, page, a.port))
    print(f"http://127.0.0.1:{a.port} (검증자 {a.who}, {a.dir})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
