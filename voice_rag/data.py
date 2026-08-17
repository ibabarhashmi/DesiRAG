"""Load Hindi MS MARCO-XI rows from a local parquet and slice them into a
retrieval corpus (deduplicated passages) plus a held-out evaluation set.

The demo index is built on a slice of the validation file (a fixed 462 MB
snapshot); the same code path works against any train/validation parquet for a
full-scale build.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pyarrow import parquet as pq


@dataclass
class Passage:
    pid: str                    # stable id (first slot that contributed text)
    text: str                   # Hindi translated passage
    eng: str
    query_ids: list[int] = field(default_factory=list)  # rows where selected
    answers: list[str] = field(default_factory=list)
    query_types: list[str] = field(default_factory=list)


@dataclass
class EvalRow:
    qid: int
    query: str
    answer: str
    gold_pid: str | None        # pid of a passage marked is_selected==1
    gold_text: str
    query_type: str


def iter_rows(parquet: Path, limit: int | None = None):
    """Yield dict rows from a parquet file without holding it in memory."""
    pf = pq.ParquetFile(str(parquet))
    seen = 0
    for batch in pf.iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            yield r
            seen += 1
            if limit is not None and seen >= limit:
                return


def build_slices(settings, force: bool = False) -> tuple[Path, Path]:
    """Persist corpus passages + eval queries derived from the raw parquet.

    Corpus covers rows [0, corpus+eval); eval queries come from the trailing
    `n_eval_queries` rows, so their gold passages are inside the corpus
    (standard MS MARCO setup: the query retrieves against the fixed collection).
    Returns (corpus_jsonl, eval_jsonl).
    """
    settings.corpus_jsonl.parent.mkdir(parents=True, exist_ok=True)
    settings.eval_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not force and settings.corpus_jsonl.exists() and settings.eval_jsonl.exists():
        return settings.corpus_jsonl, settings.eval_jsonl

    total = settings.n_corpus_examples + settings.n_eval_queries
    slice_start = total - settings.n_eval_queries
    cap = getattr(settings, "max_corpus_passages", 80_000)

    rows = list(iter_rows(settings.raw_parquet, limit=total))
    passages: dict[str, Passage] = {}   # keyed by text (dedup)
    eval_rows: list[EvalRow] = []

    def slot_text(row, slot):
        trans = (row.get("passages") or {}).get("Translated_passages") or []
        return trans[slot] if slot < len(trans) else None

    def add_passage(row, slot, selected) -> Passage | None:
        text = slot_text(row, slot)
        if not text or not str(text).strip():
            return None
        key = " ".join(str(text).split())[:512]
        p = passages.get(key)
        if p is None:
            eng = (row.get("passages") or {}).get("English_passages") or []
            p = Passage(pid=f"{row['_i']}:{slot}", text=str(text).strip(),
                        eng=(eng[slot] if slot < len(eng) else "").strip())
            passages[key] = p
        if selected:
            qid = row.get("query_id")
            if qid is not None:
                p.query_ids.append(int(qid))
            p.query_types.append(row.get("query_type") or "")
            if row.get("Answer"):
                p.answers.append(str(row["Answer"]).strip())
        return p

    def selected_at(row, slot):
        sel = (row.get("passages") or {}).get("is_selected") or []
        return bool(slot < len(sel) and sel[slot] == 1)

    # 1) eval window rows: record queries AND guarantee their gold passages
    #    are in the corpus (MS MARCO: the query retrieves against the fixed
    #    collection, so the gold passage is a corpus member).
    for i, row in enumerate(rows):
        if i < slice_start:
            continue
        row["_i"] = i
        query = (row.get("query") or "").strip()
        ns = len((row.get("passages") or {}).get("Translated_passages") or [])
        gold_pid = gold_text = None
        for slot in range(ns):
            p = add_passage(row, slot, selected_at(row, slot))
            if p is not None and selected_at(row, slot) and gold_pid is None:
                gold_pid = p.pid
                gold_text = p.text
        if query and row.get("Answer"):
            eval_rows.append(EvalRow(
                qid=int(row.get("query_id") or i), query=query,
                answer=str(row["Answer"]).strip(), gold_pid=gold_pid,
                gold_text=gold_text or "",
                query_type=row.get("query_type") or ""))
            if len(eval_rows) >= settings.n_eval_queries:
                break

    # 2) corpus rows: fill with every passage until the cap.
    for i, row in enumerate(rows):
        if len(passages) >= cap:
            break
        row["_i"] = i
        for slot in range(len((row.get("passages") or {}).get(
                "Translated_passages") or [])):
            add_passage(row, slot, selected_at(row, slot))
            if len(passages) >= cap:
                break

    with settings.corpus_jsonl.open("w", encoding="utf-8") as f:
        for p in passages.values():
            f.write(json.dumps({
                "pid": p.pid, "text": p.text, "eng": p.eng,
                "query_ids": p.query_ids, "answers": p.answers,
                "query_types": p.query_types}, ensure_ascii=False) + "\n")
    with settings.eval_jsonl.open("w", encoding="utf-8") as f:
        for e in eval_rows:
            f.write(json.dumps({
                "qid": e.qid, "query": e.query, "answer": e.answer,
                "gold_pid": e.gold_pid, "gold_text": e.gold_text,
                "query_type": e.query_type}, ensure_ascii=False) + "\n")
    return settings.corpus_jsonl, settings.eval_jsonl


def read_corpus(path: Path) -> list[Passage]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(Passage(d["pid"], d["text"], d.get("eng", ""),
                               d.get("query_ids", []), d.get("answers", []),
                               d.get("query_types", [])))
    return out


def read_eval(path: Path) -> list[EvalRow]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(EvalRow(d["qid"], d["query"], d["answer"],
                               d.get("gold_pid"), d.get("gold_text", ""),
                               d.get("query_type", "")))
    return out