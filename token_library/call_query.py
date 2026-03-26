import pandas as pd
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from token_library.vocab import ALL_VALID_TOKENS, VOCAB
# ----------------------------
# Token configuration
# ----------------------------

ALIAS = {
    "SQUIGLE": "SQUIGGLE",
}

TOKEN_WEIGHT = {tok: 1.0 for tok in ALL_VALID_TOKENS}
DEFAULT_W = 1.0

def weight(tok: str) -> float:
    return float(TOKEN_WEIGHT.get(tok, DEFAULT_W))


# ----------------------------
# Parsing
# ----------------------------

TOKEN_RE = re.compile(r"[A-Z_]+")


def normalize_token(tok: str) -> str:
    tok = tok.strip().upper()
    tok = ALIAS.get(tok, tok)
    return tok


def extract_tokens(s: str) -> List[str]:
    raw = TOKEN_RE.findall(s.upper())
    return [normalize_token(t) for t in raw if t.strip()]


@dataclass
class CallEntry:
    call_id: str
    repr_str: str
    tokens: Set[str]


def build_index(db: Dict[str, str]) -> List[CallEntry]:
    entries = []
    for k, v in db.items():
        toks = set(extract_tokens(v))
        entries.append(CallEntry(call_id=k, repr_str=v, tokens=toks))
    return entries


# ----------------------------
# Scoring
# ----------------------------

def weighted_jaccard(query: Set[str], doc: Set[str]) -> float:
    inter = query & doc
    union = query | doc

    inter_w = sum(weight(t) for t in inter)
    union_w = sum(weight(t) for t in union)

    return inter_w / union_w if union_w > 0 else 0.0


def score_call(query_tokens: Set[str],
               entry: CallEntry,
               require: Optional[Set[str]] = None,
               forbid: Optional[Set[str]] = None) -> float:
    q_toks = query_tokens
    e_toks = entry.tokens

    if require and not require.issubset(e_toks):
        return 0.0
    if forbid and len(forbid & e_toks) > 0:
        return 0.0
    return weighted_jaccard(q_toks, e_toks)


# ----------------------------
# Query language
# ----------------------------
# "+TOK" forces require, "-TOK" forces forbid, everything else is soft.

def parse_query(q: str) -> Tuple[Set[str], Set[str], Set[str]]:
    toks = extract_tokens(q)
    require, forbid, soft = set(), set(), set()

    raw_parts = re.findall(r"([+-])\s*([A-Za-z_]+)", q)
    for sign, tok in raw_parts:
        t = normalize_token(tok)
        if sign == "+":
            require.add(t)
        else:
            forbid.add(t)

    for t in toks:
        if t not in require and t not in forbid:
            soft.add(t)

    soft |= require
    return soft, require, forbid


# ----------------------------
# Search
# ----------------------------

def search_calls(q: str,
                 index: List[CallEntry],
                 topk: int = 10) -> pd.DataFrame:
    q_tokens, require, forbid = parse_query(q)

    rows = []
    for entry in index:
        s = score_call(q_tokens, entry, require=require, forbid=forbid)
        if s > 0:
            rows.append({
                "Call": entry.call_id,
                "Score": s,
                "Repr": entry.repr_str,
                "Matched": " ".join(sorted(q_tokens & entry.tokens)),
                "Missing": " ".join(sorted(q_tokens - entry.tokens)),
            })

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).head(topk)
    return df.reset_index(drop=True)


def load_index_from_csv(path: str, call_col: str = "filename"):
    parts_df = pd.read_csv(path)
    parts_df = parts_df[parts_df["Annotated"] == True]
    part_cols = [f'P{i}' for i in range(1, 6)]

    def build_tokens_string(row):
        parts = []
        for col in part_cols:   
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(f"({str(val).strip()})")
        return ' '.join(parts) if parts else None

    parts_df['tokens_string'] = parts_df.apply(build_tokens_string, axis=1)

    DB = dict(zip(parts_df[call_col], parts_df['tokens_string']))

    index = build_index(DB)
    return index, DB, parts_df