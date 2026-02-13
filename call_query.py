import pandas as pd
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd

# ----------------------------
# 2) Token configuration
# ----------------------------

# Optional: map common typos / synonyms to canonical tokens
ALIAS = {
    # Typos
    "SQUIGLE": "SQUIGGLE",
    "ASENDING": "ASC",
    # "BB": "TIGHT"
    "ASCENDING": "ASC",
    "DESCENDING": "DESC",
    # You asked to keep UPSWEEP/DOWNSWEEP distinct, so no collapsing here.
    # But you can add aliases like "UPSWEEPING":"UPSWEEP" if you want.
}

# Token weights: core contour > SBI > length/type > modifiers/speed
TOKEN_WEIGHT = {
    # Types
    "BB": 3.5,
    "PULSED": 3.5,
    "GAP": 2.0,
    "T": 1.0,  # if you ever include it

    # Length
    "SHORT": 2.0,
    "LONG": 2.0,

    # Contour / shape (core)
    "ASC": 6.0,
    "DESC": 6.0,
    "FLAT": 5.0,
    "PEAK": 6.0,
    "DIP": 6.0,
    "SQUIGGLE": 6.0,
    "UPSWEEP": 6.0,
    "DOWNSWEEP": 6.0,

    # Speed (bonus)
    "FAST": 1.2,
    "SLOW": 1.2,

    # SBI / PRR-like (important, but not as core as contour)
    "SBI_TIGHT": 4.0,
    "SBI_WIDE": 4.0,
    "SBI_INC": 4.0,
    "SBI_DEC": 4.0,

    # Peak modifiers (bonus)
    "LEFT": 1.0,
    "RIGHT": 1.0,
    "SMALL": 1.0,
    "LARGE": 1.0,

    # Rare (high value)
    "BIPHO": 7.0,
    "SINGLE_TONE": 7.0,

    # Uncertainty / unknown (if you later add)
    "UNCERTAIN": 0.5,
}

DEFAULT_W = 2.0  # fallback for tokens not in TOKEN_WEIGHT

# SBI tokens to filter out when ignore_sbi=True
SBI_TOKENS = {"SBI_TIGHT", "SBI_WIDE", "SBI_INC", "SBI_DEC"}


def weight(tok: str) -> float:
    return float(TOKEN_WEIGHT.get(tok, DEFAULT_W))


def filter_sbi(tokens: Set[str]) -> Set[str]:
    """Remove SBI tokens from a set of tokens."""
    return tokens - SBI_TOKENS



# ----------------------------
# 3) Parsing (simple, robust)
# ----------------------------

TOKEN_RE = re.compile(r"[A-Z_]+")

def normalize_token(tok: str) -> str:
    tok = tok.strip().upper()
    tok = ALIAS.get(tok, tok)
    return tok

def extract_tokens(s: str) -> List[str]:
    """
    Extracts tokens from any string:
    - supports your "(...)" grouping but doesn't require it
    - ignores punctuation, commas, pipes, etc.
    """
    raw = TOKEN_RE.findall(s.upper())
    toks = [normalize_token(t) for t in raw if t.strip()]
    return toks

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
# 4) Scoring
# ----------------------------

# Define "optional-ish" tokens: missing them should not tank score.
OPTIONAL_TOKENS = {
    "FAST", "SLOW",
    "LEFT", "RIGHT", "SMALL", "LARGE",
    "SHORT", "LONG",
    "T",
}

def weighted_jaccard(query: Set[str], doc: Set[str]) -> float:
    """
    Weighted Jaccard with optional-token handling:
    - query tokens in OPTIONAL_TOKENS contribute less to union penalty
    """
    inter = query & doc
    union = query | doc

    inter_w = sum(weight(t) for t in inter)

    union_w = 0.0
    for t in union:
        w = weight(t)
        # reduce penalty for optional tokens that are only in query but not doc
        if (t in OPTIONAL_TOKENS) and (t in query) and (t not in doc):
            w *= 0.25
        union_w += w

    return inter_w / union_w if union_w > 0 else 0.0

def score_call(query_tokens: Set[str],
               entry: CallEntry,
               require: Optional[Set[str]] = None,
               forbid: Optional[Set[str]] = None,
               ignore_sbi: bool = False) -> float:
    """
    Basic retrieval score:
    - optional hard constraints via require/forbid
    - ignore_sbi: if True, filter out SBI tokens before comparison
    """
    q_toks = query_tokens
    e_toks = entry.tokens
    
    if ignore_sbi:
        q_toks = filter_sbi(q_toks)
        e_toks = filter_sbi(e_toks)
    
    if require:
        req = filter_sbi(require) if ignore_sbi else require
        if not req.issubset(e_toks):
            return 0.0
    if forbid:
        forb = filter_sbi(forbid) if ignore_sbi else forbid
        if len(forb & e_toks) > 0:
            return 0.0
    return weighted_jaccard(q_toks, e_toks)


# ----------------------------
# 5) Query language (minimal)
# ----------------------------
# Supported:
#   "tokens: ASC FAST PEAK FLAT SBI_INC"
#   "+BIPHO -GAP ASC PEAK"
#   "require:(BIPHO) forbid:(GAP) ASC PEAK"
#
# You can keep it super simple: just type tokens separated by spaces.

def parse_query(q: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns (query_tokens, require_tokens, forbid_tokens)
    Rules:
      - "+TOK" forces require
      - "-TOK" forces forbid
      - everything else is a soft token in query_tokens
    """
    toks = extract_tokens(q)
    require, forbid, soft = set(), set(), set()

    # We also parse + and - directly from the raw string to preserve intent
    raw_parts = re.findall(r"([+-])\s*([A-Za-z_]+)", q)
    for sign, tok in raw_parts:
        t = normalize_token(tok)
        if sign == "+":
            require.add(t)
        else:
            forbid.add(t)

    # Soft tokens: all extracted tokens that aren't explicitly forbid/require markers
    for t in toks:
        if t not in require and t not in forbid:
            soft.add(t)

    # Also include require tokens in soft tokens (so they contribute to similarity too)
    soft |= require

    return soft, require, forbid


# ----------------------------
# 6) Search function + demo
# ----------------------------

def search_calls(q: str,
                 index: List[CallEntry],
                 topk: int = 10,
                 ignore_sbi: bool = False) -> pd.DataFrame:
    q_tokens, require, forbid = parse_query(q)
    # print(q_tokens, require, forbid)

    rows = []
    for entry in index:
        s = score_call(q_tokens, entry, require=require, forbid=forbid, ignore_sbi=ignore_sbi)
        if s > 0:
            # For display purposes, use filtered tokens if ignore_sbi is True
            q_display = filter_sbi(q_tokens) if ignore_sbi else q_tokens
            e_display = filter_sbi(entry.tokens) if ignore_sbi else entry.tokens
            
            rows.append({
                "Call": entry.call_id,
                "Score": s,
                "Repr": entry.repr_str,
                "Matched": " ".join(sorted(q_display & e_display)),
                "Missing(optional downweighted)": " ".join(sorted((q_display - e_display) & OPTIONAL_TOKENS)),
                "Missing(core)": " ".join(sorted((q_display - e_display) - OPTIONAL_TOKENS)),
            })

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).head(topk)
    return df.reset_index(drop=True)


