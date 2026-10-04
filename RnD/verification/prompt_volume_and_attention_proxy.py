"""Verifies the index-time cost numbers of Sect. 3.5, Table 6 and Table 7 of the Springer manuscript:
  prompt volume in characters sent to the summariser for the full-document baseline (18.73 M), fixed k = 0..4
  (2.11 / 4.25 / 6.36 / 8.43 / 10.32 M), the routed pipeline (4.39 M = 23.4 % of full, 69.0 % of k=2) and its
  replicate; the quadratic attention proxy sum(P_i^2) ratios (full/k=3 = 4.56, full/routed = 10.81); the
  approximate token counts (chars / 4.17); the typical k=2 prompt (~4 180 tokens) versus the 12 190-token bound.
Re-implements cost()/slice_chars() of RnD/dynamic_slice_length/router_experiments/scripts/cost.py: the prompt of
one call is the window text (boundary-shifted, as in the generation notebooks) plus the chunk itself; the
baseline prompt is the space-joined document plus the chunk; the paper's fixed k=0 ablation still calls the
summariser with the chunk as its own slice; a routed k=0 means no call.  System prompt (789 chars) excluded.
Run from the repository root:  .venv/bin/python RnD/verification/prompt_volume_and_attention_proxy.py"""
import json
from _common import SPLIT, DYNAMIC_JSON, REAL_RUNS, load_docs

docs = load_docs(SPLIT)
CHARS_PER_TOKEN = 1056773 / 253488     # from chunk_character_statistics.py / chunk_token_statistics.py
SYSTEM_TOKENS = 789 / CHARS_PER_TOKEN

def slice_chars(chunks, i, k):
    if k == 0:
        return 0
    lo, hi = i - k, i + k
    if lo < 0:
        hi, lo = min(len(chunks) - 1, hi - lo), 0
    if hi > len(chunks) - 1:
        lo, hi = max(0, lo - (hi - (len(chunks) - 1))), len(chunks) - 1
    return sum(len(c["text"]) for c in chunks[lo:hi + 1])

def routing(records):
    """(document, index within document) -> radius, from a generated-chunks file in notebook order."""
    out, counters = {}, {}
    for r in records:
        j = counters.get(r["document"], 0); counters[r["document"]] = j + 1
        out[(r["document"], j)] = r["doc_slice_radius"]
    return out

def cost(policy):
    calls = chars = 0; sq = 0.0
    for name, chunks in docs.items():
        for i, c in enumerate(chunks):
            k = policy(name, i)
            if k == "full":
                P = sum(len(x["text"]) for x in chunks) + len(chunks) - 1 + len(c["text"])
            elif k == "self":
                P = 2 * len(c["text"])
            elif k > 0:
                P = slice_chars(chunks, i, k) + len(c["text"])
            else:
                continue
            calls += 1; chars += P; sq += P * P
    return calls, chars, sq

primary = routing(json.load(open(DYNAMIC_JSON, encoding="utf-8")))
replicate = routing(json.load(open(REAL_RUNS / "binary_t2_heldout_chunks.json", encoding="utf-8")))
policies = {"full document": lambda n, i: "full", "fixed k=0 (self-slice)": lambda n, i: "self"}
for kk in range(1, 5):
    policies[f"fixed k={kk}"] = (lambda kk: (lambda n, i: kk))(kk)
policies["routed (primary run)"] = lambda n, i: primary[(n, i)]
policies["routed (replicate)"] = lambda n, i: replicate[(n, i)]
res = {name: cost(p) for name, p in policies.items()}
full, k2 = res["full document"], res["fixed k=2"]
print(f"{'policy':24s} {'calls':>5s} {'chars':>10s} {'M':>6s} {'vs full':>8s} {'vs k=2':>7s} {'sumP2 full/this':>15s} {'~tokens':>9s}")
for name, (calls, chars, sq) in res.items():
    print(f"{name:24s} {calls:5d} {chars:10d} {chars/1e6:6.2f} {chars/full[1]:8.1%} {chars/k2[1]:7.1%} {full[2]/sq:15.2f} {chars/CHARS_PER_TOKEN:9.0f}")
mean_k2_prompt = k2[1] / k2[0] / CHARS_PER_TOKEN + SYSTEM_TOKENS
print(f"\nmean fixed-k=2 prompt = {k2[1]}/{k2[0]}/{CHARS_PER_TOKEN:.2f} + {SYSTEM_TOKENS:.0f} = {mean_k2_prompt:.0f} tokens; window of median-length chunks = 6*450+{SYSTEM_TOKENS:.0f} = {6*450+SYSTEM_TOKENS:.0f}; bound s+(2k+2)*2000 = {SYSTEM_TOKENS+12000:.0f} (k=2), {SYSTEM_TOKENS+16000:.0f} (k=3)")
