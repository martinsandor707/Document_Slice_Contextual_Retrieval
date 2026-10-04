# Verification scripts for the Springer Nature manuscript

Each script recomputes, from the frozen artefacts under `RnD/`, the numbers that the corresponding part of
`RnD/citds_article/sn_green_rag_article.tex` reports, and prints them next to the derivation. All scripts are
read-only. Run them from the repository root with the project interpreter, e.g.

    .venv/bin/python RnD/verification/chunk_token_statistics.py

| Script | Verifies |
|---|---|
| `chunk_character_statistics.py` | Table 1 rows 1–4 and 7–11, Table 4 (chunks, characters, questions, gold references, exemplar pool), Sect. 3.1 artefact counts |
| `chunk_token_statistics.py` | Table 1 rows 5–6 (token quartiles, bands, cap), Table 4 token means/medians for both corpora, the token median used in Sect. 3.5 |
| `prompt_length_statistics.py` | Sect. 3.2 (summariser system prompt 789 chars / 164 tokens, per-call bound), Sect. 3.3 (router system and user prompt lengths, ~2 000 tokens), Sect. 3.5 (typical k=2 prompt), Table 3; per-call character and nomic-token lengths of every summariser and router message for the baseline, fixed k = 0..4 and the routed runs |
| `prompt_volume_and_attention_proxy.py` | Sect. 3.5 prompt volumes and sum(P_i^2) ratios, Table 6 / Table 7 "prompt volume" columns, Fig. 4 data, the typical k=2 prompt vs the 12 190-token bound |
| `energy_deltas_from_emissions_csv.py` | Table 5, Table 6 energy column, Table 7 time/energy/GPU/emission rows and deltas, Table 8, Figs 7 and 9, abstract and Sect. 4.2 / 5.1 percentages, grid carbon intensity |
| `retrieval_significance_tests.py` | Table 7 `|Q_20^+|`, the provisional BCa bootstrap / McNemar footnote, per-question better/worse counts, no-hit counts (Sect. 3.6, 4.4) |
| `router_routing_statistics.py` | Sect. 4.3 / Table 7 routing statistics (k distribution, k-bar, calls, summary length, replicate), Sect. 3.3 gold-set agreement, the negative short-chunk/skip overlap check |
| `generated_summary_statistics.py` | Sect. 3.2 and 4.3 statements about the generated summaries: min / Q1 / median / mean / Q3 / max of summary length in characters and nomic tokens for every stored generation (baseline, fixed k = 0..4, routed runs, policy corpus), forbidden-phrase and degenerate-copy counts, per-radius breakdown |
| `latex_manuscript_consistency_check.py` | Structural checks of the .tex file: braces, environments, labels/references, citation keys, placeholders, word budgets, notes coverage |

`_common.py` holds the shared paths and the position-based reader for `emissions.csv` (the rows written by
codecarbon 3.3.1 are shifted by two columns after `region` relative to the 3.2.3 header).

Retrieval metrics themselves (Recall/MRR/MAR of Tables 6–8) are not recomputed here: they are the printed outputs of
`RnD/benchmarking_retrievals.ipynb` and `RnD/ablation_doc_slice_retrieval.ipynb` and require the GPU stack; the
per-question files used by `retrieval_significance_tests.py` are their stored per-question results.
