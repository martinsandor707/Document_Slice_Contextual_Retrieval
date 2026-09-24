| Configuration | Kind | R@20 | MRR@20 | MAR@20 | R@5 | R@10 | summariser calls / prompt text vs k=2 | Note |
|---|---|---|---|---|---|---|---|---|
| Anthropic full-document baseline | real (paper) | 0.9540 | 0.7863 | 3.935 | 0.7373 | 0.8650 | — |  |
| Hybrid retrieval, no context | real (paper) | 0.9393 | 0.7748 | 4.036 | 0.7173 | 0.8470 | 0 / 0% |  |
| Fixed k=0 (chunk-only summary) | real (paper) | 0.9373 | 0.7861 | 3.808 | 0.7570 | 0.8600 | — |  |
| Fixed k=1 | real (paper) | 0.9433 | 0.7990 | 3.726 | 0.7563 | 0.8703 | 382 / 67% |  |
| Fixed k=2 | real (paper) | 0.9467 | 0.7680 | 3.809 | 0.7453 | 0.8777 | 382 / 100% |  |
| Fixed k=3 | real (paper) | 0.9453 | 0.7877 | 4.020 | 0.7410 | 0.8590 | 382 / 133% |  |
| Dynamic router, original config (Martin's run) | real | 0.9380 | 0.7949 | 3.685 | 0.7597 | 0.8650 | 273 / 73% |  |
| Constant k=2 through the dynamic notebook (FIXED_K=2) | real (re-generation) | 0.9467 | 0.7923 | 3.921 | 0.7463 | 0.8643 | 382 / 100% |  |
| `sim_dyn_T1`: current routing but tables -> 1 | offline sim | 0.9500 | 0.7814 | 3.787 | 0.7603 | 0.8683 | 273 / 66% | k {'0': 107, '1': 95, '2': 161
| `sim_dyn_T1_D2`: tables -> 1, fragments D -> 2 | offline sim | 0.9500 | 0.7816 | 3.792 | 0.7570 | 0.8683 | 273 / 65% | k {'0': 107, '1': 95, '2': 17
| `sim_dyn_T1_B2`: tables -> 1, B -> 2 | offline sim | 0.9500 | 0.7693 | 3.821 | 0.7563 | 0.8703 | 273 / 70% | k {'0': 107, '1': 42, '2': 214, '3': 17
| `sim_dyn_T2`: current routing but tables -> 2 | offline sim | 0.9487 | 0.7822 | 3.636 | 0.7630 | 0.8797 | 273 / 70% | k {'0': 107, '1': 53, '2': 203
| `sim_dyn_T2_D2`: tables -> 2, fragments D -> 2 | offline sim | 0.9487 | 0.7821 | 3.643 | 0.7597 | 0.8797 | 273 / 69% | k {'0': 107, '1': 53, '2': 22
| `sim_dyn_T1_A1`: tables -> 1, A -> 1 (never skip) | offline sim | 0.9480 | 0.7875 | 3.704 | 0.7590 | 0.8643 | 382 / 83% | k {'1': 202, '2': 161, '3'
| `sim_all2`: fixed k=2 rebuilt from JSON (should match table ablation_doc_slice_radius_2) | offline sim | 0.9467 | 0.7680 | 3.809 | 0.7453 | 0.8777 |
| `sim_A_to_1`: A -> k=1 (never skip, cheap) | offline sim | 0.9467 | 0.7859 | 3.628 | 0.7657 | 0.8730 | 382 / 91% | k {'1': 160, '2': 161, '3': 59} |
| `sim_D_to_2`: A -> 2, D(slm) -> 2 | offline sim | 0.9467 | 0.7762 | 3.772 | 0.7527 | 0.8757 | 382 / 96% | k {'1': 53, '2': 327} |
| `sim_T_to_2`: A -> 2, tables -> 2 | offline sim | 0.9467 | 0.7763 | 3.762 | 0.7580 | 0.8757 | 382 / 97% | k {'1': 53, '2': 310, '3': 17} |
| `sim_dyn_skip`: current routing, contexts from fixed-k JSONs, k=0 skipped (proxy of the real dynamic run) | offline sim | 0.9447 | 0.7870 | 3.579 | 
| `sim_dyn_k0summary`: current routing, k=0 chunks get the fixed k=0 self-summary instead of nothing | offline sim | 0.9447 | 0.7879 | 3.637 | 0.7613 
| `sim_A_to_2`: A -> k=2 (never skip) | offline sim | 0.9447 | 0.7786 | 3.737 | 0.7533 | 0.8730 | 382 / 100% | k {'1': 53, '2': 268, '3': 59} |
| `sim_B_to_2`: A,B -> 2 (only C=2, D=3, T=3 differ from all-2) | offline sim | 0.9447 | 0.7706 | 3.773 | 0.7473 | 0.8750 | 382 / 105% | k {'2': 321, 
| `sim_len_s3_m2_l2`: length only: <700->3, else 2, T->3 | offline sim | 0.9447 | 0.7715 | 3.760 | 0.7400 | 0.8737 | 382 / 111% | k {'2': 260, '3': 12
| `sim_len_s3_m3_l2`: length only: <=2500->3, >2500->2, T->3 | offline sim | 0.9447 | 0.7740 | 3.744 | 0.7520 | 0.8697 | 382 / 122% | k {'2': 132, '3'
| `sim_pos_edge1_skip`: position only: first/last chunk skipped, else 2, T->3 | offline sim | 0.9447 | 0.7701 | 3.742 | 0.7520 | 0.8790 | 333 / 90% | 
| `sim_T_to_1`: A -> 2, tables -> 1 | offline sim | 0.9440 | 0.7765 | 3.812 | 0.7537 | 0.8643 | 382 / 93% | k {'1': 95, '2': 268, '3': 17} |
| `sim_B_to_3`: A -> 2, B -> 3 | offline sim | 0.9427 | 0.7832 | 3.655 | 0.7587 | 0.8803 | 382 / 109% | k {'2': 268, '3': 112} |
| `sim_pos_edge1`: position only: first/last chunk ->1, else 2, T->3 | offline sim | 0.9427 | 0.7772 | 3.695 | 0.7460 | 0.8750 | 382 / 99% | k {'1': 4
| `sim_anchor8_A1B2C3`: anchor few-shot: A->1 B->2 C->3, T->3 | offline sim | 0.9400 | 0.7866 | 3.754 | 0.7370 | 0.8703 | 382 / 107% | k {'1': 96, '2'
| `sim_anchor0_A1B2C3`: anchor zero-shot: A->1 B->2 C->3, T->3 | offline sim | 0.9400 | 0.7849 | 3.655 | 0.7490 | 0.8677 | 382 / 97% | k {'1': 111, '2
| `sim_C_to_3`: A -> 2, C -> 3 | offline sim | 0.9393 | 0.7831 | 3.929 | 0.7410 | 0.8557 | 382 / 113% | k {'1': 53, '2': 107, '3': 220} |
| `sim_C_to_1`: A -> 2, C -> 1 | offline sim | 0.9373 | 0.7970 | 3.639 | 0.7533 | 0.8723 | 382 / 86% | k {'1': 214, '2': 107, '3': 59} |
| `sim_len_s3_m2_l1`: length only: <700 chars->3, 700-2500->2, >2500->1, T->3 | offline sim | 0.9373 | 0.7811 | 3.711 | 0.7410 | 0.8733 | 382 / 99% | 
...
[chain] GPU memory in use at start: 15 MiB
Shell cwd was reset to /home/martin/projects/TDK/Document_Slice_Contextual_Retrieval