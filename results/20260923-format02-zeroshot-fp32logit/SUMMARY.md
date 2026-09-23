# 20260923-format02-zeroshot-fp32logit

<!-- GENERATED from metrics.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · quantization `nf4-bf16` · engine `hf`
- **Data**: `pngwn/system-one-decisions` (CC-BY-NC-4.0), report split `test`, calibration fitted on `val`
- **Prompt format**: `0.2` · template `qwen3.8-chatml-nothink`
- **Commit**: `7a1502f22d4b` **(tree dirty at run time)**
- **Created**: 2026-09-23T09:35:19+00:00 · Windows 10 · Python 3.11.16

- **Kernels**: attention `sdpa` · accelerated: `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule` · torch: `causal_conv1d_fn`, `causal_conv1d_update`

## Coverage

| split | kept | total | fraction | dropped >26 options | dropped invalid |
|---|---|---|---|---|---|
| val | 1292 | 1452 | 89.0% | 160 (banking77 100, tickets_queue 60) | 0 |
| test | 1520 | 1751 | 86.8% | 230 (banking77 150, tickets_queue 80) | 1 |

## Headline

| row | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model, uncalibrated | 1520 | 0.7638 | 0.1032 | 0.206 | 0.1656 | 0.3536 | 0.7476 | 0.7525 | 0.867 | 0.8289 | 0.6959 | 0.8048 | 0.5655 |
| model, calibrated | 1520 | 0.7638 | 0.0428 | 0.115 | 0.1545 | 0.3369 | 0.5952 | 0.7580 | 0.736 | 0.8306 | 0.5492 | 0.7997 | 0.4570 |
| control: uniform | 1520 | 0.4138 | 0.0790 | 0.136 | 0.2126 | 0.6406 | 1.0986 | 0.7058 | 0.359 | 0.4688 | 0.2500 | 0.4452 | 0.2000 |
| control: base_rate | 1520 | 0.4296 | 0.0341 | 0.110 | 0.2149 | 0.6236 | 1.0700 | 0.6987 | 0.454 | 0.4803 | 0.2922 | 0.4547 | 0.2774 |

Temperature scaling cannot change which option wins, so accuracy is identical in the first two rows by construction. Read the controls before the model: a predictor that ignores the state can post a competitive ECE, which is why ECE never appears here without accuracy, Brier and AUROC beside it — and why the section below reports skill against those controls rather than ECE alone.

## Skill against the controls

`BSS = 1 - Brier / Brier_control`. 1.0 is perfect, 0.0 is no better than the control, negative is worse. **This is the headline calibration number**, and ECE sits beside it rather than above it, because ECE can be won by a predictor that never commits: the base-rate control does exactly that on this set. A proper scoring rule cannot be won that way, so stating the result as skill *against that control* puts the comparison in the open.

| model | control | BSS | accuracy gain |
|---|---|---|---|
| uncalibrated | uniform | +0.4480 | +0.3500 |
| uncalibrated | base_rate | +0.4329 | +0.3342 |
| calibrated | uniform | +0.4741 | +0.3500 |
| calibrated | base_rate | +0.4598 | +0.3342 |

**BSS here is the multiclass Brier skill**, over the full distribution — one number, so the report never shows two things both called BSS. The top-label variant is in `metrics.json` under `skill.*.brier_top_label` for anyone who wants the confidence-only view; it is systematically smaller because the top-label score ignores how the remaining mass is spread.

## Risk-coverage

Questions are answered most-confident-first; at coverage *c* the least confident `1 - c` are abstained on. This is the deployment question — *if the least confident 20% go to a human, how good is what is left?* — and it depends only on the **order** of the confidences, not on their values, which makes it a second opinion rather than a restatement of the reliability diagram. `thr@80%` is the confidence cut that produces 80% coverage — the number an operator configures, where `acc@80%` is what they get for it. Calibration still moves it, because our temperatures are fitted per option-count bucket and therefore re-rank questions across buckets; a single global temperature would leave these rows identical. `AURC` is the area under the risk curve; lower is better.

| model | acc@20% | acc@40% | acc@60% | acc@80% | acc@100% | thr@80% | AURC |
|---|---|---|---|---|---|---|---|
| uncalibrated | 0.9408 | 0.9062 | 0.8805 | 0.8289 | 0.7638 | 0.6959 | 0.1154 |
| calibrated | 0.9441 | 0.9128 | 0.8893 | 0.8306 | 0.7638 | 0.5492 | 0.1142 |

Drawn in `risk-coverage.png`, with the negative controls on the same axes.

## Calibration

| bucket | n | temperature | NLL before | NLL after |
|---|---|---|---|---|
| 2 | 695 | 2.5021 | 0.6727 | 0.5249 |
| 3-5 | 597 | 2.2498 | 0.9591 | 0.7405 |

## By option-count bucket

| bucket | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 721 | 0.7920 | 0.0899 | 0.155 | 0.1546 | 0.3092 | 0.5754 | 0.7028 | 0.875 | 0.8492 | 0.7527 | 0.8228 | 0.6161 |
| 2 (calibrated) | 721 | 0.7920 | 0.0840 | 0.144 | 0.1535 | 0.3070 | 0.4838 | 0.7028 | 0.743 | 0.8492 | 0.6094 | 0.8228 | 0.5471 |
| 3-5 | 799 | 0.7384 | 0.1214 | 0.370 | 0.1756 | 0.3937 | 0.9030 | 0.7943 | 0.860 | 0.8125 | 0.6571 | 0.7778 | 0.5258 |
| 3-5 (calibrated) | 799 | 0.7384 | 0.0463 | 0.138 | 0.1554 | 0.3639 | 0.6957 | 0.7931 | 0.730 | 0.8078 | 0.4685 | 0.7792 | 0.3776 |

## By primitive

| primitive | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| choice | 519 | 0.8227 | 0.0897 | 0.276 | 0.1237 | 0.2729 | 0.6993 | 0.8253 | 0.912 | 0.8990 | 0.8426 | 0.8675 | 0.6398 |
| choice (calibrated) | 519 | 0.8227 | 0.0368 | 0.143 | 0.1145 | 0.2602 | 0.5196 | 0.8244 | 0.805 | 0.9014 | 0.5851 | 0.8654 | 0.4655 |
| noul | 721 | 0.7920 | 0.0899 | 0.155 | 0.1546 | 0.3092 | 0.5754 | 0.7028 | 0.875 | 0.8492 | 0.7527 | 0.8228 | 0.6161 |
| noul (calibrated) | 721 | 0.7920 | 0.0840 | 0.144 | 0.1535 | 0.3070 | 0.4838 | 0.7028 | 0.743 | 0.8492 | 0.6094 | 0.8228 | 0.5471 |
| score | 280 | 0.5821 | 0.2053 | 0.441 | 0.2716 | 0.6176 | 1.2806 | 0.6600 | 0.762 | 0.6250 | 0.5349 | 0.6032 | 0.4785 |
| score (calibrated) | 280 | 0.5821 | 0.1108 | 0.297 | 0.2313 | 0.5562 | 1.0222 | 0.6578 | 0.591 | 0.6339 | 0.3769 | 0.6071 | 0.3354 |

## By family

| family | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | 200 | 0.8750 | 0.0897 | 0.286 | 0.1049 | 0.2129 | 0.7371 | 0.8293 | 0.965 | 0.9313 | 0.9919 | 0.9111 | 0.9011 |
| ag_news (calibrated) | 200 | 0.8750 | 0.0712 | 0.251 | 0.0937 | 0.1970 | 0.4169 | 0.8299 | 0.912 | 0.9250 | 0.8434 | 0.9111 | 0.6683 |
| go_emotions | 641 | 0.7769 | 0.0968 | 0.187 | 0.1630 | 0.3260 | 0.5647 | 0.7002 | 0.860 | 0.8363 | 0.7167 | 0.8059 | 0.5977 |
| go_emotions (calibrated) | 641 | 0.7769 | 0.0888 | 0.194 | 0.1619 | 0.3237 | 0.5014 | 0.7002 | 0.720 | 0.8363 | 0.5917 | 0.8059 | 0.5395 |
| mmlu | 249 | 0.7992 | 0.0823 | 0.192 | 0.1195 | 0.2859 | 0.6578 | 0.8687 | 0.881 | 0.9000 | 0.7119 | 0.8533 | 0.5223 |
| mmlu (calibrated) | 249 | 0.7992 | 0.0590 | 0.197 | 0.1153 | 0.2815 | 0.5657 | 0.8686 | 0.748 | 0.8950 | 0.4952 | 0.8489 | 0.4031 |
| tickets_language | 80 | 0.9125 | 0.0902 | 0.500 | 0.0874 | 0.1748 | 0.6606 | 0.2661 | 0.997 | 0.8906 | 0.9941 | 0.9028 | 0.9924 |
| tickets_language (calibrated) | 80 | 0.9125 | 0.1347 | 0.458 | 0.0862 | 0.1724 | 0.3424 | 0.2661 | 0.924 | 0.8906 | 0.8861 | 0.9028 | 0.8750 |
| tickets_priority | 80 | 0.3250 | 0.3403 | 0.919 | 0.3366 | 0.8901 | 1.8997 | 0.4779 | 0.580 | 0.3125 | 0.4604 | 0.3194 | 0.4048 |
| tickets_priority (calibrated) | 80 | 0.3250 | 0.2631 | 0.665 | 0.2701 | 0.7975 | 1.5220 | 0.4402 | 0.410 | 0.3125 | 0.3246 | 0.3056 | 0.3017 |
| tickets_type | 70 | 0.7571 | 0.1567 | 0.539 | 0.1927 | 0.3983 | 0.7388 | 0.6926 | 0.874 | 0.7679 | 0.7834 | 0.7778 | 0.6381 |
| tickets_type (calibrated) | 70 | 0.7571 | 0.1786 | 0.610 | 0.1709 | 0.3646 | 0.6491 | 0.6848 | 0.701 | 0.7500 | 0.5396 | 0.7619 | 0.4832 |
| yelp_score | 200 | 0.6850 | 0.1971 | 0.416 | 0.2456 | 0.5086 | 1.0329 | 0.6375 | 0.836 | 0.7125 | 0.6590 | 0.6944 | 0.5570 |
| yelp_score (calibrated) | 200 | 0.6850 | 0.1362 | 0.435 | 0.2158 | 0.4597 | 0.8223 | 0.6298 | 0.664 | 0.7063 | 0.4702 | 0.6778 | 0.4177 |

## Timing

| split | questions | states | wall (s) | ms/question |
|---|---|---|---|---|
| val | 1292 | 760 | 331 | 256 |
| test | 1520 | 928 | 403 | 265 |

## Files

- `metrics.json` — the source of truth for every number above.
- `predictions-*.jsonl` — raw masked logits per question; re-score with `uv run task eval --rescore results/<run_id>` without a GPU.
- `calibration.json` — fitted temperatures, tied to the quantization above.
- `reliability-*.png`, `risk-coverage.png` — drawn from `metrics.json`.
