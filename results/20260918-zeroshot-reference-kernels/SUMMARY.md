# 20260918-zeroshot-reference-kernels

<!-- GENERATED from metrics.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · quantization `nf4-bf16` · engine `hf`
- **Data**: `pngwn/system-one-decisions` (CC-BY-NC-4.0), report split `test`, calibration fitted on `val`
- **Prompt format**: `0.2` · template `qwen3.8-chatml-nothink`
- **Commit**: `8c4152585f71` **(tree dirty at run time)**
- **Created**: 2026-09-17T23:37:35+00:00 · Windows 10 · Python 3.11.16

- **Kernels**: attention `sdpa` · accelerated: `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule` · torch: `causal_conv1d_fn`, `causal_conv1d_update`

## Coverage

| split | kept | total | fraction | dropped >26 options | dropped invalid |
|---|---|---|---|---|---|
| val | 1292 | 1452 | 89.0% | 160 (banking77 100, tickets_queue 60) | 0 |
| test | 1520 | 1751 | 86.8% | 230 (banking77 150, tickets_queue 80) | 1 |

## Headline

| row | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model, uncalibrated | 1520 | 0.7645 | 0.1026 | 0.176 | 0.1656 | 0.3534 | 0.7468 | 0.7519 | 0.867 | 0.8314 | 0.7052 | 0.8019 | 0.5622 |
| model, calibrated | 1520 | 0.7645 | 0.0446 | 0.096 | 0.1546 | 0.3367 | 0.5948 | 0.7570 | 0.736 | 0.8314 | 0.5497 | 0.7997 | 0.4592 |
| control: uniform | 1520 | 0.4138 | 0.0790 | 0.136 | 0.2126 | 0.6406 | 1.0986 | 0.7058 | 0.359 | 0.4688 | 0.2500 | 0.4452 | 0.2000 |
| control: base_rate | 1520 | 0.4296 | 0.0341 | 0.110 | 0.2149 | 0.6236 | 1.0700 | 0.6987 | 0.454 | 0.4803 | 0.2922 | 0.4547 | 0.2774 |

Temperature scaling cannot change which option wins, so accuracy is identical in the first two rows by construction. Read the controls before the model: a predictor that ignores the state can post a competitive ECE, which is why ECE never appears here without accuracy, Brier and AUROC beside it — and why the section below reports skill against those controls rather than ECE alone.

## Skill against the controls

`BSS = 1 - Brier / Brier_control`. 1.0 is perfect, 0.0 is no better than the control, negative is worse. **This is the headline calibration number**, and ECE sits beside it rather than above it, because ECE can be won by a predictor that never commits: the base-rate control does exactly that on this set. A proper scoring rule cannot be won that way, so stating the result as skill *against that control* puts the comparison in the open.

| model | control | BSS | accuracy gain |
|---|---|---|---|
| uncalibrated | uniform | +0.4484 | +0.3507 |
| uncalibrated | base_rate | +0.4334 | +0.3349 |
| calibrated | uniform | +0.4745 | +0.3507 |
| calibrated | base_rate | +0.4602 | +0.3349 |

**BSS here is the multiclass Brier skill**, over the full distribution — one number, so the report never shows two things both called BSS. The top-label variant is in `metrics.json` under `skill.*.brier_top_label` for anyone who wants the confidence-only view; it is systematically smaller because the top-label score ignores how the remaining mass is spread.

## Risk-coverage

Questions are answered most-confident-first; at coverage *c* the least confident `1 - c` are abstained on. This is the deployment question — *if the least confident 20% go to a human, how good is what is left?* — and it depends only on the **order** of the confidences, not on their values, which makes it a second opinion rather than a restatement of the reliability diagram. `thr@80%` is the confidence cut that produces 80% coverage — the number an operator configures, where `acc@80%` is what they get for it. Calibration still moves it, because our temperatures are fitted per option-count bucket and therefore re-rank questions across buckets; a single global temperature would leave these rows identical. `AURC` is the area under the risk curve; lower is better.

| model | acc@20% | acc@40% | acc@60% | acc@80% | acc@100% | thr@80% | AURC |
|---|---|---|---|---|---|---|---|
| uncalibrated | 0.9375 | 0.9046 | 0.8805 | 0.8314 | 0.7645 | 0.7052 | 0.1159 |
| calibrated | 0.9441 | 0.9128 | 0.8914 | 0.8314 | 0.7645 | 0.5497 | 0.1136 |

Drawn in `risk-coverage.png`, with the negative controls on the same axes.

## Calibration

| bucket | n | temperature | NLL before | NLL after |
|---|---|---|---|---|
| 2 | 695 | 2.5064 | 0.6737 | 0.5253 |
| 3-5 | 597 | 2.2526 | 0.9604 | 0.7408 |

## By option-count bucket

| bucket | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 721 | 0.7920 | 0.0874 | 0.145 | 0.1545 | 0.3090 | 0.5745 | 0.7034 | 0.875 | 0.8475 | 0.7549 | 0.8228 | 0.6225 |
| 2 (calibrated) | 721 | 0.7920 | 0.0755 | 0.145 | 0.1533 | 0.3067 | 0.4833 | 0.7039 | 0.743 | 0.8475 | 0.6104 | 0.8228 | 0.5497 |
| 3-5 | 799 | 0.7397 | 0.1202 | 0.314 | 0.1756 | 0.3934 | 0.9023 | 0.7922 | 0.860 | 0.8109 | 0.6542 | 0.7778 | 0.5203 |
| 3-5 (calibrated) | 799 | 0.7397 | 0.0470 | 0.118 | 0.1558 | 0.3637 | 0.6954 | 0.7912 | 0.730 | 0.8094 | 0.4684 | 0.7819 | 0.3780 |

## By primitive

| primitive | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| choice | 519 | 0.8247 | 0.0876 | 0.247 | 0.1243 | 0.2730 | 0.6992 | 0.8219 | 0.912 | 0.9038 | 0.8414 | 0.8675 | 0.6397 |
| choice (calibrated) | 519 | 0.8247 | 0.0412 | 0.141 | 0.1154 | 0.2602 | 0.5196 | 0.8202 | 0.805 | 0.8990 | 0.5836 | 0.8654 | 0.4623 |
| noul | 721 | 0.7920 | 0.0874 | 0.145 | 0.1545 | 0.3090 | 0.5745 | 0.7034 | 0.875 | 0.8475 | 0.7549 | 0.8228 | 0.6225 |
| noul (calibrated) | 721 | 0.7920 | 0.0755 | 0.145 | 0.1533 | 0.3067 | 0.4833 | 0.7039 | 0.743 | 0.8475 | 0.6104 | 0.8228 | 0.5497 |
| score | 280 | 0.5821 | 0.1930 | 0.441 | 0.2706 | 0.6165 | 1.2785 | 0.6611 | 0.763 | 0.6295 | 0.5360 | 0.6032 | 0.4882 |
| score (calibrated) | 280 | 0.5821 | 0.1101 | 0.349 | 0.2307 | 0.5556 | 1.0214 | 0.6598 | 0.591 | 0.6384 | 0.3772 | 0.6032 | 0.3380 |

## By family

| family | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | thr@80% | acc@90% | thr@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | 200 | 0.8700 | 0.0949 | 0.287 | 0.1048 | 0.2128 | 0.7362 | 0.8406 | 0.965 | 0.9313 | 0.9915 | 0.9111 | 0.9045 |
| ag_news (calibrated) | 200 | 0.8700 | 0.0761 | 0.250 | 0.0931 | 0.1967 | 0.4163 | 0.8400 | 0.912 | 0.9250 | 0.8435 | 0.9111 | 0.6681 |
| go_emotions | 641 | 0.7769 | 0.0934 | 0.211 | 0.1629 | 0.3258 | 0.5637 | 0.7008 | 0.860 | 0.8382 | 0.7311 | 0.8042 | 0.5927 |
| go_emotions (calibrated) | 641 | 0.7769 | 0.0856 | 0.194 | 0.1617 | 0.3234 | 0.5009 | 0.7014 | 0.720 | 0.8382 | 0.5984 | 0.8059 | 0.5373 |
| mmlu | 249 | 0.8072 | 0.0743 | 0.206 | 0.1209 | 0.2862 | 0.6583 | 0.8572 | 0.881 | 0.8950 | 0.7235 | 0.8533 | 0.5191 |
| mmlu (calibrated) | 249 | 0.8072 | 0.0615 | 0.195 | 0.1177 | 0.2817 | 0.5659 | 0.8573 | 0.748 | 0.8950 | 0.5032 | 0.8533 | 0.4042 |
| tickets_language | 80 | 0.9125 | 0.0902 | 0.600 | 0.0874 | 0.1748 | 0.6610 | 0.2661 | 0.997 | 0.8906 | 0.9941 | 0.9028 | 0.9924 |
| tickets_language (calibrated) | 80 | 0.9125 | 0.1350 | 0.556 | 0.0863 | 0.1725 | 0.3424 | 0.2661 | 0.924 | 0.8906 | 0.8854 | 0.9028 | 0.8749 |
| tickets_priority | 80 | 0.3250 | 0.3386 | 0.920 | 0.3355 | 0.8902 | 1.8984 | 0.4815 | 0.580 | 0.3125 | 0.4513 | 0.3333 | 0.3984 |
| tickets_priority (calibrated) | 80 | 0.3250 | 0.2383 | 0.664 | 0.2689 | 0.7970 | 1.5214 | 0.4480 | 0.410 | 0.3125 | 0.3257 | 0.3194 | 0.3015 |
| tickets_type | 70 | 0.7571 | 0.1585 | 0.537 | 0.1923 | 0.3977 | 0.7393 | 0.6903 | 0.873 | 0.7679 | 0.7924 | 0.7778 | 0.6383 |
| tickets_type (calibrated) | 70 | 0.7571 | 0.1788 | 0.611 | 0.1709 | 0.3647 | 0.6497 | 0.6815 | 0.700 | 0.7500 | 0.5438 | 0.7619 | 0.4766 |
| yelp_score | 200 | 0.6850 | 0.1969 | 0.416 | 0.2447 | 0.5071 | 1.0306 | 0.6393 | 0.836 | 0.7063 | 0.6674 | 0.6889 | 0.5534 |
| yelp_score (calibrated) | 200 | 0.6850 | 0.1163 | 0.433 | 0.2153 | 0.4590 | 0.8214 | 0.6314 | 0.664 | 0.7125 | 0.4684 | 0.6778 | 0.4218 |

## Timing

| split | questions | states | wall (s) | ms/question |
|---|---|---|---|---|
| val | 1292 | 760 | 335 | 260 |
| test | 1520 | 928 | 404 | 266 |

## Files

- `metrics.json` — the source of truth for every number above.
- `predictions-*.jsonl` — raw masked logits per question; re-score with `uv run task eval --rescore results/<run_id>` without a GPU.
- `calibration.json` — fitted temperatures, tied to the quantization above.
- `reliability-*.png`, `risk-coverage.png` — drawn from `metrics.json`.
