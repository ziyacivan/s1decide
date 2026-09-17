# 20260917-format02-zeroshot

<!-- GENERATED from metrics.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · quantization `nf4-bf16` · engine `hf`
- **Data**: `pngwn/system-one-decisions` (CC-BY-NC-4.0), report split `test`, calibration fitted on `val`
- **Prompt format**: `0.2` · template `qwen3.8-chatml-nothink`
- **Commit**: `7ef9e68bcb22` **(tree dirty at run time)**
- **Created**: 2026-09-17T16:10:57+00:00 · Windows 10 · Python 3.11.16

## Coverage

| split | kept | total | fraction | dropped >26 options | dropped invalid |
|---|---|---|---|---|---|
| val | 1292 | 1452 | 89.0% | 160 (banking77 100, tickets_queue 60) | 0 |
| test | 1520 | 1751 | 86.8% | 230 (banking77 150, tickets_queue 80) | 0 |

## Headline

| row | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | acc@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model, uncalibrated | 1520 | 0.7645 | 0.1026 | 0.185 | 0.1656 | 0.3534 | 0.7468 | 0.7519 | 0.867 | 0.8314 | 0.8019 |
| model, calibrated | 1520 | 0.7645 | 0.0449 | 0.103 | 0.1546 | 0.3367 | 0.5948 | 0.7570 | 0.736 | 0.8314 | 0.7997 |
| control: uniform | 1520 | 0.4138 | 0.0767 | 0.134 | 0.2126 | 0.6406 | 1.0986 | 0.7058 | 0.359 | 0.4688 | 0.4452 |
| control: base_rate | 1520 | 0.4296 | 0.0342 | 0.111 | 0.2149 | 0.6236 | 1.0700 | 0.6987 | 0.454 | 0.4803 | 0.4547 |

Temperature scaling cannot change which option wins, so accuracy is identical in the first two rows by construction. Read the controls before the model: a predictor that ignores the state can post a competitive ECE, which is why ECE never appears here without accuracy, Brier and AUROC beside it — and why the section below reports skill against those controls rather than ECE alone.

## Skill against the controls

`BSS = 1 - Brier / Brier_control`. 1.0 is perfect, 0.0 is no better than the control, negative is worse. **This is the headline calibration number**, and ECE sits beside it rather than above it, because ECE can be won by a predictor that never commits: the base-rate control does exactly that on this set. A proper scoring rule cannot be won that way, so stating the result as skill *against that control* puts the comparison in the open.

| model | control | BSS (multiclass) | BSS (top label) | accuracy gain |
|---|---|---|---|---|
| uncalibrated | uniform | +0.4484 | +0.2210 | +0.3507 |
| uncalibrated | base_rate | +0.4334 | +0.2296 | +0.3349 |
| calibrated | uniform | +0.4745 | +0.2725 | +0.3507 |
| calibrated | base_rate | +0.4602 | +0.2806 | +0.3349 |

## Risk-coverage

Questions are answered most-confident-first; at coverage *c* the least confident `1 - c` are abstained on. This is the deployment question — *if the least confident 20% go to a human, how good is what is left?* — and unlike ECE it depends only on the **order** of the confidences, so temperature scaling cannot move it. That makes it a genuine second opinion rather than a restatement of the reliability diagram. `AURC` is the area under the risk curve; lower is better.

| model | acc@20% | acc@40% | acc@60% | acc@80% | acc@100% | AURC |
|---|---|---|---|---|---|---|
| uncalibrated | 0.9375 | 0.9046 | 0.8805 | 0.8314 | 0.7645 | 0.1159 |
| calibrated | 0.9441 | 0.9128 | 0.8914 | 0.8314 | 0.7645 | 0.1136 |

Drawn in `risk-coverage.png`, with the negative controls on the same axes.

## Calibration

| bucket | n | temperature | NLL before | NLL after |
|---|---|---|---|---|
| 2 | 695 | 2.5064 | 0.6737 | 0.5253 |
| 3-5 | 597 | 2.2526 | 0.9604 | 0.7408 |

## By option-count bucket

| bucket | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | acc@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 721 | 0.7920 | 0.0874 | 0.145 | 0.1545 | 0.3090 | 0.5745 | 0.7034 | 0.875 | 0.8475 | 0.8228 |
| 2 (calibrated) | 721 | 0.7920 | 0.0755 | 0.145 | 0.1533 | 0.3067 | 0.4833 | 0.7039 | 0.743 | 0.8475 | 0.8228 |
| 3-5 | 799 | 0.7397 | 0.1202 | 0.334 | 0.1756 | 0.3934 | 0.9023 | 0.7922 | 0.860 | 0.8109 | 0.7778 |
| 3-5 (calibrated) | 799 | 0.7397 | 0.0513 | 0.140 | 0.1558 | 0.3637 | 0.6954 | 0.7912 | 0.730 | 0.8094 | 0.7819 |

## By primitive

| primitive | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | acc@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| choice | 519 | 0.8247 | 0.0876 | 0.240 | 0.1243 | 0.2730 | 0.6992 | 0.8219 | 0.912 | 0.9038 | 0.8675 |
| choice (calibrated) | 519 | 0.8247 | 0.0399 | 0.139 | 0.1154 | 0.2602 | 0.5196 | 0.8202 | 0.805 | 0.8990 | 0.8654 |
| noul | 721 | 0.7920 | 0.0874 | 0.145 | 0.1545 | 0.3090 | 0.5745 | 0.7034 | 0.875 | 0.8475 | 0.8228 |
| noul (calibrated) | 721 | 0.7920 | 0.0755 | 0.145 | 0.1533 | 0.3067 | 0.4833 | 0.7039 | 0.743 | 0.8475 | 0.8228 |
| score | 280 | 0.5821 | 0.1932 | 0.520 | 0.2706 | 0.6165 | 1.2785 | 0.6611 | 0.763 | 0.6295 | 0.6032 |
| score (calibrated) | 280 | 0.5821 | 0.1139 | 0.400 | 0.2307 | 0.5556 | 1.0214 | 0.6598 | 0.591 | 0.6384 | 0.6032 |

## By family

| family | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf | acc@80% | acc@90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | 200 | 0.8700 | 0.0949 | 0.354 | 0.1048 | 0.2128 | 0.7362 | 0.8406 | 0.965 | 0.9313 | 0.9111 |
| ag_news (calibrated) | 200 | 0.8700 | 0.0679 | 0.337 | 0.0931 | 0.1967 | 0.4163 | 0.8400 | 0.912 | 0.9250 | 0.9111 |
| go_emotions | 641 | 0.7769 | 0.0940 | 0.235 | 0.1629 | 0.3258 | 0.5637 | 0.7008 | 0.860 | 0.8382 | 0.8042 |
| go_emotions (calibrated) | 641 | 0.7769 | 0.0861 | 0.195 | 0.1617 | 0.3234 | 0.5009 | 0.7014 | 0.720 | 0.8382 | 0.8059 |
| mmlu | 249 | 0.8072 | 0.0742 | 0.212 | 0.1209 | 0.2862 | 0.6583 | 0.8572 | 0.881 | 0.8950 | 0.8533 |
| mmlu (calibrated) | 249 | 0.8072 | 0.0629 | 0.255 | 0.1177 | 0.2817 | 0.5659 | 0.8573 | 0.748 | 0.8950 | 0.8533 |
| tickets_language | 80 | 0.9125 | 0.0902 | 0.800 | 0.0874 | 0.1748 | 0.6610 | 0.2661 | 0.997 | 0.8906 | 0.9028 |
| tickets_language (calibrated) | 80 | 0.9125 | 0.1362 | 0.756 | 0.0863 | 0.1725 | 0.3424 | 0.2661 | 0.924 | 0.8906 | 0.9028 |
| tickets_priority | 80 | 0.3250 | 0.2856 | 0.920 | 0.3355 | 0.8902 | 1.8984 | 0.4815 | 0.580 | 0.3125 | 0.3333 |
| tickets_priority (calibrated) | 80 | 0.3250 | 0.2729 | 0.664 | 0.2689 | 0.7970 | 1.5214 | 0.4480 | 0.410 | 0.3125 | 0.3194 |
| tickets_type | 70 | 0.7571 | 0.1736 | 0.640 | 0.1923 | 0.3977 | 0.7393 | 0.6903 | 0.873 | 0.7679 | 0.7778 |
| tickets_type (calibrated) | 70 | 0.7571 | 0.1418 | 0.521 | 0.1709 | 0.3647 | 0.6497 | 0.6815 | 0.700 | 0.7500 | 0.7619 |
| yelp_score | 200 | 0.6850 | 0.1897 | 0.347 | 0.2447 | 0.5071 | 1.0306 | 0.6393 | 0.836 | 0.7063 | 0.6889 |
| yelp_score (calibrated) | 200 | 0.6850 | 0.1264 | 0.433 | 0.2153 | 0.4590 | 0.8214 | 0.6314 | 0.664 | 0.7125 | 0.6778 |

## Timing

| split | questions | states | wall (s) | ms/question |
|---|---|---|---|---|
| val | 1292 | 760 | 338 | 261 |
| test | 1520 | 928 | 416 | 274 |

## Files

- `metrics.json` — the source of truth for every number above.
- `predictions-*.jsonl` — raw masked logits per question; re-score with `uv run task eval --rescore results/<run_id>` without a GPU.
- `calibration.json` — fitted temperatures, tied to the quantization above.
- `reliability-*.png`, `risk-coverage.png` — drawn from `metrics.json`.
