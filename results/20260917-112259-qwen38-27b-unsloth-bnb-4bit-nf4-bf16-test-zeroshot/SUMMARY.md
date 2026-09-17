# 20260917-112259-qwen38-27b-unsloth-bnb-4bit-nf4-bf16-test-zeroshot

<!-- GENERATED from metrics.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · quantization `nf4-bf16` · engine `hf`
- **Data**: `pngwn/system-one-decisions` (CC-BY-NC-4.0), report split `test`, calibration fitted on `val`
- **Prompt format**: `0.1` · template `qwen3.8-chatml-nothink`
- **Commit**: `1e7cd2584a13` **(tree dirty at run time)**
- **Created**: 2026-09-17T11:36:22+00:00 · Windows 10 · Python 3.11.16

## Coverage

| split | kept | total | fraction | dropped >26 options | dropped invalid |
|---|---|---|---|---|---|
| val | 1292 | 1452 | 89.0% | 160 (banking77 100, tickets_queue 60) | 0 |
| test | 1520 | 1751 | 86.8% | 230 (banking77 150, tickets_queue 80) | 1 |

## Headline

| row | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf |
|---|---|---|---|---|---|---|---|---|---|
| model, uncalibrated | 1520 | 0.7671 | 0.1274 | 0.289 | 0.1746 | 0.3688 | 0.7802 | 0.7612 | 0.890 |
| model, calibrated | 1520 | 0.7671 | 0.0375 | 0.131 | 0.1535 | 0.3331 | 0.5848 | 0.7598 | 0.749 |
| control: uniform | 1520 | 0.4138 | 0.0767 | 0.134 | 0.2126 | 0.6406 | 1.0986 | 0.7058 | 0.359 |
| control: base_rate | 1520 | 0.4296 | 0.0342 | 0.111 | 0.2149 | 0.6236 | 1.0700 | 0.6987 | 0.454 |

Temperature scaling cannot change which option wins, so accuracy is identical in the first two rows by construction. Read the controls before the model: a predictor that ignores the state can post a competitive ECE, which is why ECE never appears here without accuracy, Brier and AUROC beside it.

## Calibration

| bucket | n | temperature | NLL before | NLL after |
|---|---|---|---|---|
| 2 | 695 | 3.3966 | 0.8365 | 0.4934 |
| 3-5 | 597 | 2.0116 | 0.9110 | 0.7466 |

## By option-count bucket

| bucket | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 721 | 0.7975 | 0.1313 | 0.307 | 0.1691 | 0.3383 | 0.7080 | 0.7076 | 0.929 |
| 2 (calibrated) | 721 | 0.7975 | 0.0535 | 0.095 | 0.1485 | 0.2971 | 0.4637 | 0.7081 | 0.766 |
| 3-5 | 799 | 0.7397 | 0.1350 | 0.292 | 0.1795 | 0.3963 | 0.8453 | 0.7819 | 0.856 |
| 3-5 (calibrated) | 799 | 0.7397 | 0.0441 | 0.202 | 0.1580 | 0.3657 | 0.6940 | 0.7828 | 0.733 |

## By primitive

| primitive | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf |
|---|---|---|---|---|---|---|---|---|---|
| choice | 519 | 0.8266 | 0.0821 | 0.269 | 0.1237 | 0.2686 | 0.5915 | 0.8223 | 0.891 |
| choice (calibrated) | 519 | 0.8266 | 0.0471 | 0.202 | 0.1182 | 0.2602 | 0.5079 | 0.8220 | 0.788 |
| noul | 721 | 0.7975 | 0.1313 | 0.307 | 0.1691 | 0.3383 | 0.7080 | 0.7076 | 0.929 |
| noul (calibrated) | 721 | 0.7975 | 0.0535 | 0.095 | 0.1485 | 0.2971 | 0.4637 | 0.7081 | 0.766 |
| score | 280 | 0.5786 | 0.2397 | 0.474 | 0.2829 | 0.6329 | 1.3157 | 0.6594 | 0.791 |
| score (calibrated) | 280 | 0.5786 | 0.1131 | 0.243 | 0.2317 | 0.5612 | 1.0389 | 0.6590 | 0.631 |

## By family

| family | n | acc | ECE | MCE | Brier(top) | Brier(mc) | NLL | AUROC | conf |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | 200 | 0.8650 | 0.0827 | 0.275 | 0.1016 | 0.2104 | 0.5571 | 0.8632 | 0.948 |
| ag_news (calibrated) | 200 | 0.8650 | 0.0435 | 0.149 | 0.0883 | 0.1911 | 0.3824 | 0.8651 | 0.892 |
| go_emotions | 641 | 0.7832 | 0.1403 | 0.313 | 0.1793 | 0.3587 | 0.7249 | 0.7012 | 0.920 |
| go_emotions (calibrated) | 641 | 0.7832 | 0.0520 | 0.127 | 0.1566 | 0.3133 | 0.4832 | 0.7018 | 0.753 |
| mmlu | 249 | 0.8112 | 0.0730 | 0.239 | 0.1259 | 0.2849 | 0.6002 | 0.8335 | 0.861 |
| mmlu (calibrated) | 249 | 0.8112 | 0.0875 | 0.202 | 0.1270 | 0.2866 | 0.5648 | 0.8314 | 0.740 |
| tickets_language | 80 | 0.9125 | 0.0901 | 0.331 | 0.0874 | 0.1747 | 0.5724 | 0.5705 | 0.996 |
| tickets_language (calibrated) | 80 | 0.9125 | 0.1324 | 0.231 | 0.0836 | 0.1672 | 0.3075 | 0.5714 | 0.871 |
| tickets_priority | 80 | 0.3250 | 0.3493 | 0.908 | 0.3560 | 0.9082 | 1.8705 | 0.5299 | 0.665 |
| tickets_priority (calibrated) | 80 | 0.3250 | 0.2444 | 0.665 | 0.2621 | 0.7868 | 1.4966 | 0.5071 | 0.471 |
| tickets_type | 70 | 0.7714 | 0.1459 | 0.395 | 0.1786 | 0.3770 | 0.6586 | 0.6979 | 0.832 |
| tickets_type (calibrated) | 70 | 0.7714 | 0.1328 | 0.554 | 0.1725 | 0.3637 | 0.6643 | 0.7118 | 0.665 |
| yelp_score | 200 | 0.6800 | 0.2143 | 0.532 | 0.2537 | 0.5228 | 1.0938 | 0.6420 | 0.841 |
| yelp_score (calibrated) | 200 | 0.6800 | 0.1225 | 0.439 | 0.2195 | 0.4709 | 0.8558 | 0.6359 | 0.695 |

## Timing

| split | questions | states | wall (s) | ms/question |
|---|---|---|---|---|
| val | 1292 | 760 | 360 | 279 |
| test | 1520 | 928 | 422 | 278 |

## Files

- `metrics.json` — the source of truth for every number above.
- `predictions-*.jsonl` — raw masked logits per question; re-score with `uv run task eval --rescore results/<run_id>` without a GPU.
- `calibration.json` — fitted temperatures, tied to the quantization above.
- `reliability-*.png` — drawn from `metrics.json`.
