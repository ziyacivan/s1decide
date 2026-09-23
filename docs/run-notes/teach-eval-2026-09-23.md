# Teacher run 2 — `Score` evaluation rows — launched 2026-09-23 11:44 UTC

**Status commands:**

```
uv run task teach --status teach-qwen-low-1024-eval
uv run task teach --status teach-gptoss-medium-1024-eval
```

## What is running

The queued run from `docs/phase-1-plan.md` ("teacher run 2"): 600 `val` and 600 `test` states,
drawn from those partitions and never from `train`, through the **same two teachers, rubric,
1,024-token cap, agreement filter and soft-target rule** as the training batch. It closes the
gap where `Score` has zero genuine rows in `val` and `test`.

```
uv run task teach-overnight --detach --suffix eval \
    --items data/processed/teach_items_val.jsonl data/processed/teach_items_test.jsonl
```

Detached chain, pid 22036. One chain over both files: the ids carry their split
(`teach-val-…`, `teach-test-…`), the chain refuses colliding ids, and the fold can separate them.

| | leg 1 | leg 2 |
|---|---|---|
| run id | `teach-qwen-low-1024-eval` | `teach-gptoss-medium-1024-eval` |
| model | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | `openai/gpt-oss-20b` |
| batch | 4 | 16 |
| rows | 1,200 | 1,200 |
| rate measured on run 1 | 0.0596 rows/s | 0.67 rows/s |
| estimate | ~5.6 h | ~0.5 h |

**ETA ≈ 6.2 h from launch → about 18:00 UTC (21:00 local) on 2026-09-23.**

The `-eval` suffix is new (`plan_with_suffix`): run ids are resume keys, and reusing the
training batch's ids would either be refused by the items-digest check or put evaluation labels
in the directories the training rows were folded from.

## Pre-flight

| check | result |
|---|---|
| Unsloth Studio / llama-server | not running |
| foreign VRAM | 853 MiB (desktop compositor, browsers) — `doctor` OK |
| sysmem fallback | off — `doctor` raised OOM as it should |
| pending reboot | none (`RebootRequired`, `RebootPending` both absent) |
| **Windows Update paused** | **no** — `docs/windows-setup.md` §3bb requires it; it is a Settings-UI action on the owner's machine and was not changed from here. The runner checkpoints every 50 rows and resumes without recomputing, so a reboot costs at most ~50 rows. |
| disk | 518 GiB free |

## Watchers

Armed for: a state change on either leg, `FAILED`, a heartbeat older than 15 minutes, and
`REBOOTED MID-RUN` from `--status`.

## Progress at the first checkpoint

```
run teach-qwen-low-1024-eval: RUNNING
  rows      60 / 1,200 (5.0%)
  remaining 1,140
  rate      0.06 rows/s
  eta       5.2 h
  heartbeat 43s ago
  booted    2026-09-18 04:59 UTC (127.0 h ago)
  teacher   qwen-low-1024
  truncated 0
```

Rate matches run 1 (0.06 rows/s), nothing truncated.

## Outcome — DONE 2026-09-23 18:45 UTC, after two failed starts of leg 2

| | |
|---|---|
| leg 1, `teach-qwen-low-1024-eval` | 1,200 / 1,200, 1,197 committed, 3 uncommitted, truncation 0.25%, finished 17:09 UTC |
| leg 2, `teach-gptoss-medium-1024-eval` | 1,200 / 1,200, finished 18:45 UTC |
| fold | `results/teach-qwen-low-1024-eval-fold/fold.json` — **959 kept** of 1,193 compared (80.4%), exact 49.9%, κ 0.337, weighted κ 0.632, 0 unlicensed |
| into the corpus | `data/processed/score_teacher_eval.jsonl`; the state-hash split placed **468 in val and 491 in test**, none in train (the build raises if one does) |

Agreement matches the training batch (81.0% kept, κ 0.344, weighted κ 0.650), so the eval rows
describe the same labelling process as the training rows. The 300-row floor per primitive per
evaluation split now holds; `test_every_primitive_has_a_real_evaluation_set` was a strict xfail,
failed on the unexpected pass as designed, and the marker was removed.

### Leg 2 failed twice before it ran — the 09-22 fix had not held

**Attempt 1 (in-chain, 17:09).** Leg 2 died loading gpt-oss: `CUDA out of memory … 19.46 GiB is
allocated by PyTorch` — the same error as 09-22 that commit `4cd5edc` was written to fix. That
fix called `model.to("meta")` inside `suppress(Exception)`; a bitsandbytes 4-bit model refuses
`.to()`, the suppress hid it, and `run()`'s own locals still held the 27B when the cache was
emptied. Nothing was freed.

**Attempt 2 (relaunch, 17:29).** `run()` loaded the 27B again for leg 1 although all 1,200 rows
were done, then OOMed leg 2 identically. It also rewrote leg 1's `summary.json` and `meta.json`
with a zero elapsed time; the rows were untouched.

**Fixed in `191c7fc`:** the model lives only in a dict emptied into the release call; a module
that refuses `.to()` has its parameter storage dropped directly (a GPU test reproduces the
refusal with a caller reference alive); `run()` returns without loading when nothing is pending.
Leg 1's summary was rebuilt from `rows.jsonl` and `progress.jsonl` (`rebuilt_from` marks it).
Its 34.5 tokens/s counts only rows labelled in the timed invocation; run 1's 44.5 divided all
6,000 rows' tokens by the last invocation's time, which overstates it.

**Attempt 3 (17:33)** skipped leg 1 without loading and ran leg 2 in a fresh process.

### Leg 2 slowed 4x from 17:48 UTC

Per-batch time went from ~20 s to ~75 s at row ~530 and stayed there to the end. Not the data
(trace length flat at ~225 tokens throughout, and the step is not at the val/test boundary at
row 600), not throttling (1,965 MHz, 56 °C), not Discord (closing it changed nothing). The card
was at 24.2 of 24 GiB from that point. The hardware profile already records the cause on this
card (ADR 0003): above ~22 GiB it slows with no error — 22.34 GiB ran 2.8x slower, 22.78 GiB
6.4x — so this is the documented memory cliff rather than something new. (First guess in this
note was the caching allocator; the measured cliff is the better explanation.) Speed does
not change the labels. `summary.json` records the overall rate.

### An environment change during the run

Leg 1 ran with `.venv` torchvision from PyPI; at 11:54 UTC an early status watcher of mine
called plain `uv run`, which synced the relocked `torchvision 0.25.0+cu130` into the environment
while leg 1 was running. Leg 1 had already imported torchvision and was unaffected; leg 2 ran
entirely on the cu130 build. torchvision is not on the text-generation path. Every watcher since
uses `uv run --no-sync`.
