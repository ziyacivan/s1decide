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
