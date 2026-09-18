# Overnight teacher run — launched 2026-09-17 23:39 UTC

**Status command (this is the one to run in the morning):**

```
uv run task teach --status teach-qwen-low-1024
```

When that says `DONE`, the second leg is already running:

```
uv run task teach --status teach-gptoss-medium-1024
```

## What is running

A detached chain (`uv run task teach-overnight --detach --limit 6000`, pid 22976) running both
teachers back to back over the same 6,000 source rows. Each leg keeps its own run directory,
checkpoints and resume state, so a failure in the second cannot cost the first.

| | leg 1 | leg 2 |
|---|---|---|
| run id | `teach-qwen-low-1024` | `teach-gptoss-medium-1024` |
| model | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | `openai/gpt-oss-20b` |
| quantization | bitsandbytes nf4, bf16 compute | MXFP4 |
| reasoning | native effort `low` | native effort `medium` |
| generation cap | 1,024 tokens | 1,024 tokens |
| batch | 4 | 16 |
| measured rate | 0.060 rows/s | 0.63 rows/s |
| estimated wall time | **~27.6 h** | ~2.6 h |

Both legs label the same `data/processed/teach_items.jsonl` rows, with the same rubric and the
same prompt, so the agreement rate measures the teachers and not the prompt.

## Runtime path: bitsandbytes, not llama.cpp — and why

The time-boxed experiment ran teacher 1 through `llama-server` with Q4_K_M and continuous
batching at `--parallel 16`, on the same 100 pilot rows.

| | bitsandbytes nf4 | llama.cpp Q4_K_M |
|---|---|---|
| rows/s | 0.060 | **0.312** (5.18x) |
| tokens/s | 34.4 | 165.2 |
| committed | 100/100 | 100/100 |
| hit the cap | 0% | 0% |
| **exact match vs nf4** | — | **78.0%** |
| within ±1 level | — | 94.0% |

The rule required **both** a 2.5x speed-up and 90% exact match. Speed passed at 5.18x; exact
match failed at 78%. So the speed-up is declined and teacher 1 runs on bitsandbytes, at the cost
of roughly a day of wall time.

That is the right call beyond rule-following. Two quantizations of one model disagreeing on a
fifth of an ordinal task is not a runtime detail — the disagreement is concentrated on adjacent
levels, exactly where the scale is hardest and where the ±1 keep-rule is already doing the most
work. Adopting Q4_K_M would have changed what the teacher said while the dataset card went on
naming one model.

The client survives at `src/s1decide/engine/llamacpp_client.py` as the seed of the GGUF engine.

## Progress at the time of writing

```
run teach-qwen-low-1024: RUNNING
  rows      208 / 6,000 (3.5%)
  remaining 5,792
  rate      0.06 rows/s
  eta       26.2 h
  heartbeat 24s ago
  teacher   qwen-low-1024
  truncated 0
```

First checkpoint written (200 rows in `rows.jsonl`). ETA matches the pilot's 0.060 rows/s, and
nothing has truncated.

## Interrupted by a Windows Update reboot, 2026-09-18 04:58 UTC

The run was killed 13 hours in by an unattended Windows Update restart.

| | |
|---|---|
| Launched | 2026-09-17 23:39 UTC |
| Restart initiated | **2026-09-18 04:58:03 UTC** — `MoUsoCoreWorker.exe`, "Service pack (Planned)", then `TrustedInstaller.exe`, "Upgrade (Planned)" at 04:59:04 |
| Boot completed | 2026-09-18 04:59:30 UTC |
| Ran for | 5 h 19 min |
| Last progress line | 04:57:11 UTC — 1,144 rows |
| Last checkpoint | **1,000 rows** |
| **Rows lost** | **144** (~40 min of work) |
| `DONE` / `FAILED` | neither written — the process died with the OS, so nothing caught anything |
| Leg 2 | never started |
| Resumed | **not yet — see `PREFLIGHT-FAILED.md`** |

Resume was blocked at pre-flight: Unsloth Studio and its `llama-server` were holding 21.2 GB of
the card, both started by the owner minutes earlier. Nothing of theirs was killed and nothing
was launched over the top of it.

Two things came out of this, both done:

- **Windows Update pausing is now a required setting** in `docs/windows-setup.md`, with the
  event-log evidence. Checkpointing made the reboot survivable; it is not what prevents it.
- **`--status` prints machine boot time** and flags `<-- REBOOTED MID-RUN`. A job killed by a
  reboot and a job wedged on a hung kernel both show a stale heartbeat and nothing else, and
  they need different responses — one resumes, the other needs investigating first.

## If something has gone wrong

- **`FAILED`** — the traceback is in `results/teach-qwen-low-1024/FAILED` and the job has
  stopped; it never retries in a loop. Re-running the same command resumes from the last
  checkpoint and recomputes nothing.
- **`STALE`** — the heartbeat is older than 15 minutes. The process is wedged rather than slow.
  `uv run task gpu-kill` clears our own trees, then re-run to resume.
- **Nothing on the GPU** — the machine rebooted. Resume with
  `uv run task teach-overnight --detach --limit 6000`; finished rows are never recomputed.

## Pre-flight, recorded

- Unsloth Studio: not running.
- Foreign VRAM holders: none touched. 490 MiB held by the desktop compositor, Chrome and VS
  Code, all left alone; `gpu-kill` only ever targets our own process trees.
- Sysmem fallback: confirmed off by `doctor` — a 24.70 GiB request raised `OutOfMemoryError`.
- Kernels: `gdn 2/4 accelerated | fla: chunk, fused_recurrent | torch: conv:fn, conv:update`,
  recorded in the run's `meta.kernels`.
- Standby and monitor timeout: disabled by the owner.
- `doctor`: 17 checks, 0 fail, 1 warn (the optional `llama-cpp-python`, unrelated).

## What happens after

The two legs produce per-teacher labels. The corpus step that keeps only exact and ±1 agreement,
records the drop rate and folds the result into `Score` has **not** run yet and is not part of
this job — it is the next piece of work, and it is CPU-side.

## Pausing for days, then resuming

Stopping is safe at any moment and resuming days later costs nothing but the rows generated
since the last checkpoint — at `CHECKPOINT_EVERY = 50` that is at most ~14 minutes. Resume
matches by row id read from `rows.jsonl`, so it depends on nothing in memory and nothing about
the process that wrote it.

```
uv run task gpu-kill                                   # our own process trees only
uv run task teach-overnight --detach --limit 6000      # days later; continues where it stopped
uv run task teach --status teach-qwen-low-1024
```

What must not change while it is paused — because resume matches by id, it cannot notice any of
it on its own:

- **`data/processed/teach_items.jsonl`.** It is not tracked by git, and the ids embed a position
  (`teach-00042-9f3c1ab2`). Regenerating it after any upstream change shifts every id from the
  first altered row onward; a resume would then re-label thousands of rows and write a corpus
  with two numbering schemes. Backed up to `~/Documents/teach_items.backup.jsonl`,
  sha256 `3548d7c86b04910d…`, 1,820,033 bytes, 7,328 rows.
- **The environment.** A `uv sync` that moves `transformers` or `fla` changes the kernels, and
  a kernel change is a new result row rather than a silent upgrade.
- **Batch size and teacher setting.**

Since 2026-09-18 these are enforced rather than trusted: `meta.json` records the configuration
and a digest of the exact items the run was asked to label, and `check_resume_meta` refuses a
resume whose configuration differs, before the model is loaded. `--force` overrides it and says
what it overrode. The digest for this run is `1c8ca0b707de26eb…` over 6,000 items; it was
written into the existing `meta.json` mid-run so that the *first* resume is already covered, not
only the second.

A field the earlier run never recorded is deliberately not a mismatch — otherwise shipping the
guard onto a job that was already 22% finished would have refused the very run it protects.
