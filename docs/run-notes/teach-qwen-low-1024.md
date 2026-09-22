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

## Stopped 2026-09-18 21:15 UTC, at 1,400 rows

Stopped deliberately so the machine could be shut down, not by a failure: no `FAILED` file, no
`DONE`, 23.3% complete. Resume with the command above.

Stopping exposed a bug worth recording. `uv run task gpu-kill` reported *"no s1decide process is
holding GPU memory"* while our own run held 23,169 MiB. The detached command line is

```
…\uv\python\cpython-3.11-windows-x86_64-none\python.exe -m data.build.teach_overnight --items … --limit 6000
```

and none of `s1decide`, `task.exe` or `uv run task` appears in it: the interpreter is uv's shared
install rather than the project venv, the working directory is not part of a command line, and
`data.build.*` is not named after the project. The safety check designed to kill only our own
trees could not see our own tree — at the one moment it would be needed, after a wedge. The run
was stopped by verifying the pid against the module name and calling `kill_process_tree`
directly; `OUR_MARKERS` now includes the module entry points, with the real command line as a
regression test.

## Leg 1 finished 2026-09-22 19:41 UTC — and leg 2 died on the handover

Leg 1 completed all 6,000 rows: 76,905 s of GPU time across three sessions, 4,636 rows in the
final one and 1,364 resumed. Truncation settled at 0.79%, unchanged between 3,000 and 6,000 rows,
so the 1,024-token cap was the right size.

Leg 2 then failed immediately, loading its model:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 254.00 MiB.
GPU 0 has a total capacity of 24.00 GiB of which 0 bytes is free.
Of the allocated memory 19.45 GiB is allocated by PyTorch, and 3.76 GiB is reserved but unallocated.
```

The chain runs both legs in one process and leg 1's 27B was still resident. Two causes, and
fixing either alone would not have been enough: the `work` closure holds a reference to the
model, so it outlives `run()`'s locals and a `del` of the caller's names frees nothing; and
PyTorch's caching allocator does not hand freed blocks back to the driver by itself.

`release_teacher` now runs in a `finally` around the job, moving the model to the meta device
through the object — `Module.to()` moves parameters in place, which is what detaches the storage
while the closure still holds the reference — then collecting and emptying the cache. A
gpu-marked test asserts reserved memory actually falls; the CPU tests cover that it runs at all
and that it runs when the job raises.

**Nothing was lost.** Separate run directories per leg meant leg 1's 6,000 rows were already
checkpointed and its `DONE` written before leg 2 ever started. Leg 2 was relaunched standalone
with the same config the plan specifies (`gptoss-medium-1024`, batch 16, 6,000 rows) and is
running at 0.77 rows/s, faster than the pilot's 0.628.

The failure mode is worth naming: it costs nothing when it happens on the *first* leg and a whole
night when it happens on the second, because it fails after the expensive work is done.
