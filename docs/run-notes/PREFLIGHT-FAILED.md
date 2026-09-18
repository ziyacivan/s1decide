# Pre-flight failed 2026-09-18 15:25 UTC — the run was NOT resumed

`teach-qwen-low-1024` is still stopped at its last checkpoint. Nothing was launched, and
**nothing of yours was killed.**

## Why

The GPU is in use — by you, right now.

| pid | process | VRAM | started | whose |
|---|---|---|---|---|
| 12264 | `llama-server.exe` from `C:\Users\yusuf\.unsloth\llama.cpp\build\bin\Release\` | **21,212 MiB** | 18:20 local, 4 min before the check | **yours** |
| 19380 | `unsloth-studio.exe` | — | 18:11 local, 13 min before the check | **yours** |

That `llama-server` is Unsloth Studio's own binary, serving
`orcarouter/Qwen3.8-27B-Uncensored-GGUF` — a model this project has never downloaded or used.
It is not the one I started for last night's llama.cpp experiment; that one lived in `%TEMP%`
and was stopped when the experiment ended, and this machine has rebooted since.

22,292 MiB of 24,576 are taken. The teacher run needs about 20,000 MiB. Launching would have
OOMed within seconds, or — worse, if it had squeezed in — fought your session for VRAM for the
next 22 hours.

Two of your instructions point the same way here: *"use gpu-kill only on our own process trees,
never on anything else"*, and *"if pre-flight fails for a reason you cannot fix safely, do NOT
launch"*. Closing Studio and killing a model server you loaded minutes ago is not something I
can do safely on your behalf, so I stopped.

`uv run task gpu-kill` was run and correctly touched nothing — it reported both as
`(not ours, left alone)`.

## What the reboot cost

| | |
|---|---|
| Windows Update restart initiated | **2026-09-18 04:58:03 UTC** (`TrustedInstaller.exe`, "Operating System: Upgrade (Planned)") |
| Boot completed | 2026-09-18 04:59:30 UTC |
| Last progress line | 04:57:11 UTC, 1,144 rows |
| Last checkpoint on disk | **1,000 rows** |
| **Rows lost** | **144** (~40 min at 0.060 rows/s) |
| `DONE` / `FAILED` | neither — the process was killed with the OS, so nothing caught anything |

Leg 2 (`teach-gptoss-medium-1024`) never started; its directory does not exist.

## To resume

Free the GPU — close Unsloth Studio and its `llama-server`, or wait until you are done with
them — then:

```
uv run task doctor                                   # expect gpu-processes OK, sysmem-fallback OK
uv run task teach-overnight --detach --limit 6000
uv run task teach --status teach-qwen-low-1024
```

Resuming re-reads `rows.jsonl` and skips the 1,000 finished rows before the model is even
loaded, so the 144 lost rows are the entire cost. Remaining work from 1,000: **5,000 rows,
about 23 hours**, plus ~2.6 h for leg 2.

I can do this myself the moment the GPU is free — I just will not do it over the top of your
session.

## Already fixed

- **`--status` now prints machine boot time** and flags `<-- REBOOTED MID-RUN` when the boot is
  newer than the last heartbeat. Without it a killed-by-reboot job is indistinguishable from a
  wedged one, and they need different responses.
- **`docs/windows-setup.md` now lists pausing Windows Update as required**, with the event-log
  evidence from this reboot.
