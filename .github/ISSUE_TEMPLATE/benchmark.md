---
name: Benchmark result
about: Report `uv run task bench` on hardware we do not have
title: "bench: <GPU> <quantization>"
labels: benchmark
---

Thank you — measurements on cards we cannot test are the most useful thing you can send.

**Please attach `results/<run_id>/latency.json` rather than pasting numbers.** It is
self-describing: it already carries the GPU, the torch and CUDA versions, the kernel
configuration, the quantization, the prefix length and the fitted cost model. A PR adding the
run directory under `results/` is even better than an issue.

## Hardware

- **GPU** (name and VRAM):
- **Driver / CUDA**:
- **OS**:
- **Other GPU processes during the run** (`uv run task doctor` prints this):

## Command

```
# e.g. uv run task bench --counts 1,4,16,64
```

- **Commit hash** (`git rev-parse HEAD`):
- **`uv run task doctor` output** — at least the `kernels`, `gpu`, and `sysmem-fallback` lines:

```
paste here
```

## Results

Attach or paste `results/<run_id>/latency.json`.

<details>
<summary>latency.json</summary>

```json
paste here
```

</details>

## Anything surprising?

Slower than expected, an OOM, a kernel that would not compile, a target that failed — those are
worth reporting too. A run that did **not** work tells us more than one that did.
