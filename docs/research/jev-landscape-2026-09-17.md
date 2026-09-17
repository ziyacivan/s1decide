# Jev / System One landscape — research snapshot (2026-09-17)

This file is the factual basis for the project. Everything here comes from public
sources gathered on 17 Sep 2026. Treat vendor numbers as vendor numbers.
When something below is contradicted by newer evidence, update this file and
note the date.

## 1. What Jev is (TypeSafe AI)

- Launched 15 Sep 2026 by TypeSafe AI (SF). Founder Diogo Almeida (ex-OpenAI,
  InstructGPT co-author). $40M seed led by DCVC.
- Not a chat model. Input: unstructured/structured `state` (text or JSON) +
  a dict of typed `questions`. Output: typed values + probability distributions
  + a confidence value, in a single parallel query. No autoregressive decode.
- Three primitives (docs.typesafe.ai):
  - `Choice` — pick one option from a caller-supplied set → `choice`, `probabilities`, `confidence`
  - `Score`  — position on an ordered rubric → `score`, `probabilities`, `confidence`
  - `Noul`   — "is this statement true?" → `noul` ∈ [0,1] (name = Bernoulli)
- All questions in one call are evaluated in parallel and in isolation against
  the same state; adding questions barely changes latency.
- Design guidance from their docs: one question = one atomic gut-check;
  decompose multi-factor judgments into separate questions and combine in code;
  keep each Score one-dimensional.
- Training method: "RLCD" (Reinforcement Learning for Calibrated Decisions).
  CEO confirmed on HN that RLCD is applied on top of a pre-trained base model.
  Architecture is unpublished ("close to the chest", paper maybe later).
- Cardinality: blog says up to 255 options via a two-stage
  score-then-choose scheme. One HN commenter claimed Choice allows only 10
  options; unresolved — verify against current docs before mirroring limits.
- Pricing: $0.042 / M input tokens, output free. Latency 70–500 ms end-to-end
  (measured from US West Coast).
- Delivery: closed weights, hosted US API, early-access waitlist only.
  No self-host, VPC or on-prem option. Their `system-one-adapter-python` repo
  (wraps ordinary LLMs into the same typed API) had 1 commit + README at launch.
- Python SDK shape (from docs):

  ```python
  response = client.system_one(
      state={"document": "I was charged twice. Please fix this ASAP."},
      questions={
          "billing": Noul(instructions="Is this ticket about billing?"),
          "tone": Choice(instructions="What is the customer's tone?",
                         criteria={"calm": None, "frustrated": None, "angry": None}),
          "urgency": Score(instructions="How urgent is this ticket?",
                           criteria=["can wait", "this week", "today"]),
      },
  )
  response.nouls["billing"].noul
  response.choices["tone"].choice
  response.scores["urgency"].score
  ```
  Criteria can carry per-option `not_for` exclusions (criticised on HN as boilerplate-heavy).

## 2. Accuracy evidence

Vendor dashboard (evals.typesafe.ai). Reference answer = average of GPT-6 Astra
and Claude Fable 5.1, so scores measure *agreement with two frontier models*,
not ground truth. Workflows were written by TypeSafe's own capabilities team.

| Workflow                     | Jev   | Best comparator      |
|------------------------------|-------|----------------------|
| Aggregate                    | 67.8% | 74.1%                |
| Security incidents           | 61.7% | 66.2% (Opus 5)       |
| Agent-trace observability    | 71.6% | 76.6%                |
| Invoice processing           | 61.8% | 79.1%                |
| Customer service             | 76.0% | 78.3%                |

Jev ties Sonnet 5 at 67.8% on its own chart. It wins cost and latency
columns by 1–2 orders of magnitude; it does not win accuracy.

Independent tests (all small):
- Every (Mike Taylor): 11 experiments, 1,709 judgments, < $0.01 total.
  Planted-defect test: Jev caught 6/7, Fable 5.1 caught 7/7. Median 0.35 s
  per passage vs 8.83 s (≈25× faster, ≈1/580 cost). Verdict: "good but not perfect".
- Good Start Labs: 6,003 rubric checks, 91.5% agreement with Fable 5.1,
  ≈$160 per M graded answers vs ≈$260 for DeepSeek V4.1 Flash (1.6× cheaper).
- Another test vs Mistral Small 4: ≈5× faster, 8.6× cheaper (not 193×/444×).
- TypeSafe employee's DSPy integration: swapping one decision step gave
  15.9% faster / 30.1% cheaper end-to-end.

**No calibration curve, ECE, Brier or reliability diagram has been published.**
This is the single most important gap and our main differentiator.

## 3. Substantive critiques (design input for us)

- "Can't hallucinate" = cannot emit an off-schema value. It can still be
  confidently wrong. Their 0% hallucination figure is by construction, not empirical.
- Per-question calibration does not imply calibration of a composite decision
  built from several questions in code.
- Speed comparisons had the LLM baseline autoregressively emitting the entire
  schema; a fair baseline emits only the values (or uses constrained decoding
  with logprobs). We must publish fair baselines.
- Question phrasing burden shifts to the developer ("Does the user want a
  human agent?" vs "...now?"). Forced boolean answers cannot abstain.
- Home Assistant demo delegated multi-intent splitting to Claude Haiku;
  Doom demo used structured game state, not pixels.
- Frontier-model framing was widely rejected; "advanced the speed/cost
  frontier for structured decisions" is the defensible claim.

## 4. Open-source replicas that already exist (as of 17 Sep 2026)

All ≤ 8B. Nobody has published a ≥ 27B decision model yet.

| Repo / model | Base | Approach | Notes |
|---|---|---|---|
| rorshopping/jev-on-a-laptop | Qwen2.5 1.5B–8B (MLX) | Prefill once, broadcast KV cache across one batch row per field, evaluate all fields in one forward pass; JSON assembled in code | No training. Has a HF Space. Also has a PyTorch/CUDA path. |
| pngwn/system-one-qwen3.5-4b-scorer | Qwen3.5-4B | Sequence-classification head over (state, question, option) triples; softmax per question | Temp scaling ECE 0.135→0.044 at same accuracy; 112 ms @4 options, 560 ms @77. Dataset: `pngwn/system-one-decisions` (12,913 train questions, 9 task families incl. banking77, ticket routing). |
| pngwn/nanodiff-350m-typed-decisions | nanoDiff 350M (masked diffusion) | Calibrated masked-slot distribution | ECE 0.065→0.036; k=4 multi-question block ECE 0.020. |
| mithalouni/system-one-open | Gemma 4 E2B (attention LoRA) / Gemma 3 270M | Three primitives, Jev-shaped API, FastAPI `/decide`, Modal deploy, MIT | Has `typesafe_eval.py`: strict common-subset head-to-head vs Opus/Sol/Jev. Zero-shot Doom worked; Mario failed. |
| shamazharikh/qwen-rlcd | Qwen3.5-0.8B | Prototype + design doc | Warns: calibration ≠ ranking; base-rate predictor can have good ECE. Notes bumpy ordinal distributions → use ordinal-aware loss or cumulative-link head; relative options ("partially", "other") are hard; shuffle option lists in training. |
| GLiClass (knowledgator) | DeBERTa / ModernBERT 0.2–0.4B | Single-pass zero-shot multi-label classifier | Mentioned on HN as the open baseline. Commercially licensed data. |

An HN user reported reproducing the interface on open models in ~2 hours.

## 5. Relevant literature

- SALSA (arXiv 2510.22691): single forward pass; map each class to one token;
  read placeholder-token logits; filter+softmax; LoRA + cross-entropy. Threshold
  tuning controls precision/recall. → our Stage-1 recipe.
- RLCR (arXiv 2507.16806, MIT CSAIL, Apr 2026 press): Brier-based calibration
  reward in RL; any bounded proper scoring rule yields accurate + calibrated
  models; ECE reduced up to 90% with no accuracy loss; beats post-hoc classifiers.
  Standard RLVR worsens calibration. → our Stage-3 recipe.
- Rewarding Doubt (arXiv 2503.02623): log-score reward, betting-game framing.
- Self-Ensemble (arXiv 2506.01951): attention-mask + positional re-encoding to
  score all choice groups in one forward pass.
- Calibration must be re-fit at deployment quantization; Q4 logits ≠ BF16 logits.

## 6. Base model: Qwen3.8-27B

- Released 14 Aug 2026, Apache 2.0, dense 27B, vision-capable, 262,144 ctx.
  Hybrid Mamba-Transformer (Gated DeltaNet + full attention), same family
  design as Qwen3.5/3.6-27B. Supports Multi-Token Prediction.
- Qwen's self-reported benchmarks beat Qwen3.6-27B and closed Qwen3.7-Plus;
  independent benchmarks pending. (There is no open "Qwen3.7-27B".)
- Q4_K_M GGUF ≈ 17 GB. Fits RTX 3090 (24 GB) with room for KV cache.
- Default `reasoning_effort=xhigh` massively over-thinks; run with reasoning
  off for decision work (we never decode anyway).
- Dense → memory-bandwidth bound for decode; irrelevant for prefill-only use.
- llama.cpp: `--spec-type draft-mtp` gave ≈72% decode speedup (not needed here).
- RISK: verify PEFT/Unsloth/TRL support for the GDN hybrid layers before
  committing to it. Fallback base: Qwen3.6-27B.

## 6b. Dev environment facts: Unsloth Studio on Windows (checked 2026-09-17)

- Unsloth is usable two ways: **Unsloth Studio** (web UI, `unsloth studio -p 8888`)
  and **Unsloth Core** (code, `pip install unsloth`). Studio is Beta and works
  on Windows, Linux, WSL and macOS; on Windows it runs **natively without WSL**.
- Studio Windows requirements: Windows 10/11 64-bit, NVIDIA driver, App
  Installer (`winget`), Git, Python 3.11 ≤ x < 3.14, a Python environment
  (uv / venv / conda). Training is supported on RTX 30/40/50 series.
- The Windows installer (`irm https://unsloth.ai/install.ps1 | iex`) creates a
  venv (Python 3.13 in observed logs) under `%USERPROFILE%\.unsloth\studio\`,
  installs a CUDA PyTorch build, enables **Windows Long Paths** (needed for
  Triton compilation), and installs CMake, Visual Studio 2022 Build Tools and a
  CUDA toolkit via winget if missing. Re-running the same command updates.
- Known issues seen in GitHub reports: PyTorch flavour mismatch (CPU build
  installed where a cu13x build was expected → training silently on CPU);
  the installer reinstalls, but **verify `torch.cuda.is_available()`** yourself.
- Unsloth Core on Windows: `pip install unsloth` works if a CUDA PyTorch is
  already installed. Docker (`unsloth/unsloth` image) and WSL2 are the
  officially "easiest" alternatives; not needed here since Studio is installed.
- Unsloth ships ready recipes/notebooks for Qwen3.5 (4B, GSPO) and Qwen3
  GRPO, gpt-oss GRPO, Gemma 4 — i.e. the Qwen3.5+ hybrid family is on their
  supported path. Confirm Qwen3.8-27B specifically before relying on it.
- Implications for this repo: no `make`, no bash; cross-platform task runner;
  no FlashAttention-2 (use SDPA); no vLLM locally (Linux/WSL2 only); Triton via
  `triton-windows`; keep HF cache on a short path without spaces; close Studio
  before GPU work (VRAM).

Sources: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements ·
https://unsloth.ai/docs/new/studio/install · https://unsloth.ai/docs/get-started/install/windows-installation ·
https://github.com/unslothai/unsloth (README) · issues #5073, #6898

## 7. What "competitive with Jev" means for this project

1. Same three primitives, same call shape, one forward pass for N questions.
2. Type safety by construction (logit masking / programmatic assembly).
3. Published calibration: ECE, Brier, reliability diagrams, per option-count,
   in-distribution and OOD, plus a base-rate negative control.
4. Head-to-head on TypeSafe's public workflow evals (strict common subset).
5. Latency vs #questions curve on RTX 3090 and one datacenter GPU.
6. Quantization table (BF16 / Q8 / Q5 / Q4): accuracy + ECE.
7. Fair LLM baselines: constrained decoding + logprobs, values only.
8. `pip install`-able library + OpenAI-compatible-ish `/decide` server.
9. Fully reproducible recipe so the community can port it to larger bases.

## 8. Source links

- https://typesafe.ai/ · https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/ · https://evals.typesafe.ai/
- https://news.ycombinator.com/item?id=49717558
- https://every.to/also-true-for-humans/mini-vibe-check-typesafe-s-jev-judged-everything-i-ve-written-in-0-7-seconds
- https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know
- https://thecherrycreeknews.com/typesafe-jev-system-one-model-claims-evals-independent-tests-cherry_creek/
- https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/
- https://actionbox.cloud/blog/typesafe-ai-jev-review/
- https://flowtivity.ai/blog/jev-typesafe-ai-decision-model/
- https://www.datacamp.com/blog/system-one-models-jev
- https://github.com/rorshopping/jev-on-a-laptop
- https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer · https://huggingface.co/datasets/pngwn/system-one-decisions
- https://github.com/mithalouni/system-one-open
- https://github.com/shamazharikh/qwen-rlcd
- https://github.com/knowledgator/gliclass
- https://arxiv.org/abs/2510.22691 (SALSA) · https://arxiv.org/abs/2507.16806 (RLCR) · https://arxiv.org/abs/2503.02623 (Rewarding Doubt)
- https://huggingface.co/Qwen/Qwen3.8-27B · https://simonwillison.net/2026/Aug/16/qwen-38-27b/
