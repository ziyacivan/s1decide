"""Build ``data/processed/`` from the registered sources.

    uv run task data

Deterministic and idempotent: the same seed produces byte-identical files, so a rebuild is a
no-op and a diff means something really changed.

The stages are fetch -> normalise -> expand -> augment -> split -> gate -> write.

* **expand** turns a question with more options than there are single-token labels into the
  two-stage form from ADR 0001 and format 0.2: stage-1 rows (one yes/no per candidate, with the
  candidate embedded in the instruction so the ordinary Noul renderer reproduces
  `render_stage1`'s bytes) and one stage-2 row (a Choice over a shortlist containing the answer).
* **augment** shuffles option order per example, so no answer can be learned from position.
* **split** partitions by **state hash**, never by row, so paraphrases of one state cannot
  straddle train and test.
* **gate** is ADR 0004: the build fails if any training row carries a non-allowlisted licence.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.build.licences import assert_train_splits_are_licensed, summarise_licences
from data.build.sources import SOURCES, Source, load_source, row_id
from data.build.synthetic import generate_ordinal_control

from s1decide.tokens import MAX_SINGLE_TOKEN_OPTIONS

__all__ = ["BuildConfig", "build", "expand_high_cardinality", "shuffle_options", "split_by_state"]

#: Negative candidates emitted per high-cardinality example in stage 1, alongside the positive.
STAGE1_NEGATIVES = 3

#: Size of the stage-2 shortlist, including the true answer.
STAGE2_SHORTLIST = 8


@dataclass(frozen=True)
class BuildConfig:
    """Options for one build.

    Attributes:
        seed: Everything random derives from this.
        val_fraction: Share of train-role *states* held out for validation.
        test_fraction: Share held out for test.
        out_dir: Destination, normally ``data/processed``.
        limit_per_source: Cap for smoke builds; ``None`` uses each source's own cap.
    """

    seed: int = 20260917
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    out_dir: Path | None = None
    limit_per_source: int | None = None


def state_hash(state: str) -> str:
    """Stable hash of a state, used for splitting and for the leakage guard."""
    return hashlib.blake2b(state.strip().encode("utf-8"), digest_size=16).hexdigest()


def shuffle_options(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Shuffle a question's options, moving ``answer_idx`` with them.

    Noul is left alone: its labels are fixed as ``("no", "yes")`` so that index 1 always means
    true, and shuffling them would break that contract.

    Args:
        row: A normalised question row.
        rng: Seeded source of randomness.

    Returns:
        A new row with options permuted.
    """
    if row["qtype"] == "noul" or len(row["options"]) < 2:
        return row
    order = list(range(len(row["options"])))
    rng.shuffle(order)
    return {
        **row,
        "options": tuple(row["options"][i] for i in order),
        "answer_idx": order.index(row["answer_idx"]),
    }


def expand_high_cardinality(row: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    """Turn one over-wide Choice into stage-1 and stage-2 rows (ADR 0001).

    Stage-1 rows are Noul questions whose instruction already carries the candidate, so the
    ordinary renderer reproduces :func:`s1decide.prompt.render_stage1`'s output byte for byte.
    Stage 2 is a Choice over a shortlist that always contains the true answer.

    Args:
        row: A ``choice`` row with more options than :data:`MAX_SINGLE_TOKEN_OPTIONS`.
        rng: Seeded source of randomness.

    Returns:
        The rows to emit in place of ``row``.
    """
    options: Sequence[str] = row["options"]
    answer = row["answer_idx"]
    parent = row["id"]
    others = [i for i in range(len(options)) if i != answer]

    out: list[dict[str, Any]] = []
    for index in [answer, *rng.sample(others, k=min(STAGE1_NEGATIVES, len(others)))]:
        out.append(
            {
                **row,
                "id": f"{parent}#s1-{index}",
                "qtype": "noul",
                "instructions": f"{row['instructions']}\nCandidate: {options[index]}",
                "options": ("no", "yes"),
                "answer_idx": 1 if index == answer else 0,
                "stage": 1,
                "parent_id": parent,
            }
        )

    shortlist = [answer, *rng.sample(others, k=min(STAGE2_SHORTLIST - 1, len(others)))]
    rng.shuffle(shortlist)
    out.append(
        {
            **row,
            "id": f"{parent}#s2",
            "qtype": "choice",
            "options": tuple(options[i] for i in shortlist),
            "answer_idx": shortlist.index(answer),
            "stage": 2,
            "parent_id": parent,
        }
    )
    return out


def split_by_state(rows: Sequence[dict[str, Any]], config: BuildConfig) -> list[dict[str, Any]]:
    """Assign a split to every row, partitioning by state hash.

    Rows from `heldout` and `ood` sources go entirely to ``eval``: a held-out family is only
    held out if none of it is trained on. Train-role rows are partitioned by hashing the state,
    so every question about one state lands in the same split.

    Args:
        rows: Normalised rows carrying ``role`` and ``state``.
        config: Fractions and seed.

    Returns:
        The rows, each with ``split`` set.
    """
    out = []
    for row in rows:
        if row["role"] != "train":
            out.append({**row, "split": "eval"})
            continue
        # A hash of (seed, state) gives a stable, uniform assignment that does not depend on
        # iteration order and survives adding or removing sources.
        digest = hashlib.blake2b(
            f"{config.seed}\x1f{row['state_hash']}".encode(), digest_size=8
        ).digest()
        draw = int.from_bytes(digest, "big") / 2**64
        if draw < config.test_fraction:
            split = "test"
        elif draw < config.test_fraction + config.val_fraction:
            split = "val"
        else:
            split = "train"
        out.append({**row, "split": split})
    return out


def _normalised(source: Source, raw: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw):
        state = str(item["state"]).strip()
        instructions = str(item["instructions"]).strip()
        options = tuple(str(o).strip() for o in item["options"])
        if not state or not instructions or len(options) < 2:
            continue
        if len({o.casefold() for o in options}) != len(options):
            continue
        rows.append(
            {
                "id": row_id(source.family, index, state, instructions),
                "family": source.family,
                "state": state,
                "state_hash": state_hash(state),
                "qtype": source.primitive if source.primitive != "choice" else "choice",
                "instructions": instructions,
                "options": options,
                "answer_idx": int(item["answer_idx"]),
                "source": source.canonical_repo or source.repo,
                "loaded_from": source.repo,
                "license": source.license,
                "role": source.role,
                "stage": None,
                "parent_id": None,
            }
        )
    return rows


def build(config: BuildConfig | None = None) -> dict[str, Any]:
    """Run the whole pipeline and write ``data/processed/``.

    Returns:
        A manifest: per-source and per-split counts, licence summary, and the leakage report.

    Raises:
        LicenceError: If any training row is not permissively licensed (ADR 0004).
    """
    from s1decide.tasks import repo_root

    cfg = config or BuildConfig()
    out_dir = cfg.out_dir or (repo_root() / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)

    rows: list[dict[str, Any]] = []
    per_source: dict[str, Any] = {}
    for source in SOURCES:
        loaded = load_source(source)
        if cfg.limit_per_source is not None:
            loaded = loaded[: cfg.limit_per_source]
        normalised = _normalised(source, loaded)
        per_source[source.family] = {
            "repo": source.repo,
            "canonical_repo": source.canonical_repo or source.repo,
            "license": source.license,
            "role": source.role,
            "primitive": source.primitive,
            "rows_loaded": len(loaded),
            "rows_kept": len(normalised),
            "note": source.note,
        }
        rows.extend(normalised)

    synthetic = list(generate_ordinal_control(seed=cfg.seed))
    if cfg.limit_per_source is not None:
        synthetic = synthetic[: cfg.limit_per_source]
    synthetic_source = Source(
        family="ordinal_control",
        repo="s1decide/ordinal_control",
        license="apache-2.0",
        primitive="score",
        role="train",
        note="Rule-labelled ordinal control set, generated here (ADR 0005 rule c). No model in "
        "the loop, so its labels are exact.",
    )
    synthetic_rows = _normalised(synthetic_source, synthetic)
    per_source["ordinal_control"] = {
        "repo": synthetic_source.repo,
        "canonical_repo": synthetic_source.repo,
        "license": synthetic_source.license,
        "role": "train",
        "primitive": "score",
        "rows_loaded": len(synthetic),
        "rows_kept": len(synthetic_rows),
        "note": synthetic_source.note,
    }
    rows.extend(synthetic_rows)

    expanded: list[dict[str, Any]] = []
    two_stage_parents = 0
    for row in rows:
        if row["qtype"] == "choice" and len(row["options"]) > MAX_SINGLE_TOKEN_OPTIONS:
            expanded.extend(expand_high_cardinality(row, rng))
            two_stage_parents += 1
        else:
            expanded.append(row)

    augmented = [shuffle_options(row, rng) for row in expanded]
    assigned = split_by_state(augmented, cfg)

    # ADR 0004. Before anything is written, not after.
    assert_train_splits_are_licensed(assigned)

    counts: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for row in assigned:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        by_family.setdefault(row["family"], {})
        by_family[row["family"]][row["split"]] = by_family[row["family"]].get(row["split"], 0) + 1

    written = {}
    for split in sorted(counts):
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in assigned:
                if row["split"] != split:
                    continue
                handle.write(
                    json.dumps(
                        {
                            k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in row.items()
                            if k != "role"
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        # out_dir is normally inside the repo but need not be (tests build into tmp_path),
        # so fall back to the absolute path rather than raising.
        try:
            written[split] = str(path.relative_to(repo_root()))
        except ValueError:
            written[split] = str(path)

    manifest = {
        "seed": cfg.seed,
        "rows": len(assigned),
        "splits": counts,
        "by_family": {k: by_family[k] for k in sorted(by_family)},
        "by_primitive": primitive_mix(assigned),
        "sources": per_source,
        "two_stage_parents_expanded": two_stage_parents,
        "licences": summarise_licences(assigned),
        "files": written,
        "leakage": leakage_report(assigned),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


#: Minimum share of the training mix that must be **genuine** `Noul` — statements about the
#: state that are true or false — as opposed to stage-1 option-membership rows, which are
#: `noul`-shaped but are a different task. 15% is a floor, not a target: below it the primitive
#: is being carried by incidental data rather than trained deliberately. Asserted against the
#: built manifest by `tests/test_noul_mix.py`, in the same spirit as the stage-1 no:yes ratio.
#:
#: The floor exists because the NLI sources that would normally supply `Noul` — SNLI, BoolQ,
#: FEVER, MultiNLI — are all share-alike and eval-only under ADR 0005. Without it the corpus
#: silently drifted to 9% genuine `Noul`, all of it from one source, while a `qtype` count
#: cheerfully reported 79%.
MIN_GENUINE_NOUL_SHARE = 0.15

#: Genuine `Noul` must come from at least this many families. One source carrying a whole
#: primitive means its quirks are indistinguishable from the primitive's behaviour.
MIN_GENUINE_NOUL_FAMILIES = 2


def primitive_mix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rows per primitive and split, with stage-1 rows counted separately.

    A plain ``qtype`` count is misleading here, and was. Expanding a 60- or 151-option question
    produces a stage-1 row per candidate — "is `card arrival` the intent of this message?" —
    which is a ``noul`` by shape and an option-membership question by nature. Counting them
    together made `Noul` look like 79% of training when the primitive we actually publish was
    9%, all of it from a single source.

    So every count here is split into ``stage1`` and ``genuine``, and the share that matters —
    the one the minimum-Noul test asserts on — is the genuine one.

    Args:
        rows: Every assigned row.

    Returns:
        ``{split: {qtype: {"total", "stage1", "genuine", "genuine_share"}}}`` plus a
        ``genuine_by_family`` breakdown, so a primitive carried by one source is visible.
    """
    out: dict[str, Any] = {}
    for split in sorted({row["split"] for row in rows}):
        in_split = [row for row in rows if row["split"] == split]
        per_qtype: dict[str, Any] = {}
        for qtype in sorted({row["qtype"] for row in in_split}):
            same = [row for row in in_split if row["qtype"] == qtype]
            stage1 = [row for row in same if row.get("stage") == 1]
            genuine = [row for row in same if row.get("stage") != 1]
            families: dict[str, int] = {}
            for row in genuine:
                families[row["family"]] = families.get(row["family"], 0) + 1
            per_qtype[qtype] = {
                "total": len(same),
                "stage1": len(stage1),
                "genuine": len(genuine),
                "genuine_share": len(genuine) / len(in_split) if in_split else 0.0,
                "genuine_by_family": {k: families[k] for k in sorted(families)},
            }
        out[split] = per_qtype
    return out


def leakage_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check the Tier-2 external eval set against our training states (amendment B).

    ``pngwn/system-one-decisions`` is derived from banking77 and other sources we now train on,
    so overlap is possible even though the datasets differ. Comparing state hashes catches it;
    comparing dataset names would not.

    Args:
        rows: Every built row, carrying ``split`` and ``state``.

    Returns:
        How many external rows would have to be dropped, or a note if the set is unavailable.
    """
    trained = {r["state_hash"] for r in rows if r["split"] in {"train", "val"}}
    try:
        from eval.data import load_system_one_decisions

        external, _ = load_system_one_decisions("test", max_options=10**6)
    except Exception as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}

    leaked = [item for item in external if state_hash(item.state) in trained]
    return {
        "checked": True,
        "external_rows": len(external),
        "leaked_rows": len(leaked),
        "leaked_families": sorted({item.family for item in leaked}),
        "train_states": len(trained),
    }
