"""The training-data licence gate (ADR 0004).

The project releases under **Apache 2.0**. Weights derived from data that forbids commercial
use cannot honestly carry that licence, so every row destined for a training split must come
from a source whose licence permits redistribution of derived work.

This module is the gate, and it is deliberately unforgiving in one direction: a licence that is
not recognised is **refused**, never assumed permissive. Getting this wrong is discovered after
the GPU-hours, after the model card, possibly after a publication — so the default has to be no.

It is enforced twice, by design:

* :func:`assert_train_splits_are_licensed` at the end of ``uv run task data``, and
* ``tests/test_licences.py`` over the committed manifest,

so the gate holds whether or not anyone re-runs the build. A comment saying "don't train on NC
data" is worth nothing at 2 a.m. three months from now; a red test is worth everything.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "EVAL_ONLY_MARKERS",
    "TRAIN_LICENCE_ALLOWLIST",
    "TRAIN_SPLITS",
    "LicenceError",
    "LicenceVerdict",
    "assert_train_splits_are_licensed",
    "classify_licence",
    "normalise_spdx",
    "summarise_licences",
]

#: Licences whose terms permit redistributing derived weights under Apache 2.0.
#: Attribution-only licences are included; their attribution is recorded in the dataset card.
TRAIN_LICENCE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
    }
)

#: Substrings that mark a licence as usable for evaluation but never for training.
#: ``NC`` forbids commercial use; ``ND`` forbids derivatives, which a fine-tuned model is.
#: ``SA`` (share-alike) is *not* here: it is not automatically eval-only, but it is not on the
#: allowlist either, so it lands in ``refused`` and needs a human decision (see ADR 0005).
EVAL_ONLY_MARKERS: tuple[str, ...] = ("-NC", "NC-", "-ND", "ND-", "NONCOMMERCIAL", "NO-DERIV")

#: Split names subject to the gate. Validation is included: a temperature fitted on data we may
#: not use is as much a derived artefact as a weight.
TRAIN_SPLITS: frozenset[str] = frozenset({"train", "val", "validation", "dev"})

LicenceVerdict = Literal["train", "eval-only", "refused"]

_SPDX_ALIASES: dict[str, str] = {
    "APACHE2": "Apache-2.0",
    "APACHE20": "Apache-2.0",
    "APACHELICENSE20": "Apache-2.0",
    "MIT": "MIT",
    "BSD2CLAUSE": "BSD-2-Clause",
    "BSD3CLAUSE": "BSD-3-Clause",
    "BSD": "BSD-3-Clause",
    "CC0": "CC0-1.0",
    "CC010": "CC0-1.0",
    "CCBY30": "CC-BY-3.0",
    "CCBY40": "CC-BY-4.0",
    "CCBYSA30": "CC-BY-SA-3.0",
    "CCBYSA40": "CC-BY-SA-4.0",
    "CCBYNC40": "CC-BY-NC-4.0",
    "CCBYNCSA40": "CC-BY-NC-SA-4.0",
    "CCBYNCND40": "CC-BY-NC-ND-4.0",
    "ODCBY10": "ODC-BY-1.0",
    "ODCBY": "ODC-BY-1.0",
    "GPL30": "GPL-3.0",
    "LGPL30": "LGPL-3.0",
    "UNKNOWN": "unknown",
    "OTHER": "unknown",
    "": "unknown",
}


class LicenceError(RuntimeError):
    """A row in a training split carries a licence that is not allowed."""


def normalise_spdx(licence: str | None) -> str:
    """Normalise a licence string to an SPDX-ish identifier.

    Hugging Face card metadata is inconsistent about case and punctuation
    (``apache-2.0``, ``Apache 2.0``, ``cc-by-4.0``), so compare on a canonical form rather than
    on whatever the uploader typed.

    Args:
        licence: A licence string, or ``None``.

    Returns:
        A canonical identifier, or ``"unknown"`` when it cannot be recognised.
    """
    if licence is None:
        return "unknown"
    key = re.sub(r"[^A-Za-z0-9]", "", licence).upper()
    if key in _SPDX_ALIASES:
        return _SPDX_ALIASES[key]
    # Fall back to the caller's own spelling when it already looks like a known SPDX id.
    for known in (*TRAIN_LICENCE_ALLOWLIST, "CC-BY-SA-3.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0"):
        if re.sub(r"[^A-Za-z0-9]", "", known).upper() == key:
            return known
    return licence.strip() or "unknown"


def classify_licence(licence: str | None) -> LicenceVerdict:
    """Decide what a licence permits.

    Args:
        licence: A licence string, in any common spelling.

    Returns:
        ``"train"`` if it is on :data:`TRAIN_LICENCE_ALLOWLIST`; ``"eval-only"`` if it is a
        recognised non-commercial or no-derivatives licence; ``"refused"`` otherwise —
        including anything unrecognised, which is the point.
    """
    spdx = normalise_spdx(licence)
    if spdx in TRAIN_LICENCE_ALLOWLIST:
        return "train"
    upper = spdx.upper()
    if any(marker in upper for marker in EVAL_ONLY_MARKERS):
        return "eval-only"
    return "refused"


@dataclass(frozen=True)
class _Offence:
    source: str
    licence: str
    verdict: str
    split: str
    rows: int


def _rows_of(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    materialised = list(rows)
    for index, row in enumerate(materialised):
        for field in ("split", "license", "source"):
            if field not in row:
                raise LicenceError(
                    f"row {index} is missing the {field!r} field; every row must carry "
                    "source, license and split (CLAUDE.md schema)"
                )
    return materialised


def assert_train_splits_are_licensed(rows: Iterable[Mapping[str, Any]]) -> None:
    """Raise unless every row in a training split is permissively licensed.

    Args:
        rows: Rows in the project's jsonl schema; each needs ``source``, ``license`` and
            ``split``.

    Raises:
        LicenceError: If any row in a split named in :data:`TRAIN_SPLITS` carries a licence
            that is not on the allowlist. The message names every offending
            ``(source, licence, split)`` and its row count, so the fix is obvious.
    """
    offences: dict[tuple[str, str, str], int] = {}
    for row in _rows_of(rows):
        split = str(row["split"]).lower()
        if split not in TRAIN_SPLITS:
            continue
        verdict = classify_licence(str(row["license"]))
        if verdict != "train":
            key = (str(row["source"]), normalise_spdx(str(row["license"])), split)
            offences[key] = offences.get(key, 0) + 1

    if not offences:
        return
    listed = sorted(
        _Offence(source, licence, classify_licence(licence), split, count)
        for (source, licence, split), count in offences.items()
    )
    detail = "\n".join(
        f"  {o.rows:6d} rows  {o.source} [{o.licence}] -> {o.verdict}  in split {o.split!r}"
        for o in listed
    )
    raise LicenceError(
        "training splits contain rows that may not be trained on (ADR 0004):\n"
        f"{detail}\n"
        f"allowed for training: {', '.join(sorted(TRAIN_LICENCE_ALLOWLIST))}\n"
        "Tag these rows as split='eval' or drop the source."
    )


def summarise_licences(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarise licences per source, for the dataset card and the build log.

    Args:
        rows: Rows in the project's jsonl schema.

    Returns:
        Per-source mapping of licence, verdict, total rows and rows per split.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in _rows_of(rows):
        source = str(row["source"])
        licence = normalise_spdx(str(row["license"]))
        entry = out.setdefault(
            source,
            {"license": licence, "verdict": classify_licence(licence), "rows": 0, "splits": {}},
        )
        if entry["license"] != licence:
            raise LicenceError(
                f"source {source!r} carries two different licences: "
                f"{entry['license']!r} and {licence!r}"
            )
        entry["rows"] += 1
        split = str(row["split"])
        entry["splits"][split] = entry["splits"].get(split, 0) + 1
    return {k: out[k] for k in sorted(out)}


def state_hashes(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect the set of state hashes present in ``rows``.

    Used by the Tier-2 leakage guard: any external eval row whose state we trained on is
    dropped before reporting (Phase 1 plan, amendment B).

    Args:
        rows: Rows carrying a ``state`` field.

    Returns:
        Hex digests, one per distinct state.
    """
    import hashlib

    return {
        hashlib.blake2b(str(row["state"]).strip().encode("utf-8"), digest_size=16).hexdigest()
        for row in rows
    }


def drop_leaked_rows(
    external: Sequence[Mapping[str, Any]], train_rows: Iterable[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], int]:
    """Drop external eval rows whose state appears in our training data.

    ``pngwn/system-one-decisions`` is derived from banking77 and other sources Phase 1 trains
    on, so the external Tier-2 set can overlap our train splits even though the *datasets*
    differ. Comparing state hashes catches that; comparing dataset names would not.

    Args:
        external: The external evaluation rows.
        train_rows: Every row in our training splits.

    Returns:
        ``(kept_rows, dropped_count)``. The count belongs in ``metrics.json``.
    """
    trained = state_hashes(train_rows)
    kept = [row for row in external if not state_hashes([row]) & trained]
    return kept, len(external) - len(kept)
