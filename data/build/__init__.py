"""Dataset construction: fetch, normalise, augment, split.

Everything here writes to ``data/processed/``; ``data/raw/`` is never edited by hand.
The licence gate in :mod:`data.build.licences` runs at the end of every build and again as a
test over the committed manifest, so an unlicensed row cannot reach a training split by any
route (ADR 0004).
"""
