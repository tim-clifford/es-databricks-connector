#!/usr/bin/env python3
"""Verify the docs enumerate the code: every shipped module / test fixture / release script is
mentioned by name in the README that is supposed to list it.

Why this exists: a diff-scoped code review can't catch "a file was added but the doc that lists it
wasn't updated" (the doc lines that should have changed simply don't appear in the diff). This is a
whole-tree invariant check instead: for each (files, doc) rule below, every file must be referenced
by basename in that doc. Exits non-zero and prints the gaps on any drift, so it gates a release the
same way scripts/check_requirements_match.py does.

It checks PRESENCE (the file's basename appears somewhere in the doc), not that the description is
correct, that still needs human eyes. But it makes "forgot to document a new module/fixture"
impossible to miss, which is the recurring drift it targets.

Usage:
    python scripts/check_readme_sync.py         # from the repo root
Exits 0 when every rule is satisfied, 1 (with a per-rule gap list) otherwise.
"""
from __future__ import annotations

import glob
import os
import re
import sys

# Each rule: (human label, glob of files that must be documented, list of docs any of which may
# mention the file, predicate to exclude files). A file satisfies the rule if its basename appears
# in AT LEAST ONE of the docs. Paths are relative to the repo root (this script's parent's parent).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _skip_dunder(path: str) -> bool:
    return os.path.basename(path).startswith("__")


RULES = [
    ("library modules -> README.md",
     "src/databricks_es_connector/*.py", ["README.md"], _skip_dunder),
    ("integration fixtures -> integration_tests/README.md",
     "integration_tests/test_*.py", ["integration_tests/README.md"], None),
    ("integration fixtures -> README.md repo-layout",
     "integration_tests/test_*.py", ["README.md"], None),
    ("release scripts -> RELEASING.md or README.md",
     "scripts/*.py", ["RELEASING.md", "README.md"], _skip_dunder),
]


def _read(path: str) -> str:
    full = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        sys.exit(f"error: doc not found: {path}")
    with open(full) as f:
        return f.read()


def _mentions(text: str, base: str) -> bool:
    """True if `text` references the filename `base` as a whole token, not as a substring of a
    longer name. A plain `base in text` check false-NEGATIVES: `transform.py` is a substring of
    `read_transform.py`, so documenting only the latter would wrongly satisfy the former and let the
    gate pass green over a real gap. Require the char before the basename to be a path/word boundary
    (start, whitespace, or a path separator), so `read_transform.py` does not count as a mention of
    `transform.py`, while `src/transform.py` and a bare `transform.py` do."""
    for m in re.finditer(re.escape(base), text):
        prev = text[m.start() - 1] if m.start() > 0 else ""
        if prev == "" or prev.isspace() or prev in "/\\`(":
            return True
    return False


def main() -> int:
    any_gap = False
    for label, file_glob, docs, skip in RULES:
        doc_texts = {d: _read(d) for d in docs}
        files = sorted(glob.glob(os.path.join(REPO_ROOT, file_glob)))
        gaps = []
        for path in files:
            if skip and skip(path):
                continue
            base = os.path.basename(path)
            if not any(_mentions(text, base) for text in doc_texts.values()):
                gaps.append(base)
        where = " / ".join(docs)
        if gaps:
            any_gap = True
            print(f"MISSING from {where}  [{label}]:")
            for g in gaps:
                print(f"    - {g}")
        else:
            print(f"OK: {label}")

    if any_gap:
        print("\nDocs are out of sync with the code. Add the missing file(s) to the doc(s) above "
              "(with a one-line description matching the neighbours) and re-run.")
        return 1
    print("\nOK: every module / fixture / release script is documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
