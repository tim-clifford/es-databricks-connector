# Reference 5: Holistic documentation review

The connector's prose docs describe the code but are **not executable**, so nothing fails when they
drift. `scripts/check_readme_sync.py` catches one narrow class (a module/fixture/script that no
README lists), but it cannot catch a stale *description*, a wrong version, or an out-of-date claim.
This review is the human/agent backstop for the rest.

**When to run it:** as a **pre-change step** at the start of a work session that will modify the repo
(so you start from docs that match reality, and know what you're changing), AND as part of **code
review** for any PR that changes behavior, the public API, datatype handling, the release process, or
adds/removes files. It is cheap (a few minutes) and it is the only thing that keeps the skill itself
honest.

## The documentation set (every file that can drift)

| Doc | Describes | Most likely to go stale when... |
|-----|-----------|--------------------------------|
| `README.md` | public API, datatype tables, config, reading, repo layout, build/release pointer | a datatype's transform changes; a public function/config field is added or renamed; a file is added |
| `integration_tests/README.md` | each integration fixture's purpose | a fixture is added, removed, or changes what it owns |
| `RELEASING.md` | the 4-step release process + the two gates | the release flow, gate scripts, FEVM paths, or step order change |
| `HANDOFF.md` | production-readiness / known limitations, pinned to a version | a limitation is closed, a version ships, or a gap changes status (check the version in its header matches `pyproject.toml`) |
| `.agents/skills/es-connector/SKILL.md` + `references/*` | this skill: architecture, fidelity model, datatype contract, ES gotchas, release | ANY of the above change, because the skill restates them; the skill is the doc most prone to silent drift |

## The checklist

Run top to bottom; each item is a concrete "does the doc still match the code" check, not a vibe.

1. **Version consistency.** `pyproject.toml` `[project].version`, `__init__.py` `__version__`, the
   wheel filename in every `%pip install` line (README + demos), and `HANDOFF.md`'s header version
   all agree. (HANDOFF has been pinned to an older version before; this catches it.)
2. **Public API.** Every name in `__init__.py`'s `__all__` is documented in the README, and the
   README names no function/config field that no longer exists. Config fields
   (`EsWriteConfig`/`EsReadConfig`) match `config.py`.
3. **Datatype tables.** The README "Datatype coverage" (write) and "Read fidelity" (read) tables, and
   this skill's `references/1-fidelity-model.md`, all match `coerce_value` / `read_coerce`, and the
   one-way-delta list is exactly three (decimal, sub-ms timestamp, float32) unless a change
   deliberately added a fourth (which must be documented in all three places). See ref 2.
4. **File enumeration.** `scripts/check_readme_sync.py` exits 0 (modules, fixtures, scripts are
   listed). Then eyeball that each listed description is still ACCURATE, not just present.
5. **Release process.** `RELEASING.md` steps match the actual scripts (`scripts/*.py`), the FEVM
   paths/profile are current, and the step order still has build+upload BEFORE the integration run.
6. **The skill vs. the code.** Spot-check that SKILL.md's "architecture in one screen" and the
   critical rules still hold (write-path order, the serverless constraints, the five-places rule),
   and that refs 1, 3, 4 don't cite a function, path, or behavior that changed. Because the skill
   duplicates the READMEs by design, it drifts first, treat it as guilty until checked.
7. **HANDOFF status.** Any "Open item" that has since been addressed (e.g. the timezone fix) is moved
   out of open items or annotated, so the readiness doc doesn't understate the connector.

## Output of a review

State findings as: file, the specific stale line/claim, and the code it no longer matches. If a doc
is correct, say so explicitly (don't imply drift where there is none). Fixing the drift is a docs
change like any other, land it with the behavior change that caused it, not in a separate "docs
catch-up" pass later (that later pass is how drift accumulates in the first place).

## Why this isn't a script

Items 1 and 4-presence are mechanizable (and item 4 already is). The rest, "is this description still
accurate," is a judgment call that needs reading the code, exactly what a script can't do and a
review can. If a check here becomes fully mechanical (e.g. version consistency), promote it to
`scripts/` and cite it from RELEASING.md, don't leave it as a manual step. Keep the deterministic
parts enforced and the judgment parts reviewed.
