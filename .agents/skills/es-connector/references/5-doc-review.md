# Reference 5: Holistic documentation review

The connector's prose docs describe the code but are **not executable**, so nothing fails when they
drift. Two narrow classes ARE enforced: `scripts/check_readme_sync.py` (a module/fixture/script no
README lists) and `scripts/check_version_consistency.py` (a stale version or wheel pin). Neither can
catch a stale *description* or an out-of-date claim. This review is the backstop for the rest.

**When to run it:** as a **pre-change step** at the start of a work session that will modify the repo
(so you start from docs that match reality, and know what you're changing), AND as part of **code
review** for any PR that changes behavior, the public API, datatype handling, the release process, or
adds/removes files. It is cheap (a few minutes) and it is the only thing that keeps the skill itself
honest.

## The goal: describe the current state, not the history

A doc's job is to let a user or agent understand **what the repo is now, how to use it, and what to
watch out for**, in as few words as make that clear. It is NOT a changelog or a design journal. The
review below pushes you to keep docs *in sync* with the code, but sync cuts both ways: syncing is as
much **removing** now-irrelevant text as adding new text. Resist the natural pull to append an
explanation for every change, that ratchet is how a clean README turns into an archaeological record
no one reads.

Concrete rules:
- **Describe behavior in the present tense as if it were always so.** "`timestamp` stores the true
  UTC instant regardless of session zone." NOT "as of 0.4.1 we fixed a bug where..." The fix is real
  and belongs in git history and the release notes; the *doc* states the current guarantee.
- **Version/history narration lives in exactly two places:** `HANDOFF.md` (current status of known
  limitations, which legitimately tracks what is and isn't addressed) and the GitHub release notes /
  commit history. Everywhere else, write the current truth without the story of how it got there.
- **Prune when you sync.** If a change makes a caveat, workaround, or explanation obsolete, DELETE it
  in the same edit. A stale "known issue" that's been fixed is as harmful as a missing one.
- **Prefer the shortest form that a first-time reader can act on.** A gotcha is one tight paragraph
  with the fix, not a retrospective of why it exists. If depth is genuinely needed, it goes in a
  skill reference (progressive disclosure), not inline in the README a newcomer reads first.
- **When adding vs. pruning are in tension, ask:** "does a reader USING the connector today need
  this to use it correctly or avoid a surprise?" If yes, keep it, tightly. If it only explains a past
  decision or a change already invisible in current behavior, leave it to history.

## The documentation set (every file that can drift)

| Doc | Describes | Most likely to go stale when... |
|-----|-----------|--------------------------------|
| `README.md` | public API, datatype tables, config, reading, repo layout, build/release pointer | a datatype's transform changes; a public function/config field is added or renamed; a file is added |
| `integration_tests/README.md` | each integration fixture's purpose | a fixture is added, removed, or changes what it owns |
| `RELEASING.md` | the 4-step release process + the hard gates | the release flow, gate scripts, FEVM paths, or step order change |
| `HANDOFF.md` | production-readiness / known limitations, pinned to a version | a limitation is closed, a version ships, or a gap changes status (check the version in its header matches `pyproject.toml`) |
| `.agents/skills/es-connector/SKILL.md` + `references/*` | this skill: architecture, fidelity model, datatype contract, ES gotchas, release | ANY of the above change, because the skill restates them; the skill is the doc most prone to silent drift |
| `CLAUDE.md` | the short auto-loaded pointer to this skill, plus the handful of rules most likely to be broken without it | a skill rule is added/changed/removed, a gate script is added, or a path in it moves. It EXISTS because `.agents/skills/` is not auto-discovered by the agent harness, so it is the only part of the skill that loads by default: if it drifts, the skill is effectively unenforced. Treat a change to SKILL.md as a prompt to re-read it. |

## The checklist

Run top to bottom; each item is a concrete "does the doc still match the code" check, not a vibe.

1. **Version consistency: ENFORCED, run the script.** `python scripts/check_version_consistency.py`
   (exit 0). It compares `pyproject.toml` `[project].version` against `__init__.py` `__version__`,
   `HANDOFF.md`'s header, and every `databricks_es_connector-<version>-...whl` reference in the
   docs, the skill, and `integration_tests/config/test_config.yml`. That last one is why this was
   promoted from a manual check: it pins the wheel the LIVE integration tier *installs*, so a stale
   pin there means the whole tier validates the PREVIOUS release while reporting success. It sat at
   0.5.0 while the source was 0.6.0. The script only sees THIS repo, so any consumer that pins a
   wheel filename of its own is out of its reach and has to be checked wherever it lives.
2. **Public API.** Every name in `__init__.py`'s `__all__` is documented in the README, and the
   README names no function/config field that no longer exists. Config fields
   (`EsWriteConfig`/`EsReadConfig`) match `config.py`.
3. **Behavior and logic changes (the general case, easiest to miss).** Any change to what the code
   *does*, not just its API surface, must be reflected wherever the docs describe that behavior. This
   is broader than the datatype tables: a bug FIX to existing behavior, a changed default, a new or
   newly-closed limitation, an altered ordering/guarantee, or a new ES-interaction gotcha. Ask: "what
   did this change make true (or no longer true) that a doc states?" Then check every place that
   states it, README prose (not just tables), `HANDOFF.md` limitations, `RELEASING.md`, and the
   relevant skill reference. The timezone fix is the canonical example: it touched no public-API
   signature and only one datatype-table row, but it changed a correctness guarantee that belongs in
   the README timezone section, `references/3-es-gotchas.md`, AND HANDOFF, none of which a
   table-only check would flag. Fixes are the sneakiest: a closed limitation left in HANDOFF
   understates the connector, and a removed gotcha left in the docs misleads. **State the new
   behavior in the present tense (see "the goal" above), don't narrate the change; and delete any
   now-obsolete caveat rather than layering a correction on top of it.**
4. **Datatype tables.** The README "Datatype coverage" (write) and "Read fidelity" (read) tables, and
   this skill's `references/1-fidelity-model.md`, all match `coerce_value` / `read_coerce`, and the
   one-way-delta list is exactly three (decimal, sub-ms timestamp, float32) unless a change
   deliberately added a fourth (which must be documented in all three places). See ref 2.
5. **File enumeration.** `scripts/check_readme_sync.py` exits 0 (modules, fixtures, scripts are
   listed). Then eyeball that each listed description is still ACCURATE, not just present.
6. **Release process.** `RELEASING.md` steps match the actual scripts (`scripts/*.py`), the FEVM
   paths/profile are current, and the step order still has build+upload BEFORE the integration run.
7. **`CLAUDE.md` vs. the skill.** It is the ONLY auto-loaded pointer to this skill
   (`.agents/skills/` is not discovered by the harness). Check that every rule it states is still
   true, that the paths it cites exist, and that a rule added to SKILL.md since the last review is
   reflected if it is the kind someone would violate without reading the full skill. Keep it SHORT:
   a signpost, not a second copy of the skill. One that grows into a summary of everything stops
   being read, which is the failure mode it exists to fix.
8. **No outbound references to a consumer.** This library ships to customers who have only the
   wheel, so nothing here may name a downstream repo, demo, or example project. Grep for one before
   finishing (a doc, a comment, or a skill reference is just as much of a leak as code). State what a
   consumer must DO instead of pointing at where someone else did it. Awareness is inbound only.
9. **The skill vs. the code.** Spot-check that SKILL.md's "architecture in one screen" and the
   critical rules still hold (write-path order, the serverless constraints, the five-places rule),
   and that refs 1, 3, 4 don't cite a function, path, or behavior that changed. Because the skill
   duplicates the READMEs by design, it drifts first, treat it as guilty until checked.
10. **HANDOFF status.** Any "Open item" that has since been addressed (e.g. the timezone fix) is moved
   out of open items or annotated, so the readiness doc doesn't understate the connector.
11. **Conciseness and pruning (the counterbalance to items 3-10).** Items 3-10 push toward *adding*;
   this one pushes back. Read each doc as a first-time user would and cut what no longer earns its
   place: obsolete caveats/workarounds for since-fixed behavior, changelog-style "as of vX we..."
   narration, duplicate explanations of the same point across files, and depth that belongs in a
   skill reference rather than the README. Ask "does a reader USING the connector today need this?"
   A shorter doc that states the current truth beats a longer one that records how it got there. If a
   review's net effect is only additions, be suspicious you skipped this item.

## Output of a review

State findings as: file, the specific stale line/claim, and the code it no longer matches. If a doc
is correct, say so explicitly (don't imply drift where there is none). Fixing the drift is a docs
change like any other, land it with the behavior change that caused it, not in a separate "docs
catch-up" pass later (that later pass is how drift accumulates in the first place).

## Why this isn't a script

Two items are now scripts: version consistency (`check_version_consistency.py`) and item 5's
presence half (`check_readme_sync.py`). Everything else, "is this description still accurate," is a
judgment call that needs reading the code, which is exactly what a script can't do and a review can.

**Keep promoting.** When a check here becomes fully mechanical, move it into `scripts/`, cite it from
RELEASING.md, and rewrite the checklist item to say "run the script" rather than leaving it as prose
someone can skip. Both promotions so far were prompted by the check failing in reality first, so
treat a drift that this checklist *should* have caught as a signal to mechanize it, not just to fix
the instance.
