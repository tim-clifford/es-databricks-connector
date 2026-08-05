# databricks-es-connector

**Before changing anything in this repo, read `.agents/skills/es-connector/SKILL.md`.**

It is the maintenance skill for this library and it is NOT auto-discovered (it lives under
`.agents/`, not `.claude/skills/`), so nothing loads it for you. It carries the rules that this
repo's correctness depends on and that are easy to violate without knowing them:

- **The five-places rule.** Changing how a Spark datatype is handled requires five files to move
  together (write transform, read inverse, round-trip oracle, integration fixture, README tables) or
  fidelity silently breaks. `references/2-datatype-contract.md`.
- **The fidelity contract.** `coerce_value` and `read_coerce` must stay exact inverses except for
  three documented one-way deltas. `references/1-fidelity-model.md`.
- **Docs describe the current state, never the history.** No "as of 0.6.0 we...", no before/after
  tables. Pruning obsolete text is part of syncing, not optional. Run the doc-review checklist in
  `references/5-doc-review.md` at the START of a session that will modify the repo, and again at
  review time. If a doc change is net-additive only, the pruning step was skipped.
- **Serverless constraints are load-bearing.** No RDD APIs; `df.schema`/`df.columns` throw on a
  VARIANT column; executors build their own ES client from a frozen config. Do not "simplify" these.
- **Three hard release gates** (`scripts/check_requirements_match.py`, `check_readme_sync.py`,
  `check_version_consistency.py`). Run all three before claiming a change is complete.
- **A mechanizable check belongs in `scripts/`**, not in a checklist a reviewer can skip.
- **This repo knows nothing about its consumers.** It is a standalone library: customers install the
  wheel without access to any demo or example repo, so no file here may name one. Describe what a
  consumer must DO ("compare the indexed value against what you sent"), never where some other repo
  does it. Awareness is one-way, inbound only.

Both test tiers: `pytest tests/` (fast, pure-Python) and the live `integration_tests/` tier on FEVM
serverless via `dbx_test`. The Spark-side code in `spark_prep.py` can ONLY be proven in the
integration tier, so a green unit run does not cover it. `references/4-release-and-tests.md`.
