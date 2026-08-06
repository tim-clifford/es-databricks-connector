"""Structured Streaming helper: a foreachBatch function that bulk-writes each micro-batch.

Serverless note: only Trigger.AvailableNow / Trigger.Once are supported on serverless
(processingTime is rejected). Run an always-on export as a continuous Databricks job
whose runs each drain new source commits via availableNow. Checkpoints must live on
a UC Volume (dbfs:/tmp paths fail with INSUFFICIENT_PERMISSIONS on serverless).

Idempotency: with EsConfig.id_field set, streaming's at-least-once checkpoint semantics
combine with deterministic _id to give effectively exactly-once (retries upsert).

WHY THIS RAISES BY DEFAULT
--------------------------
Structured Streaming commits a micro-batch's checkpoint offset when `foreachBatch` RETURNS
NORMALLY. It has no view into what happened to the documents inside. So if this function
swallowed a failed write, a micro-batch in which Elasticsearch rejected every document would
still be marked successful, the offset would advance past those rows, and they would never be
retried: silent, permanent data loss, with a green job in the UI and nothing to alert on.

Raising instead fails the batch, so Spark retries it and the checkpoint does NOT advance. Combined
with a deterministic `_id`, that retry is an idempotent upsert rather than a duplicate. `on_error`
lets a caller opt out (`"log"`/`"ignore"`) for a deliberately loss-tolerant pipeline, but the
default is the safe direction because the failure it prevents is invisible and unrecoverable.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .config import EsConfig
from .bulk import bulk_write, reconcile_or_raise

_log = logging.getLogger(__name__)

# on_error policies.
RAISE = "raise"     # fail the micro-batch so Spark retries it and the checkpoint does not advance
LOG = "log"         # log a warning and let the checkpoint advance (rows are NOT retried)
IGNORE = "ignore"   # say nothing and let the checkpoint advance (rows are NOT retried)
_ON_ERROR_POLICIES = (RAISE, LOG, IGNORE)


def make_foreach_batch(cfg: EsConfig, on_batch: Optional[Callable] = None,
                       on_error: str = RAISE) -> Callable:
    """Return a foreachBatch(df, batch_id) function for writeStream.

    on_batch, if given, is called with (batch_id, result_dict) after each micro-batch
    for logging/metrics. Kept optional so the library stays dependency-free. It is invoked
    BEFORE the error policy is applied, so a metrics/dead-letter hook still observes a failed
    batch before it raises.

    on_error decides what a failed write does to the STREAM (see the module docstring for why the
    default is to raise):
      - "raise"  (default): raise EsWriteError, failing the micro-batch. Spark retries it and the
                 checkpoint does not advance, so the rows are not lost. Use this unless you have a
                 specific reason not to.
      - "log":   log a warning and return normally. The checkpoint ADVANCES and the rejected rows
                 are never retried. Only appropriate for a loss-tolerant pipeline.
      - "ignore": return normally with no log. Silent data loss; the checkpoint advances anyway.

    A write "failed" when Elasticsearch rejected any document (`errors > 0`) or any row went
    unaccounted for (`unaccounted > 0`, i.e. loss below the per-document level). Expected
    delete-404 no-ops (`ignored`) are NOT failures.
    """
    if on_error not in _ON_ERROR_POLICIES:
        raise ValueError(f"on_error must be one of {_ON_ERROR_POLICIES}, got {on_error!r}")

    def _foreach_batch(batch_df, batch_id: int) -> None:
        # df.isEmpty() (not df.rdd.isEmpty(), which is blocked on serverless)
        if batch_df.isEmpty():
            if on_batch:
                # Same shape bulk_write returns so a callback can read any of those keys
                # uniformly, empty batch or not, plus an `empty` flag to distinguish "no rows this
                # micro-batch" from "0 written of N".
                on_batch(batch_id, {"written": 0, "deleted": 0, "errors": 0, "ignored": 0,
                                    "coerced_nonfinite": 0, "total_input": 0, "unaccounted": 0,
                                    "overcounted": 0, "error_samples": [], "empty": True})
            return
        result = bulk_write(batch_df, cfg)
        # The metrics hook runs first so it sees a failing batch too (it may be the dead-letter path).
        if on_batch:
            on_batch(batch_id, result)

        if on_error == RAISE:
            # Raises EsWriteError; failing the batch is what keeps the checkpoint from advancing.
            reconcile_or_raise(result, index=cfg.index)
        elif on_error == LOG:
            if result.get("errors") or (result.get("unaccounted") or 0) > 0:
                _log.warning(
                    "batch %s wrote to index %r with failures and on_error='log', so the checkpoint "
                    "WILL advance and these rows will not be retried: errors=%s unaccounted=%s "
                    "total_input=%s first_failures=%s",
                    batch_id, cfg.index, result.get("errors"), result.get("unaccounted"),
                    result.get("total_input"), (result.get("error_samples") or [])[:3])

    return _foreach_batch
