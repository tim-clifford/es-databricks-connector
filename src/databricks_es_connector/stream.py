"""Structured Streaming helper: a foreachBatch function that bulk-writes each micro-batch.

Serverless note: only Trigger.AvailableNow / Trigger.Once are supported on serverless
(processingTime is rejected). Run an always-on export as a continuous Databricks job
whose runs each drain new source commits via availableNow. Checkpoints must live on
a UC Volume (dbfs:/tmp paths fail with INSUFFICIENT_PERMISSIONS on serverless).

Idempotency: with EsConfig.id_field set, streaming's at-least-once checkpoint semantics
combine with deterministic _id to give effectively exactly-once (retries upsert).
"""
from __future__ import annotations

from typing import Callable, Optional

from .config import EsConfig
from .bulk import bulk_write


def make_foreach_batch(cfg: EsConfig, on_batch: Optional[Callable] = None) -> Callable:
    """Return a foreachBatch(df, batch_id) function for writeStream.

    on_batch, if given, is called with (batch_id, result_dict) after each micro-batch
    for logging/metrics. Kept optional so the library stays dependency-free.
    """
    def _foreach_batch(batch_df, batch_id: int) -> None:
        # df.isEmpty() (not df.rdd.isEmpty(), which is blocked on serverless)
        if batch_df.isEmpty():
            if on_batch:
                # Same shape bulk_write returns (written/deleted/errors/total_input/error_samples)
                # so a callback can read any of those keys uniformly, empty batch or not, plus an
                # `empty` flag to distinguish "no rows this micro-batch" from "0 written of N".
                on_batch(batch_id, {"written": 0, "deleted": 0, "errors": 0,
                                    "total_input": 0, "error_samples": [], "empty": True})
            return
        result = bulk_write(batch_df, cfg)
        if on_batch:
            on_batch(batch_id, result)

    return _foreach_batch
