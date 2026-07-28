"""Executor-side bulk write to Elasticsearch, serverless-safe via mapInPandas.

Why mapInPandas and not foreachPartition: serverless compute blocks RDD APIs
(df.rdd / foreachPartition raise INSUFFICIENT_PERMISSIONS). mapInPandas is the
supported way to run per-partition code on serverless, and it parallelizes the
bulk write across executors — throughput scales with the cluster like the old
Spark connector did.

The Elasticsearch client is built INSIDE the partition function from EsConfig,
so nothing non-serializable is captured on the driver.
"""
from __future__ import annotations

from typing import Iterator

from .config import EsConfig
from .spark_prep import sanitize_for_arrow
from .transform import build_action

# Per-document outcomes from classify_bulk_result. Kept as module constants so the
# writer loop and the unit tests agree on the exact strings.
WRITTEN = "written"
DELETED = "deleted"
IGNORED = "ignored"   # a delete-404: expected no-op, counted as neither write nor error
ERROR = "error"


def classify_bulk_result(ok: bool, op_type: str, status: int) -> str:
    """Classify one streaming_bulk result into WRITTEN / DELETED / IGNORED / ERROR.

    Pure so the suppression rule is unit-testable without Spark or a live ES client.

    The one suppression: a *delete* that returns *404* is an expected no-op (the doc was
    never indexed, was filtered out, or a replay already deleted it). It is IGNORED, not an
    error. Every other non-ok result — including a 404 on an index/create/update, or a
    409/5xx on a delete — is an ERROR and must be counted. Suppression is scoped to the
    (op_type == 'delete' AND status == 404) pair only; nothing broader.
    """
    if ok:
        return DELETED if op_type == "delete" else WRITTEN
    if op_type == "delete" and status == 404:
        return IGNORED
    return ERROR


def make_partition_writer(cfg: EsConfig):
    """Return a mapInPandas-compatible function that bulk-writes each pandas chunk.

    The returned fn yields a one-row pandas DataFrame with the counts, so the driver
    can sum results without collecting the data itself.
    """
    def _write(iterator: "Iterator") -> "Iterator":
        import pandas as pd
        from elasticsearch import Elasticsearch, helpers

        es = Elasticsearch(**cfg.client_kwargs())
        written = 0
        deleted = 0
        errors = 0
        for pdf in iterator:
            actions = [
                build_action(
                    row,
                    index=cfg.index,
                    id_field=cfg.id_field,
                    drop_fields=cfg.drop_fields,
                    has_deletes=cfg.has_deletes,
                    delete_flag_column=cfg.delete_flag_column,
                )
                for row in pdf.to_dict("records")
            ]
            if not actions:
                continue
            # streaming_bulk with raise_on_error=False + yield_ok=True yields one
            # (ok, {op_type: item}) tuple per document, so we can classify each result
            # individually. We deliberately do NOT use helpers.bulk's ignore_status: that
            # would suppress a status across ALL op types (e.g. a 404 on an index would
            # also be swallowed). We need the suppression scoped to *delete* 404s only.
            for ok, result in helpers.streaming_bulk(
                es, actions, chunk_size=cfg.chunk_size,
                raise_on_error=False, raise_on_exception=False, yield_ok=True,
            ):
                op_type, item = next(iter(result.items()))
                outcome = classify_bulk_result(ok, op_type, item.get("status", 500))
                if outcome == WRITTEN:
                    written += 1
                elif outcome == DELETED:
                    deleted += 1
                elif outcome == ERROR:
                    errors += 1
                # IGNORED (delete-404) counts as nothing — an expected no-op.
        yield pd.DataFrame({"written": [written], "deleted": [deleted], "errors": [errors]})

    return _write


def bulk_write(df, cfg: EsConfig) -> dict:
    """Write a Spark DataFrame to Elasticsearch. Returns {'written', 'deleted', 'errors'}.

    'written' counts index/upsert ops, 'deleted' counts successful delete-by-id ops
    (only non-zero when cfg.has_deletes). Batch entry point; for streaming, use
    stream.make_foreach_batch.

    Arrow-hostile columns (VARIANT / INTERVAL, at any nesting depth) are serialized to strings
    automatically via sanitize_for_arrow before the mapInPandas export — mapInPandas cannot carry
    them otherwise (VARIANT -> JSON string, scalar INTERVAL -> its Spark string form). Callers do
    not need to pre-process; any valid Spark DataFrame works. Such columns land in ES as strings
    (map them as keyword/text, not object).
    """
    df = sanitize_for_arrow(df)
    writer = make_partition_writer(cfg)
    result = (
        df.mapInPandas(writer, "written long, deleted long, errors long")
        .groupBy()
        .sum("written", "deleted", "errors")
        .collect()
    )
    if not result:
        return {"written": 0, "deleted": 0, "errors": 0}
    row = result[0]
    return {
        "written": int(row["sum(written)"] or 0),
        "deleted": int(row["sum(deleted)"] or 0),
        "errors": int(row["sum(errors)"] or 0),
    }
