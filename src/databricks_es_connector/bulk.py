"""Executor-side bulk write to Elasticsearch, serverless-safe via mapInPandas.

Why mapInPandas and not foreachPartition: serverless compute blocks RDD APIs
(df.rdd / foreachPartition raise INSUFFICIENT_PERMISSIONS). mapInPandas is the
supported way to run per-partition code on serverless, and it parallelizes the
bulk write across executors, throughput scales with the cluster like the old
Spark connector did.

The Elasticsearch client is built INSIDE the partition function from EsConfig,
so nothing non-serializable is captured on the driver.
"""
from __future__ import annotations

import json
from typing import Iterator

from .config import EsConfig
from .spark_prep import sanitize_for_arrow, normalize_timestamps_for_utc
from .transform import build_action

# Per-document outcomes from classify_bulk_result. Kept as module constants so the
# writer loop and the unit tests agree on the exact strings.
WRITTEN = "written"
DELETED = "deleted"
IGNORED = "ignored"   # a delete-404: expected no-op, counted as neither write nor error
ERROR = "error"

# Cap on how many failed-doc diagnostics we retain, per partition AND after merging on the driver.
# The error COUNT is always exact; only the retained sample list is bounded, so a pathological
# all-failures batch can't blow up executor or driver memory. A handful is enough to diagnose the
# cause (mapping conflict, term-limit, etc.); it is a breadcrumb, not a dead-letter queue.
ERROR_SAMPLE_CAP = 20


def _extract_error_sample(op_type: str, item: dict) -> dict:
    """Pull a compact, JSON-safe diagnostic from one failed streaming_bulk result item.

    Keeps only what identifies and explains the failure: the doc _id, the op, the HTTP status,
    and ES's error reason (truncated). Deliberately small so a batch of failures stays bounded.
    """
    err = item.get("error")
    if isinstance(err, dict):
        reason = err.get("reason") or err.get("type") or ""
    else:
        reason = "" if err is None else str(err)
    return {
        "_id": item.get("_id"),
        "op_type": op_type,
        "status": item.get("status"),
        "reason": str(reason)[:300],
    }


def classify_bulk_result(ok: bool, op_type: str, status: int) -> str:
    """Classify one streaming_bulk result into WRITTEN / DELETED / IGNORED / ERROR.

    Pure so the suppression rule is unit-testable without Spark or a live ES client.

    The one suppression: a *delete* that returns *404* is an expected no-op (the doc was
    never indexed, was filtered out, or a replay already deleted it). It is IGNORED, not an
    error. Every other non-ok result (including a 404 on an index/create/update, or a
    409/5xx on a delete) is an ERROR and must be counted. Suppression is scoped to the
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
        total_input = 0            # rows fed in, so the caller can reconcile against the outcomes
        error_samples = []         # bounded diagnostics for failed docs (see ERROR_SAMPLE_CAP)
        for pdf in iterator:
            rows = pdf.to_dict("records")
            total_input += len(rows)
            actions = [
                build_action(
                    row,
                    index=cfg.index,
                    id_field=cfg.id_field,
                    drop_fields=cfg.drop_fields,
                    has_deletes=cfg.has_deletes,
                    delete_flag_column=cfg.delete_flag_column,
                )
                for row in rows
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
                    if len(error_samples) < ERROR_SAMPLE_CAP:
                        error_samples.append(_extract_error_sample(op_type, item))
                # IGNORED (delete-404) counts as nothing, an expected no-op.
        # error_samples is JSON-encoded into a single string column: mapInPandas needs a flat,
        # typed schema and can't carry a nested list<struct> of varying content cleanly.
        yield pd.DataFrame({
            "written": [written], "deleted": [deleted], "errors": [errors],
            "total_input": [total_input], "error_samples": [json.dumps(error_samples)],
        })

    return _write


def _merge_partition_results(rows) -> dict:
    """Combine the per-partition summary rows into the final result dict.

    Pure (no Spark) so it is unit-testable. Each row carries written/deleted/errors/total_input and
    a JSON string of that partition's bounded error samples. Counts are summed exactly; the sample
    lists are concatenated and re-capped at ERROR_SAMPLE_CAP so the driver result stays bounded even
    across many partitions. `errors == 0 and total_input != written + deleted + <ignored>` is the
    caller's signal that some rows were lost at chunk level (see raise_on_exception in the writer).
    """
    written = deleted = errors = total_input = 0
    samples = []
    for r in rows:
        written += int(r["written"] or 0)
        deleted += int(r["deleted"] or 0)
        errors += int(r["errors"] or 0)
        total_input += int(r["total_input"] or 0)
        if len(samples) < ERROR_SAMPLE_CAP and r["error_samples"]:
            samples.extend(json.loads(r["error_samples"]))
    return {
        "written": written, "deleted": deleted, "errors": errors,
        "total_input": total_input, "error_samples": samples[:ERROR_SAMPLE_CAP],
    }


def bulk_write(df, cfg: EsConfig) -> dict:
    """Write a Spark DataFrame to Elasticsearch.

    Returns {'written', 'deleted', 'errors', 'total_input', 'error_samples'}:
      - 'written': index/upsert ops that succeeded.
      - 'deleted': successful delete-by-id ops (only non-zero when cfg.has_deletes).
      - 'errors': docs ES rejected (exact count).
      - 'total_input': rows handed to the writer. Reconcile: written+deleted+errors < total_input
        means some rows were lost below the per-doc level (e.g. a chunk-level exception); equality
        (accounting for delete-404 no-ops, which count as none) means every row was accounted for.
      - 'error_samples': up to ERROR_SAMPLE_CAP diagnostics ({_id, op_type, status, reason}) for
        rejected docs, so a failure is actionable rather than an opaque count. Bounded, not a full
        dead-letter log.
    Batch entry point; for streaming, use stream.make_foreach_batch.

    Arrow-hostile columns (VARIANT / INTERVAL, at any nesting depth) are serialized to strings
    automatically via sanitize_for_arrow before the mapInPandas export, mapInPandas cannot carry
    them otherwise (VARIANT -> JSON string, scalar INTERVAL -> its Spark string form). Callers do
    not need to pre-process; any valid Spark DataFrame works. Such columns land in ES as strings
    (map them as keyword/text, not object).

    TimestampType columns are converted to epoch-millis longs in Spark (normalize_timestamps_for_utc)
    so the stored instant is correct regardless of spark.sql.session.timeZone, without mutating the
    caller's session. This runs AFTER sanitize_for_arrow because reading df.schema (which the
    timestamp walk needs) throws on a VARIANT column under Spark Connect, and sanitize removes those.
    """
    df = sanitize_for_arrow(df)
    df = normalize_timestamps_for_utc(df)
    writer = make_partition_writer(cfg)
    rows = df.mapInPandas(
        writer,
        "written long, deleted long, errors long, total_input long, error_samples string",
    ).collect()
    return _merge_partition_results(rows)
