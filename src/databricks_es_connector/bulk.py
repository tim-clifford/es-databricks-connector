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

import datetime as _dt
import json
import logging
import time as _time
from typing import Iterator, Optional

from .config import EsConfig
from .spark_prep import sanitize_for_arrow, normalize_timestamps_for_utc
from .transform import build_action

_log = logging.getLogger(__name__)

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

# Every EsWriteConfig field whose VALUE is the name of a DataFrame column. Each one silently
# misbehaves when the name doesn't exist, so `_preflight` validates all of them:
#   id_field           -> a per-row KeyError on the executor, mid-write, after partial commits
#   drop_fields        -> prunes nothing, shipping a field the caller believes was withheld
#   delete_flag_column -> every intended delete becomes an upsert, with clean counts (proven live)
# Declared as one tuple, rather than left implicit in the checks, so the class is enumerated in one
# place: hardening these fields one at a time is how `delete_flag_column` stayed open while
# `drop_fields` was guarded. A test asserts this tuple exactly, so adding a fourth such field fails
# until someone decides whether it needs validating.
_COLUMN_NAMING_FIELDS = ("id_field", "drop_fields", "delete_flag_column")


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


def _streaming_bulk(es, actions, cfg: EsConfig):
    """One streaming_bulk stream with the connector's fixed error/retry settings.

    Factored out so the serial path and every concurrent worker in `_iter_bulk_results` issue
    IDENTICAL requests. The choices here are load-bearing:

      - raise_on_error=False + raise_on_exception=False + yield_ok=True: we get one (ok, item) tuple
        per document and classify each ourselves (`classify_bulk_result`). We deliberately do NOT use
        helpers.bulk's `ignore_status`, which would suppress a status across ALL op types (e.g. a 404
        on an index would also be swallowed); the connector's suppression is scoped to *delete* 404s
        only, and lives in the classifier.
      - max_retries + retry_on_status: streaming_bulk retries the individual documents ES rejected
        with a retryable status (429 = write queue full) with exponential backoff, re-sending only
        the failed subset. Without this the library default (max_retries=0) makes a transient 429 a
        permanent per-doc error on its first attempt; the transport-level EsConnection.max_retries
        does NOT cover it, because _bulk returns HTTP 200 even when individual items fail.
    """
    from elasticsearch import helpers
    return helpers.streaming_bulk(
        es, actions, chunk_size=cfg.chunk_size,
        raise_on_error=False, raise_on_exception=False, yield_ok=True,
        max_retries=cfg.max_retries_per_doc,
        retry_on_status=tuple(cfg.retry_on_doc_status),
    )


def _iter_bulk_results(es, actions, cfg: EsConfig):
    """Yield (ok, result) tuples, one per document, for the writer loop to classify.

    `cfg.write_concurrency == 1` (default) is a single serial `streaming_bulk`: EXACTLY the original
    path, no threads. `> 1` fans `actions` across that many worker threads, each running its own
    `streaming_bulk` (the elasticsearch-py client is thread-safe; it holds a connection pool), and
    merges their per-document results through a bounded queue as they complete. This keeps
    `write_concurrency` bulk requests in flight per partition to fill the ES round-trip wait, WITHOUT
    losing streaming_bulk's per-document 429 retry (which parallel_bulk drops entirely).

    A worker exception (e.g. a transport error surviving transport_max_retries) is re-raised on the
    consumer thread once the queue has drained, so a partial write FAILS the partition instead of
    silently reporting the documents a dead worker never sent as a clean success -- the exact silent
    loss this module is built to prevent. If instead the CONSUMER abandons this generator early (the
    classify loop raises, or the generator is closed/GC'd mid-stream), the `stop` flag releases any
    producer parked on a full queue so `ThreadPoolExecutor.__exit__`'s shutdown(wait=True) can never
    deadlock the partition on a `put()` that will never be drained.
    """
    n = cfg.write_concurrency
    if n <= 1:
        yield from _streaming_bulk(es, actions, cfg)
        return

    import queue as _queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # Strided slices spread any positional ordering in the partition evenly across workers instead of
    # front-loading one. Order does not matter: each action is independent.
    slices = [actions[i::n] for i in range(n)]
    results = _queue.Queue(maxsize=n * 2)   # bounded => producers block when full => flat memory
    stop = threading.Event()                # set when the consumer abandons us early (see docstring)
    _DONE = object()

    def _put(item):
        # Block for room on a full queue, but poll `stop` so a producer can't hang forever once the
        # consumer has stopped draining. Returns immediately when there is room (the normal path), so
        # this adds no latency unless the queue is actually full.
        while not stop.is_set():
            try:
                results.put(item, timeout=0.2)
                return
            except _queue.Full:
                continue

    def _worker(slice_actions):
        try:
            for tup in _streaming_bulk(es, slice_actions, cfg):
                if stop.is_set():
                    return
                _put(tup)
        finally:
            # Always signal completion, even on exception, so the consumer's drain loop terminates and
            # can re-raise via the future below. `_put` honors `stop`, so this can't wedge on abort.
            _put(_DONE)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_worker, s) for s in slices]
        try:
            finished = 0
            while finished < n:
                item = results.get()
                if item is _DONE:
                    finished += 1
                    continue
                yield item
            # Every worker has now put its _DONE (the loop above drained them), so result() re-raises
            # the first worker exception rather than blocking. Skipping this would let a thread that
            # died mid-stream drop its remaining docs while the partition reported a clean partial count.
            for f in futures:
                f.result()
        finally:
            # On ANY early exit (consumer raised, or the generator was closed/GC'd), release producers
            # that may be blocked in _put before ThreadPoolExecutor.__exit__ runs shutdown(wait=True):
            # set the flag, then drain so a parked producer wakes at once rather than after its poll
            # timeout. Harmless on the normal path (workers have already finished; the queue is empty).
            stop.set()
            try:
                while True:
                    results.get_nowait()
            except _queue.Empty:
                pass


def make_partition_writer(cfg: EsConfig):
    """Return a mapInPandas-compatible function that bulk-writes each pandas chunk.

    The returned fn yields a one-row pandas DataFrame with the counts, so the driver
    can sum results without collecting the data itself.
    """
    def _write(iterator: "Iterator") -> "Iterator":
        import pandas as pd
        from elasticsearch import Elasticsearch

        # Per-partition timing + id, always captured (two clock reads; negligible) and returned in
        # the summary row. _merge_partition_results ignores these columns; they are only consumed
        # when cfg.log_table is set (see build_log_rows). partition_id is the mapInPandas partition
        # index, so a log row is traceable to a specific task.
        _p_start = _time.time()
        try:
            from pyspark import TaskContext
            _tc = TaskContext.get()
            _pid = _tc.partitionId() if _tc is not None else -1
        except Exception:  # pragma: no cover - TaskContext always present under mapInPandas
            _pid = -1

        es = Elasticsearch(**cfg.client_kwargs())
        written = 0
        deleted = 0
        errors = 0
        ignored = 0                # delete-404 no-ops: expected, but must be COUNTED so the caller
                                   # can tell them apart from rows lost below the per-doc level
        coerced_nonfinite = 0      # values silently turned to JSON null (inf/-inf/NaN)
        total_input = 0            # rows fed in, so the caller can reconcile against the outcomes
        error_samples = []         # bounded diagnostics for failed docs (see ERROR_SAMPLE_CAP)
        for pdf in iterator:
            rows = pdf.to_dict("records")
            total_input += len(rows)
            _stats = {}
            actions = [
                build_action(
                    row,
                    index=cfg.index,
                    id_field=cfg.id_field,
                    drop_fields=cfg.drop_fields,
                    has_deletes=cfg.has_deletes,
                    delete_flag_column=cfg.delete_flag_column,
                    stats=_stats,
                )
                for row in rows
            ]
            coerced_nonfinite += _stats.get("coerced_nonfinite", 0)
            if not actions:
                continue
            # _iter_bulk_results yields one (ok, {op_type: item}) tuple per document (see
            # _streaming_bulk for the raise_on_error / yield_ok / retry rationale), so each result is
            # classified individually below. With cfg.write_concurrency > 1 the tuples arrive from
            # several concurrent streaming_bulk streams over this partition, merged in completion
            # order; the classification and counting are identical either way.
            for ok, result in _iter_bulk_results(es, actions, cfg):
                op_type, item = next(iter(result.items()))
                outcome = classify_bulk_result(ok, op_type, item.get("status", 500))
                if outcome == WRITTEN:
                    written += 1
                elif outcome == DELETED:
                    deleted += 1
                elif outcome == IGNORED:
                    # A delete-404 is an expected no-op, but it must be COUNTED: it is the one
                    # outcome that legitimately makes written+deleted+errors < total_input, so
                    # without this the reconciliation check cannot tell an expected no-op from a
                    # row lost below the per-doc level. Counting it here is what makes
                    # written+deleted+errors+ignored == total_input the invariant `unaccounted`
                    # measures. Note this is about a row producing NO write, not about two rows
                    # collapsing onto one `_id`: duplicate ids each report success, so they keep the
                    # identity intact and `unaccounted` stays 0 (see the README's duplicate-id note).
                    ignored += 1
                elif outcome == ERROR:
                    errors += 1
                    if len(error_samples) < ERROR_SAMPLE_CAP:
                        error_samples.append(_extract_error_sample(op_type, item))
        # error_samples is JSON-encoded into a single string column: mapInPandas needs a flat,
        # typed schema and can't carry a nested list<struct> of varying content cleanly.
        yield pd.DataFrame({
            "written": [written], "deleted": [deleted], "errors": [errors],
            "ignored": [ignored], "coerced_nonfinite": [coerced_nonfinite],
            "total_input": [total_input], "error_samples": [json.dumps(error_samples)],
            "partition_id": [int(_pid)], "partition_start_ms": [int(_p_start * 1000)],
            "partition_duration_ms": [int((_time.time() - _p_start) * 1000)],
        })

    return _write


# Columns of the optional write-log table (EsWriteConfig.log_table). One aggregate row per
# bulk_write call (scope="batch") plus one row per Spark partition (scope="partition"). Kept as a
# module constant so build_log_rows and _write_log agree on the exact shape.
LOG_COLUMNS = ("event_time", "batch_id", "index", "scope", "partition_id", "duration_ms",
               "total_input", "written", "deleted", "errors", "ignored", "unaccounted")


def build_log_rows(result: dict, partition_rows, event_time, batch_duration_ms: int,
                   batch_id, index: str) -> list:
    """Build the write-log rows for one bulk_write call. Pure (no Spark) so it is unit-testable.

    Returns a list of dicts keyed by LOG_COLUMNS: first the scope='batch' aggregate row (the merged
    result plus the whole-call event_time / duration_ms), then one scope='partition' row per entry
    in `partition_rows`. Each partition_row is a dict (call `.asDict()` on the collected mapInPandas
    Rows) carrying partition_id / partition_duration_ms and that partition's counts.

    `unaccounted` is populated only on the batch row: it is a cross-partition reconciliation derived
    at merge time, so it is not meaningful per partition (left None there). `batch_id` is None for a
    non-streaming write (bulk_write called without one) and the micro-batch id on the streaming path.
    """
    def _i(v):
        return int(v or 0)

    bid = int(batch_id) if batch_id is not None else None
    rows = [{
        "event_time": event_time, "batch_id": bid, "index": index, "scope": "batch",
        "partition_id": None, "duration_ms": int(batch_duration_ms),
        "total_input": _i(result.get("total_input")), "written": _i(result.get("written")),
        "deleted": _i(result.get("deleted")), "errors": _i(result.get("errors")),
        "ignored": _i(result.get("ignored")), "unaccounted": _i(result.get("unaccounted")),
    }]
    for pr in partition_rows:
        pid = pr.get("partition_id")
        rows.append({
            "event_time": event_time, "batch_id": bid, "index": index, "scope": "partition",
            "partition_id": (int(pid) if pid is not None else None),
            "duration_ms": _i(pr.get("partition_duration_ms")),
            "total_input": _i(pr.get("total_input")), "written": _i(pr.get("written")),
            "deleted": _i(pr.get("deleted")), "errors": _i(pr.get("errors")),
            "ignored": _i(pr.get("ignored")), "unaccounted": None,
        })
    return rows


def _write_log(spark, log_table: str, log_rows: list) -> None:
    """Append `log_rows` (from build_log_rows) to `log_table` as a Delta table (append, auto-created
    on first write). An explicit schema is used because a partition_id/unaccounted of None in every
    row of a batch would otherwise give Spark no type to infer. Spark-side; proven in the live tier.

    APPEND-ONLY event log: on the streaming RAISE path a failed micro-batch raises AFTER this log is
    written, Spark retries the SAME batch_id, and each retry appends another full set of rows (with a
    fresh event_time) -- intentional, so the failed attempts stay visible. Consumers therefore DEDUPE
    per batch_id, keeping the latest event_time (see the README's example queries). coalesce(1) keeps
    each commit to a single small file, since the source is a tiny driver-side list."""
    from pyspark.sql.types import (StructType, StructField, TimestampType, LongType,
                                   IntegerType, StringType)
    schema = StructType([
        StructField("event_time", TimestampType(), True),
        StructField("batch_id", LongType(), True),
        StructField("index", StringType(), True),
        StructField("scope", StringType(), True),
        StructField("partition_id", IntegerType(), True),
        StructField("duration_ms", LongType(), True),
        StructField("total_input", LongType(), True),
        StructField("written", LongType(), True),
        StructField("deleted", LongType(), True),
        StructField("errors", LongType(), True),
        StructField("ignored", LongType(), True),
        StructField("unaccounted", LongType(), True),
    ])
    spark.createDataFrame(log_rows, schema).coalesce(1).write.mode("append").saveAsTable(log_table)


def _merge_partition_results(rows) -> dict:
    """Combine the per-partition summary rows into the final result dict.

    Pure (no Spark) so it is unit-testable. Each row carries
    written/deleted/errors/ignored/coerced_nonfinite/total_input and a JSON string of that
    partition's bounded error samples. Counts are summed exactly; the sample lists are concatenated
    and re-capped at ERROR_SAMPLE_CAP so the driver result stays bounded even across many
    partitions.

    Also derives `unaccounted`: rows that produced no per-document outcome at all, i.e. loss BELOW
    the per-document level (a chunk-level transport/serialization failure), which the per-doc `errors`
    count structurally cannot see. `ignored` is part of the identity precisely so an expected
    delete-404 no-op does not masquerade as loss (see reconcile_or_raise).

    The discrepancy is computed PER PARTITION and split by sign, because the two signs mean opposite
    things and must not net against each other:

      - `unaccounted` (positive): input rows with no outcome. Real data loss.
      - `overcounted` (negative side): more outcomes than input rows. Structurally impossible, so it
        indicates a counting bug in THIS library, not a problem with the caller's data.

    Summing the raw counts and subtracting once at the end would let one cancel the other: a
    partition that lost 5 rows plus a partition that over-counted 5 nets to zero, and the write
    reports a clean success while 5 rows are gone. Both totals are now reported independently.
    """
    written = deleted = errors = ignored = coerced_nonfinite = total_input = 0
    unaccounted = overcounted = 0
    samples = []
    for r in rows:
        written += int(r["written"] or 0)
        deleted += int(r["deleted"] or 0)
        errors += int(r["errors"] or 0)
        # Optional keys are read with `in` rather than `.get()`: these rows are pyspark `Row`s in
        # production, and Row has no `.get` (it raises ATTRIBUTE_NOT_SUPPORTED). `in` checks field
        # names on a Row and keys on a dict, so it works for both. Tolerating absence means a
        # stale/foreign row shape degrades to a zero/empty rather than crashing the whole write here.
        _ignored = int((r["ignored"] if "ignored" in r else 0) or 0)
        ignored += _ignored
        coerced_nonfinite += int((r["coerced_nonfinite"] if "coerced_nonfinite" in r else 0) or 0)
        _total = int(r["total_input"] or 0)
        total_input += _total
        # Derive the discrepancy PER PARTITION and keep the two signs apart. Summing the counts and
        # subtracting once at the end lets a negative in one partition cancel a positive in another:
        # 100 rows in / 95 outcomes here (5 rows LOST) plus 100 in / 105 outcomes there (a counting
        # bug) totals to zero, and the write reports a clean success while 5 rows are gone. The two
        # mean opposite things and must never net against each other.
        _delta = _total - (int(r["written"] or 0) + int(r["deleted"] or 0)
                           + int(r["errors"] or 0) + _ignored)
        if _delta > 0:
            unaccounted += _delta          # rows that produced no per-doc outcome: real loss
        elif _delta < 0:
            overcounted += -_delta         # more outcomes than inputs: a bug in THIS library
        _samples_json = r["error_samples"] if "error_samples" in r else None
        if len(samples) < ERROR_SAMPLE_CAP and _samples_json:
            samples.extend(json.loads(_samples_json))
    return {
        "written": written, "deleted": deleted, "errors": errors, "ignored": ignored,
        "coerced_nonfinite": coerced_nonfinite,
        "total_input": total_input,
        "unaccounted": unaccounted,
        # Impossible-by-construction, so non-zero means a defect in this library rather than anything
        # wrong with the caller's data. Reported separately so it can be surfaced without being
        # allowed to mask `unaccounted` (see reconcile_or_raise).
        "overcounted": overcounted,
        "error_samples": samples[:ERROR_SAMPLE_CAP],
    }


class EsWriteError(RuntimeError):
    """A write did not fully succeed: Elasticsearch rejected documents, or rows went unaccounted for.

    Carries the full `bulk_write` result dict on `.result` so a caller catching this still has the
    counts and error samples for logging or a dead-letter path.
    """

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def reconcile_or_raise(result: dict, *, index: str = "") -> dict:
    """Raise EsWriteError if `result` shows rejected documents or unaccounted-for rows.

    Three independent failure signals, all of which a plain `written` count hides:
      - `errors > 0`: Elasticsearch rejected specific documents (mapping conflict, 429 after
        retries, ...). Each has a diagnostic in `error_samples`.
      - `unaccounted > 0`: rows that produced no per-document outcome at all, i.e. loss below the
        per-doc level. `ignored` (delete-404 no-ops) is already subtracted, so an expected no-op
        does not trip this.
    `overcounted > 0` (more per-document outcomes than input rows in some partition) is structurally
    impossible and means a counting bug in this library rather than anything wrong with the caller's
    data. It is LOGGED, never raised: raising would fail a healthy write, and on the streaming path
    that means an infinite retry loop on a batch that can never pass, with no escape but
    `on_error="log"` (which would also switch off real loss detection). So the inconsistency is
    surfaced without wedging a pipeline over a library defect.

    Crucially, `overcounted` does NOT suppress `unaccounted`: they are accumulated separately per
    partition, so an over-count can never cancel real loss and turn it into a clean verdict.

    Returns the result unchanged when the write was clean, so it can be used inline.
    """
    errors = int(result.get("errors", 0) or 0)
    unaccounted = int(result.get("unaccounted", 0) or 0)
    overcounted = int(result.get("overcounted", 0) or 0)
    # Tolerate a pre-0.6.1 result shape (no `overcounted` key) that carried the discrepancy as a
    # single signed `unaccounted`, so an older cached summary degrades instead of hiding a negative.
    if unaccounted < 0:
        overcounted += -unaccounted
        unaccounted = 0
    if overcounted:
        _log.error(
            "write to index %r reported %s more per-document outcomes than input rows "
            "(total_input=%s written=%s deleted=%s errors=%s ignored=%s). That is impossible and "
            "indicates an accounting bug in databricks-es-connector, not a problem with your data. "
            "The write itself is not failed over it; please report this result dict.",
            index, overcounted, result.get("total_input"), result.get("written"),
            result.get("deleted"), errors, result.get("ignored"))
    if not errors and unaccounted <= 0:
        return result

    where = f" to index {index!r}" if index else ""
    parts = []
    if errors:
        parts.append(f"{errors} document(s) rejected by Elasticsearch")
    if unaccounted > 0:
        parts.append(f"{unaccounted} row(s) unaccounted for (lost below the per-document level)")
    detail = (f" total_input={result.get('total_input')} written={result.get('written')} "
              f"deleted={result.get('deleted')} errors={errors} ignored={result.get('ignored')}")
    samples = result.get("error_samples") or []
    sample_text = f" first failures: {samples[:3]}" if samples else ""
    raise EsWriteError(f"write{where} did not fully succeed: {'; '.join(parts)}.{detail}.{sample_text}",
                       result)


def _preflight(df, cfg: EsConfig) -> None:
    """Driver-side checks that must fail CLOSED before any row is written.

    Every check here guards against a write that reports perfect success while doing the wrong
    thing, and all of them run ONCE on the driver (never inside the mapInPandas closure, which would
    put an HTTP round-trip on every executor):

      - every config field that NAMES A DATAFRAME COLUMN (`_COLUMN_NAMING_FIELDS`) must name a
        column that exists. This is the class; see the per-field notes below for what each one
        silently does otherwise.
      - `require_existing_index`: ES auto-creates a missing index, so a TYPO'd index name silently
        produces a new dynamically-mapped index and a clean `written` count. One `indices.exists`
        call turns that into an error.

    Why the driver is the ONLY place the column checks can live: below this layer a row is just a
    dict, and `row.get(name)` returns None both for a column that is absent and for a column that is
    present-but-null. Those two cases must behave DIFFERENTLY (a null flag is legitimately "not a
    delete"; an absent flag is a misconfiguration) and only `df.columns` can tell them apart.
    """
    # df.columns is cheap and safe here: sanitize_for_arrow has already removed the VARIANT
    # columns that make schema access throw on Spark Connect.
    present = set(df.columns)

    if cfg.id_field is not None and cfg.id_field not in present:
        # Without this, _require_id raises per-row on the executor mid-write, after earlier
        # partitions may already have committed their documents. One driver-side failure before any
        # write beats a partial write plus an opaque KeyError from inside mapInPandas.
        raise ValueError(
            f"id_field {cfg.id_field!r} is not a column in the DataFrame. "
            f"Available columns: {sorted(present)}. Every row needs this column to derive its "
            "deterministic _id; fix the name, or leave id_field unset to let Elasticsearch assign "
            "random ids (note: replays then duplicate instead of upserting).")

    if cfg.strict_drop_fields and cfg.drop_fields:
        unknown = [c for c in cfg.drop_fields if c not in present]
        if unknown:
            raise ValueError(
                f"drop_fields names not present in the DataFrame: {sorted(unknown)}. "
                f"Available columns: {sorted(present)}. A misspelled drop_fields entry prunes "
                "nothing and would ship the field to Elasticsearch anyway; fix the name, or set "
                "strict_drop_fields=False if the config is intentionally reused across schemas.")

    if cfg.has_deletes and cfg.delete_flag_column not in present:
        # The worst of this class, and the reason it was hardened: with a misspelled flag column
        # every row reads as "not flagged", so NO row is routed to a delete and every intended
        # deletion is applied as an upsert instead. Verified live: deleted=0, errors=0,
        # unaccounted=0, raise_on_error=True passing clean, and the documents that were supposed to
        # be erased still in the index -- with the flag column itself indexed alongside them.
        # Unconditional (no opt-out knob): has_deletes=True is meaningless without a real flag
        # column, so there is no legitimate configuration this rejects.
        raise ValueError(
            f"delete_flag_column {cfg.delete_flag_column!r} is not a column in the DataFrame. "
            f"Available columns: {sorted(present)}. With has_deletes=True every row would read as "
            "not-flagged, so each intended DELETE would silently be applied as an upsert and the "
            "documents would stay in Elasticsearch (deleted=0, errors=0, reconciliation clean). "
            "Fix the name, or set has_deletes=False if this write has no deletes.")

    if cfg.require_existing_index:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(**cfg.client_kwargs())
        try:
            exists = bool(es.indices.exists(index=cfg.index))
        finally:
            try:
                es.close()
            except Exception:
                pass
        if not exists:
            raise ValueError(
                f"index {cfg.index!r} does not exist. Elasticsearch would auto-create it with a "
                "dynamic mapping, so a misspelled index name looks like a successful write while "
                "the documents land somewhere nobody queries. Create the index (with an explicit "
                "mapping) first, or set require_existing_index=False to allow auto-creation.")


def bulk_write(df, cfg: EsConfig, *, raise_on_error: bool = False,
               batch_id: Optional[int] = None) -> dict:
    """Write a Spark DataFrame to Elasticsearch.

    Returns {'written', 'deleted', 'errors', 'ignored', 'coerced_nonfinite', 'total_input',
    'unaccounted', 'overcounted', 'error_samples'}:
      - 'written': index/upsert ops that succeeded.
      - 'deleted': successful delete-by-id ops (only non-zero when cfg.has_deletes).
      - 'errors': docs ES rejected (exact count).
      - 'ignored': delete-404 no-ops (deleting an already-absent doc: expected, not an error).
      - 'coerced_nonfinite': values (inf/-inf/NaN) that had to become JSON null to be sent at all.
        Non-zero means real numbers landed in ES as nulls, usually an upstream divide-by-zero.
      - 'total_input': rows handed to the writer.
      - 'unaccounted': input rows that produced none of those outcomes. Every row yields exactly one
        of them, so a positive value means rows were lost BELOW the per-document level (e.g. a
        chunk-level transport error) where the `errors` count cannot see them.
      - 'overcounted': the reverse discrepancy, more outcomes than input rows. Impossible by
        construction, so non-zero means a counting bug in this library, not a problem with the data.
        Reported and logged but not raised; kept separate from 'unaccounted' so an over-count in one
        partition can never cancel real loss in another.
      - 'error_samples': up to ERROR_SAMPLE_CAP diagnostics ({_id, op_type, status, reason}) for
        rejected docs, so a failure is actionable rather than an opaque count. Bounded, not a full
        dead-letter log.

    `raise_on_error=True` applies `reconcile_or_raise` to the result, raising EsWriteError when any
    document was rejected or any row went unaccounted for. It defaults to False here so a BATCH
    caller keeps full control of the result (and every shipped demo checks it explicitly), but note
    the streaming path defaults the other way: `make_foreach_batch` raises unless told not to,
    because there a swallowed error silently advances the checkpoint past the lost rows.

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
    # Preflight AFTER sanitize (so df.columns is safe to read) but BEFORE any row is written.
    _preflight(df, cfg)
    writer = make_partition_writer(cfg)
    # event_time is the write's START (driver clock); duration_ms times the whole distributed write.
    # Both feed the optional log_table (build_log_rows) and are captured whether or not it is set.
    event_time = _dt.datetime.now(_dt.timezone.utc)
    _t0 = _time.perf_counter()
    rows = df.mapInPandas(
        writer,
        "written long, deleted long, errors long, ignored long, coerced_nonfinite long, "
        "total_input long, error_samples string, "
        "partition_id int, partition_start_ms long, partition_duration_ms long",
    ).collect()
    duration_ms = int((_time.perf_counter() - _t0) * 1000)
    result = _merge_partition_results(rows)
    # Optional write log. BEST-EFFORT and BEFORE reconcile_or_raise, so a batch that ES rejected rows
    # in is still recorded (with its error counts) before the raise. A logging failure must never
    # fail the ES write itself, which has already happened by here, so it is caught and warned.
    if cfg.log_table:
        try:
            log_rows = build_log_rows(result, [r.asDict() for r in rows], event_time,
                                      duration_ms, batch_id, cfg.index)
            _write_log(df.sparkSession, cfg.log_table, log_rows)
        except Exception as _e:  # noqa: BLE001 - logging must not fail a successful write
            _log.warning("write-log append to %r failed (%s: %s); the ES write itself succeeded",
                         cfg.log_table, type(_e).__name__, _e)
    if raise_on_error:
        reconcile_or_raise(result, index=cfg.index)
    return result
