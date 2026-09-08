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
import logging
from typing import Iterator

from .config import EsConfig
from .spark_prep import sanitize_for_arrow, normalize_timestamps_for_utc

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
    """Pull a compact, JSON-safe diagnostic from one failed _bulk response item.

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
    """Classify one _bulk response item into WRITTEN / DELETED / IGNORED / ERROR.

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


# --- ship pre-built NDJSON, classify the _bulk response -------------------------------------------
# spark_serialize.build_ndjson builds the whole `_bulk` action line in Catalyst (index/upsert
# "header\nsource", or a delete "header" with no source), so this writer does no per-row Python
# shaping or JSON encoding: it batches the pre-built lines and hands them to es.bulk(operations=...).
# elastic_transport's NdjsonSerializer forwards str/bytes list items VERBATIM (utf-8 + a trailing
# newline, no json re-encode), so the JVM-built JSON is never re-serialized in Python -- that
# pass-through is the whole point, it is what keeps the work off the GIL.


def iter_bulk_response_outcomes(items):
    """Yield (op_type, body, ok, outcome) for each item of an es.bulk() response.

    Pure (no ES, no Spark) so the classification is unit-testable against canned responses. An item
    is `{op_type: body}` with `body["status"]` the per-document HTTP status (and `body["error"]` when
    it failed); `ok` is a 2xx. Reuses `classify_bulk_result` for the WRITTEN/DELETED/IGNORED/ERROR
    rules (including the delete-404 -> IGNORED suppression).
    """
    for item in items:
        # A well-formed item is {op_type: body}. Guard an empty/malformed item explicitly: ES never
        # returns {}, but `next(iter({}.items()))` would raise StopIteration, which PEP-479 turns into
        # a RuntimeError inside this generator and aborts the ENTIRE mapInPandas partition. Fail that
        # one document closed (ERROR) instead, so a single bad item can't take the partition down.
        pair = next(iter(item.items()), None)
        if pair is None:
            yield "unknown", {}, False, ERROR
            continue
        op_type, body = pair
        status = int(body.get("status", 500) or 500)
        ok = 200 <= status < 300
        yield op_type, body, ok, classify_bulk_result(ok, op_type, status)


def _ship_ndjson_chunk(es, lines, cfg: EsConfig, counts: dict, error_samples: list) -> None:
    """Ship one chunk of pre-built NDJSON action lines and tally the outcomes into `counts`.

    `lines` is a list where each element is ONE row's action ("header\\nsource"), so es.bulk returns
    one response item per element, in order, and a retryable item maps back to its line by index.
    Implements a per-document retry the connector treats as load-bearing: the _bulk API answers HTTP
    200 even when items inside it fail, so transport-level retries never cover a 429'd document. Only
    items whose status is in cfg.retry_on_doc_status are retried, up to cfg.max_retries_per_doc, with
    exponential backoff; everything else is tallied immediately.
    """
    import time as _t

    # Fast path (GIL avoidance): when writes are idempotent (id_field set) and there are no deletes,
    # a clean chunk needs only the top-level `errors` flag, not per-item detail. Ship with
    # filter_path="errors" so Elasticsearch returns just {"errors": false} on success: the per-item
    # response array is never sent or decoded, removing the O(chunk_size) Python decode + classify
    # that runs holding the GIL and so serializes write_concurrency threads within a worker process.
    # On ANY failure (errors true, or the flag missing -> fail closed) fall through to the full path
    # below, which re-ships the SAME chunk without filter_path and runs the normal classify + 429
    # retry. Re-shipping is correct ONLY because every op is an idempotent upsert (id_field set, no
    # deletes): a doc that already succeeded on the probe is simply upserted again. Deletes are
    # excluded because a delete-404 is IGNORED (not written), so `errors: false` would not justify
    # counting the whole chunk as written; auto-id writes (id_field is None) are excluded because a
    # re-ship would duplicate the docs the probe already wrote.
    if cfg.id_field is not None and not cfg.has_deletes:
        try:
            resp = es.bulk(operations=list(lines), filter_path="errors")
        except Exception as _e:  # noqa: BLE001
            # Same fail-closed handling as the full path: a whole-request transport failure counts
            # every line as an error rather than aborting the partition.
            counts["errors"] += len(lines)
            if len(error_samples) < ERROR_SAMPLE_CAP:
                error_samples.append({"_id": None, "op_type": "bulk",
                                      "status": None, "reason": f"{type(_e).__name__}: {_e}"[:300]})
            return
        if resp.get("errors", True) is False:
            counts["written"] += len(lines)
            return
        # errors true or the flag absent: re-ship in full below to get per-item detail and retry.

    pending = list(lines)
    attempt = 0
    while pending:
        try:
            resp = es.bulk(operations=pending)
        except Exception as _e:  # noqa: BLE001
            # A whole-request transport failure (persistent 429/503, dropped connection) survived the
            # client's transport_max_retries. Record it rather than letting it abort the partition:
            # count every still-pending line as an ERROR (fail closed, surfaced via reconcile), instead
            # of propagating and failing the whole mapInPandas partition on one chunk.
            counts["errors"] += len(pending)
            if len(error_samples) < ERROR_SAMPLE_CAP:
                error_samples.append({"_id": None, "op_type": "bulk",
                                      "status": None, "reason": f"{type(_e).__name__}: {_e}"[:300]})
            return
        items = resp.get("items", []) if isinstance(resp, dict) else resp["items"]
        retry_lines = []
        for idx, (op_type, body, ok, outcome) in enumerate(iter_bulk_response_outcomes(items)):
            status = int(body.get("status", 500) or 500)
            if (not ok and status in cfg.retry_on_doc_status
                    and attempt < cfg.max_retries_per_doc):
                retry_lines.append(pending[idx])
                continue
            if outcome == WRITTEN:
                counts["written"] += 1
            elif outcome == DELETED:
                counts["deleted"] += 1
            elif outcome == IGNORED:
                counts["ignored"] += 1
            else:
                counts["errors"] += 1
                if len(error_samples) < ERROR_SAMPLE_CAP:
                    error_samples.append(_extract_error_sample(op_type, body))
        if not retry_lines:
            return
        attempt += 1
        # Exponential backoff: initial_backoff (2s) * 2**(attempt-1) => 2s, 4s, 8s, capped at 30s.
        _t.sleep(min(2 ** attempt, 30))
        pending = retry_lines


def _ship_ndjson_lines(es, lines, cfg: EsConfig, counts: dict, error_samples: list, pool=None) -> None:
    """Ship all of one partition-batch's pre-built NDJSON action `lines`, chunked by cfg.chunk_size,
    tallying into `counts` / `error_samples`.

    `cfg.write_concurrency == 1` (default) chunks and ships serially, no threads. `> 1` fans the batch
    across that many worker threads, each shipping its OWN strided slice with its OWN
    `_ship_ndjson_chunk` calls, so `write_concurrency` bulk requests are in flight at once to fill the
    ES round-trip wait. Without this a partition shipped its chunks one blocking es.bulk at a time --
    the last serial bottleneck, since the per-row work is already off the GIL (built in Catalyst).
    Strided slices (`lines[i::n]`) spread any positional ordering evenly across workers; order does not
    matter, each action is independent. Each worker tallies into a PRIVATE counts dict + sample list
    (no shared-state lock), merged here after the pool joins.

    `pool` (optional) is a caller-owned ThreadPoolExecutor reused across every batch of the partition,
    so thread spin-up is paid once per partition instead of once per Arrow batch (see
    make_ndjson_partition_writer). When omitted, a local pool is created and torn down for this call --
    the behavior direct callers/tests rely on. Either way the futures are joined here before returning,
    so shipping stays bounded to the current batch and the null-id check upstream still runs before
    any worker dispatches.

    A worker exception is re-raised on this thread after join (`f.result()`), so a partial write FAILS
    the partition rather than silently reporting the docs a dead worker never sent as a clean success
    -- the exact silent loss this module exists to prevent. (`_ship_ndjson_chunk` itself catches
    transport errors and counts them rather than raising, so in practice a worker raises only on a
    programming error, but the guard is kept as a backstop.)
    """
    n = cfg.write_concurrency
    if n <= 1:
        for i in range(0, len(lines), cfg.chunk_size):
            _ship_ndjson_chunk(es, lines[i:i + cfg.chunk_size], cfg, counts, error_samples)
        return

    slices = [lines[i::n] for i in range(n)]
    # One private (counts, samples) pair per worker, so threads never touch shared state; merged below.
    partials = [({"written": 0, "deleted": 0, "ignored": 0, "errors": 0}, []) for _ in range(n)]

    def _worker(idx):
        local_counts, local_samples = partials[idx]
        sl = slices[idx]
        for i in range(0, len(sl), cfg.chunk_size):
            _ship_ndjson_chunk(es, sl[i:i + cfg.chunk_size], cfg, local_counts, local_samples)

    if pool is not None:
        # Reuse the partition-scoped pool; join before returning so this batch fully ships first.
        futures = [pool.submit(_worker, i) for i in range(n)]
        for f in futures:
            f.result()   # re-raise the first worker exception; a partial write must fail the partition
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n) as local_pool:
            futures = [local_pool.submit(_worker, i) for i in range(n)]
            for f in futures:
                f.result()   # re-raise the first worker exception; a partial write must fail the partition

    for local_counts, local_samples in partials:
        for k in counts:
            counts[k] += local_counts[k]
        # Keep the merged sample list bounded exactly as the serial path does (ERROR_SAMPLE_CAP total).
        if len(error_samples) < ERROR_SAMPLE_CAP and local_samples:
            error_samples.extend(local_samples[:ERROR_SAMPLE_CAP - len(error_samples)])


def make_ndjson_partition_writer(cfg: EsConfig):
    """mapInPandas writer for the write path. Input has a single `_ndjson` column, one
    pre-built action line per row (see spark_serialize.build_ndjson). Yields the per-partition summary
    schema _merge_partition_results / reconcile_or_raise consume. `coerced_nonfinite` is always 0:
    non-finite floats are turned to null in Spark (build_ndjson), not counted per row. A null action
    line (build_ndjson's signal for a null/non-finite id) RAISES here, failing the write
    unconditionally, rather than being counted as `unaccounted` (which would only surface under
    raise_on_error=True). Shipping is delegated to `_ship_ndjson_lines`, which fans each batch across
    `cfg.write_concurrency` worker threads (1 = serial) using a single pool reused for the whole
    partition (created once here, not per Arrow batch).
    """
    def _write(iterator: "Iterator") -> "Iterator":
        import pandas as pd
        from elasticsearch import Elasticsearch

        es = Elasticsearch(**cfg.client_kwargs())
        counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
        total_input = 0
        error_samples = []
        # One thread pool for the whole PARTITION, not one per Arrow batch. mapInPandas hands a
        # partition to this closure as a stream of ~maxRecordsPerBatch-row batches; creating the pool
        # here (once) and reusing it for every batch pays thread spin-up once per partition instead of
        # once per batch. write_concurrency <= 1 stays threadless (pool = None). Shipping still joins
        # per batch inside _ship_ndjson_lines, so peak memory stays bounded to one batch.
        pool = None
        if cfg.write_concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=cfg.write_concurrency)
        try:
            for pdf in iterator:
                col = pdf["_ndjson"]
                total_input += len(col)
                # A null action line means build_ndjson hit a null/non-finite id (its only null-line
                # source). RAISE here, failing the partition (and so the whole write) loudly and
                # UNCONDITIONALLY. Do NOT merely count it as `unaccounted`: that only surfaces via
                # reconcile_or_raise, which the batch default (raise_on_error=False) skips, so a null
                # id would silently drop. Checked BEFORE dispatch so a null id fails the write before
                # any worker ships. pandas renders a null object cell as None OR float NaN depending on
                # dtype; Series.isna() catches BOTH in one C-level pass, so there is no per-row Python
                # loop on the hot path (the last per-row Python cost the 0.9.0 Catalyst path left).
                if col.isna().any():
                    raise ValueError(
                        "build_ndjson produced a null action line: the id_field value is null or "
                        "non-finite (NaN/inf) in at least one row. Every row needs a non-null, finite "
                        "id. Fix the id column, or leave id_field unset to let Elasticsearch assign ids.")
                lines = col.tolist()   # C-level conversion; no Python per-row iteration
                if lines:
                    _ship_ndjson_lines(es, lines, cfg, counts, error_samples, pool=pool)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        yield pd.DataFrame({
            "written": [counts["written"]], "deleted": [counts["deleted"]],
            "errors": [counts["errors"]], "ignored": [counts["ignored"]],
            "coerced_nonfinite": [0], "total_input": [total_input],
            "error_samples": [json.dumps(error_samples)],
        })

    return _write


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
        # Without this, build_ndjson would reference a missing column and the write would fail deep
        # inside mapInPandas, after earlier partitions may already have committed their documents. One
        # driver-side failure before any write beats a partial write plus an opaque error.
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

    if cfg.has_deletes and cfg.delete_flag_column in present:
        # Deletes are routed in Catalyst with `flag === true` (spark_serialize.build_ndjson), which
        # requires a real BooleanType column: Catalyst has no per-row equivalent of a raise on an
        # ambiguous flag, so a string/int flag cannot be parsed at the seam. A non-boolean flag would
        # make `flag === true` evaluate to null for every row (string==boolean is a null-yielding type
        # mismatch), so NO row would route to a delete and every intended deletion would silently
        # become an upsert -- the exact silent loss the column-name check just above exists to prevent.
        # Enforce the type on the driver instead. df.schema is safe here (sanitize_for_arrow already
        # removed the VARIANT columns that make it throw on Spark Connect); typeName() avoids importing
        # a pyspark type into this module.
        flag_dt = next((f.dataType for f in df.schema.fields if f.name == cfg.delete_flag_column), None)
        if flag_dt is None or flag_dt.typeName() != "boolean":
            raise ValueError(
                f"delete_flag_column {cfg.delete_flag_column!r} must be a boolean column, but its "
                f"type is {flag_dt.simpleString() if flag_dt is not None else 'unknown'}. Deletes are "
                "routed in Spark via `flag === true`, which has no way to parse a string/int flag; a "
                "non-boolean flag would route NO row to a delete and silently upsert every intended "
                "deletion. Cast the column to boolean in Spark (e.g. "
                "df.withColumn(col, col.cast('boolean'))) so the intent is unambiguous.")

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


def bulk_write(df, cfg: EsConfig, *, raise_on_error: bool = False) -> dict:
    """Write a Spark DataFrame to Elasticsearch.

    Returns {'written', 'deleted', 'errors', 'ignored', 'coerced_nonfinite', 'total_input',
    'unaccounted', 'overcounted', 'error_samples'}:
      - 'written': index/upsert ops that succeeded.
      - 'deleted': successful delete-by-id ops (only non-zero when cfg.has_deletes).
      - 'errors': docs ES rejected (exact count).
      - 'ignored': delete-404 no-ops (deleting an already-absent doc: expected, not an error).
      - 'coerced_nonfinite': always 0. Non-finite floats (inf/-inf/NaN) are turned to JSON null in
        Spark (build_ndjson) so ES accepts the document, but they are not counted (to_json runs in the
        JVM, with no per-row Python step to count them). Kept in the result for shape stability; a
        caller needing that signal can pre-count in Spark.
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
    summary_schema = ("written long, deleted long, errors long, ignored long, "
                      "coerced_nonfinite long, total_input long, error_samples string")
    # Build the whole `_bulk` action line in Catalyst (JVM) via build_ndjson, then ship the pre-built
    # NDJSON with no per-row Python shaping/serialization (make_ndjson_partition_writer, fanned across
    # write_concurrency worker threads). This is the only write path.
    from .spark_serialize import build_ndjson
    nd = build_ndjson(df, cfg)
    writer = make_ndjson_partition_writer(cfg)
    rows = nd.mapInPandas(writer, summary_schema).collect()
    result = _merge_partition_results(rows)
    if raise_on_error:
        reconcile_or_raise(result, index=cfg.index)
    return result
