"""Executor-side bulk write to Elasticsearch, serverless-safe via mapInPandas.

Why mapInPandas and not foreachPartition: serverless compute blocks RDD APIs
(df.rdd / foreachPartition raise INSUFFICIENT_PERMISSIONS). mapInPandas is the
supported way to run per-partition code on serverless, and it parallelizes the
bulk write across executors — throughput scales with the cluster like the old
Spark connector did.

The Elasticsearch client is built INSIDE the partition function from EsConfig,
so nothing non-serializable is captured on the driver.

Change Data Feed (CDC): when cfg.change_feed is True the input is a Delta CDF. bulk_write
repartitions by id_field so all changes for a record co-locate, then each partition collapses
the changes to one net action per id (see transform.collapse_cdf_changes) and emits mixed
index/delete bulk actions. See EsConfig for the semantics.
"""
from __future__ import annotations

from typing import Iterator

from .config import EsConfig
from .transform import build_action, collapse_cdf_changes, CDF_METADATA_FIELDS


def make_partition_writer(cfg: EsConfig):
    """Return a mapInPandas-compatible function that bulk-writes each pandas chunk.

    The returned fn yields a one-row pandas DataFrame with (written, deleted, errors),
    so the driver can sum results without collecting the data itself.
    """
    def _write(iterator: "Iterator") -> "Iterator":
        import pandas as pd
        from elasticsearch import Elasticsearch, helpers

        es = Elasticsearch(**cfg.client_kwargs())
        written = 0
        deleted = 0
        errors = 0
        drop = tuple(cfg.drop_fields) + (CDF_METADATA_FIELDS if cfg.change_feed else ())

        for pdf in iterator:
            rows = pdf.to_dict("records")

            if cfg.change_feed:
                # Collapse this partition's changes to one net action per id, then route each
                # to index (upsert) or delete. Co-location by id_field happens upstream in
                # bulk_write's repartition, so all changes for a given id are in this partition.
                collapsed = collapse_cdf_changes(
                    rows,
                    id_field=cfg.id_field,
                    change_type_field=cfg.change_type_field,
                    commit_version_field=cfg.commit_version_field,
                )
                actions = [
                    build_action(row, index=cfg.index, id_field=cfg.id_field,
                                 drop_fields=drop, delete=is_delete)
                    for row, is_delete in collapsed
                ]
            else:
                actions = [
                    build_action(row, index=cfg.index, id_field=cfg.id_field,
                                 drop_fields=drop)
                    for row in rows
                ]

            if not actions:
                continue

            # raise_on_error=False so one bad doc doesn't abort the whole partition;
            # we count failures and surface them so the batch can fail loudly if needed.
            _, results = helpers.bulk(
                es, actions, chunk_size=cfg.chunk_size, raise_on_error=False,
                stats_only=False,
            )
            # Tally per-op outcomes. A delete of an already-absent doc returns result
            # "not_found" — that is idempotently fine (the desired end state holds), not an error.
            for item in _iter_bulk_items(actions, results):
                op, ok, res = item
                if ok:
                    deleted += 1 if op == "delete" else 0
                    written += 0 if op == "delete" else 1
                elif op == "delete" and res == "not_found":
                    deleted += 1          # already gone => success
                else:
                    errors += 1

        yield pd.DataFrame({"written": [written], "deleted": [deleted], "errors": [errors]})

    return _write


def _iter_bulk_items(actions, results):
    """Yield (op_type, ok: bool, result_str) per action from helpers.bulk output.

    helpers.bulk(stats_only=False) returns (n_success, errors_list) where errors_list holds
    only the FAILED items. To classify every action (incl. delete not_found), we can't rely on
    that alone, so we re-derive: successes aren't itemized, so we instead read per-item results
    when available. In practice elasticsearch-py returns the full response only for errors, so
    we treat any action whose id appears in the error list as failed and the rest as ok.
    """
    # Map failed items by (op, _id) -> result string.
    failed = {}
    for err in results or []:
        # err looks like {op_type: {"_id":..., "status":..., "result":..., "error":...}}
        for op_type, info in err.items():
            key = (op_type, str(info.get("_id")))
            failed[key] = info.get("result") or info.get("error", {}).get("type") or "error"
    for a in actions:
        op = a.get("_op_type", "index")
        _id = str(a.get("_id"))
        key = (op, _id)
        if key in failed:
            yield op, False, failed[key]
        else:
            yield op, True, "ok"


def bulk_write(df, cfg: EsConfig) -> dict:
    """Write a Spark DataFrame to Elasticsearch. Returns {'written', 'deleted', 'errors'}.

    Batch entry point. For streaming, use stream.make_foreach_batch.

    When cfg.change_feed is True, `df` is treated as a Delta Change Data Feed and is
    repartitioned by id_field first so every change for a record lands in one partition
    (required for correct per-id collapse). 'deleted' counts CDF delete ops (0 in plain mode).
    """
    writer = make_partition_writer(cfg)

    src = df
    if cfg.change_feed:
        # Co-locate all changes for the same id so collapse_cdf_changes sees them together.
        src = df.repartition(cfg.id_field)

    result = (
        src.mapInPandas(writer, "written long, deleted long, errors long")
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
