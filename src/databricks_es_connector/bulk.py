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
from .transform import build_action


def make_partition_writer(cfg: EsConfig):
    """Return a mapInPandas-compatible function that bulk-writes each pandas chunk.

    The returned fn yields a one-row pandas DataFrame with the count written,
    so the driver can sum results without collecting the data itself.
    """
    def _write(iterator: "Iterator") -> "Iterator":
        import pandas as pd
        from elasticsearch import Elasticsearch, helpers

        es = Elasticsearch(**cfg.client_kwargs())
        total = 0
        errors = 0
        for pdf in iterator:
            actions = [
                build_action(
                    row,
                    index=cfg.index,
                    id_field=cfg.id_field,
                    drop_fields=cfg.drop_fields,
                )
                for row in pdf.to_dict("records")
            ]
            if not actions:
                continue
            # raise_on_error=False so one bad doc doesn't abort the whole partition;
            # we count failures and surface them so the batch can fail loudly if needed.
            ok, errs = helpers.bulk(
                es, actions, chunk_size=cfg.chunk_size, raise_on_error=False
            )
            total += ok
            errors += len(errs) if errs else 0
        yield pd.DataFrame({"written": [total], "errors": [errors]})

    return _write


def bulk_write(df, cfg: EsConfig) -> dict:
    """Write a Spark DataFrame to Elasticsearch. Returns {'written', 'errors'}.

    Batch entry point. For streaming, use stream.make_foreach_batch.
    """
    writer = make_partition_writer(cfg)
    result = (
        df.mapInPandas(writer, "written long, errors long")
        .groupBy()
        .sum("written", "errors")
        .collect()
    )
    if not result:
        return {"written": 0, "errors": 0}
    row = result[0]
    return {
        "written": int(row["sum(written)"] or 0),
        "errors": int(row["sum(errors)"] or 0),
    }
