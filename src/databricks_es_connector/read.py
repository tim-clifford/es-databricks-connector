"""Read an Elasticsearch index into a Spark DataFrame.

v0.4.0 spike: the **driver-side** reader (Option A in READ_DESIGN.md). Opens a Point-in-Time,
pages through the index with `search_after`, coerces each hit against the caller's declared schema
via `read_transform.read_coerce`, and builds a DataFrame with `spark.createDataFrame`.

This path is NOT distributed — every hit crosses the driver — so it is for lookups / reference data
/ bounded result sets, and it is the validation harness for the shared coercion layer. The
distributed sliced-scroll `mapInPandas` reader (Option B) will reuse `read_transform` unchanged;
only the transport differs.

The caller MUST pass an explicit Spark `StructType`: several write transforms are one-way and cannot
be inverted from `_source` alone (see READ_DESIGN.md). pyspark is imported lazily so the pure-Python
layers stay importable without Spark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .config import EsConfig
from .read_transform import read_coerce

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.types import DataType, StructType


@dataclass(frozen=True)
class EsReadConfig:
    """Read-only knobs, kept separate from the (write) EsConfig so the write surface is untouched.
    Frozen + serializable so it can ship to executors when the distributed reader lands."""
    query: Optional[dict] = None          # ES query DSL; None => match_all
    num_slices: Optional[int] = None      # distributed reader parallelism (unused by the driver path)
    batch_size: int = 1000                # docs per page
    pit_keep_alive: str = "1m"            # Point-in-Time lifetime
    include_id: bool = True               # expose ES _id when the schema declares the id_field

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("EsReadConfig.batch_size must be positive")


def _spark_type_token(dtype: "DataType") -> str:
    """Render a pyspark DataType to the lowercase token read_coerce understands.

    Spark's own `simpleString()` already produces exactly the DDL form we parse (`timestamp`,
    `decimal(10,2)`, `struct<name:type,...>`, `array<...>`, `map<...>`), so we delegate to it.
    Kept as a seam so the coercion layer never imports pyspark.
    """
    return dtype.simpleString()


def _schema_field_tokens(schema: "StructType"):
    """[(field_name, type_token), ...] for the top-level fields of the declared schema."""
    return [(f.name, _spark_type_token(f.dataType)) for f in schema.fields]


def _coerce_hit(source: dict, doc_id, field_tokens, id_field: Optional[str], include_id: bool):
    """Turn one ES hit (_source + _id) into a row dict matching the declared schema."""
    row = {}
    for name, token in field_tokens:
        if include_id and id_field is not None and name == id_field and name not in source:
            # The id lives in _source by default (writes keep it), but honor _id if _source omitted it.
            row[name] = read_coerce(doc_id, token)
        else:
            row[name] = read_coerce(source.get(name), token)
    return row


def _scroll_hits(es, index: str, query: dict, batch_size: int, keep_alive: str):
    """Yield raw hits ({_id, _source}) across the whole index via a PIT + search_after.

    A Point-in-Time gives a consistent snapshot even if the index changes mid-read; search_after
    pages without the deep-pagination limits of from/size. The PIT is always closed in finally.
    """
    pit = es.open_point_in_time(index=index, keep_alive=keep_alive)
    pit_id = pit["id"]
    try:
        search_after = None
        while True:
            body = {
                "size": batch_size,
                "query": query,
                "pit": {"id": pit_id, "keep_alive": keep_alive},
                # A deterministic tiebreak sort is required for search_after; _shard_doc is the
                # cheapest total order and is only valid with a PIT.
                "sort": [{"_shard_doc": "asc"}],
            }
            if search_after is not None:
                body["search_after"] = search_after
            resp = es.search(body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                yield h
            search_after = hits[-1]["sort"]
            pit_id = resp.get("pit_id", pit_id)   # PIT id can be refreshed across pages
    finally:
        try:
            es.close_point_in_time(body={"id": pit_id})
        except Exception:
            pass


def read_index(
    spark: "SparkSession",
    cfg: EsConfig,
    schema: "StructType",
    read: Optional[EsReadConfig] = None,
) -> "DataFrame":
    """Read an ES index into a Spark DataFrame against the caller-declared `schema` (required).

    Driver-side (Option A): pulls all matching hits through the driver, coerces each to `schema`
    via the documented inverse transforms, and returns `spark.createDataFrame(rows, schema)`. Use
    for bounded reads; the distributed reader is future work (see READ_DESIGN.md).
    """
    from pyspark.sql.types import StructType

    if not isinstance(schema, StructType) or not schema.fields:
        raise ValueError("read_index requires a non-empty Spark StructType schema (v0.4.0 has no "
                         "mapping inference — declare the schema explicitly)")
    if not cfg.index:
        raise ValueError("EsConfig.index is required to read")

    read = read or EsReadConfig()
    query = read.query or {"match_all": {}}
    field_tokens = _schema_field_tokens(schema)

    from elasticsearch import Elasticsearch
    es = Elasticsearch(**cfg.client_kwargs())

    rows = [
        _coerce_hit(h.get("_source", {}), h.get("_id"), field_tokens, cfg.id_field, read.include_id)
        for h in _scroll_hits(es, cfg.index, query, read.batch_size, read.pit_keep_alive)
    ]
    return spark.createDataFrame(rows, schema)
