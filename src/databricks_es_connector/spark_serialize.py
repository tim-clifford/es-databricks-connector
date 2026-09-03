"""Spark-side (Catalyst) construction of the Elasticsearch `_bulk` NDJSON, for the
`serialize_in_spark` write path.

The default write path shapes and JSON-serializes every row IN PYTHON on the executor
(`transform.coerce_value` -> `build_action` -> `helpers.streaming_bulk` serializes). That per-row
work is GIL-bound and is the throughput ceiling on wide/large writes. This module moves it into
Spark: `build_ndjson` produces a one-column DataFrame whose single `_ndjson` column is the COMPLETE
`_bulk` action line for each row ("header\\nsource"), built with `to_json` in the JVM. The executor
writer (`bulk.make_ndjson_partition_writer`) then ships those lines with `es.bulk(operations=...)`
without any Python re-serialization.

Must run AFTER `sanitize_for_arrow` and `normalize_timestamps_for_utc` (exactly like the default
path): sanitize has already turned VARIANT/INTERVAL into JSON strings and normalize has turned every
`TimestampType` into an epoch-millis long, so by the time we get here `df.schema` is safe to read and
`to_json` sees only Arrow-friendly, already-normalized types.

Fidelity vs `coerce_value` (documented in the README "Spark-native serialization" section, and why
this path is opt-in): `to_json` is Spark's serializer, not the Python transform, so a few edge cases
differ (all verified live via a to_json rendering probe):
  - Non-finite floats: `to_json` renders NaN/inf as the quoted STRINGS "NaN"/"Infinity"/"-Infinity",
    which a numeric ES field rejects (mapper_parsing_exception). So this builder replaces them with
    null (top level) to match the default path's NaN/inf -> null; unlike that path it is NOT counted
    (`coerced_nonfinite` is always 0 in this mode). Nested non-finite floats are not reached here.
  - decimal: rendered at FULL precision (e.g. 1.000000000000000001), more faithful than the default
    path's decimal -> double.
  - float (32-bit): rendered as its short decimal repr (0.1), not the default path's exact widened
    double (0.10000000149...).
  - A null `id_field` value renders `"_id": null` (header also uses ignoreNullFields=false), which
    ES surfaces per-doc rather than the default path's whole-partition raise on a null id.
Everything else (nested structs/arrays/maps, binary as base64, timestamps-as-epoch, kept null fields)
matches.

pyspark is imported lazily inside the function so the pure config/transform layers stay importable
without Spark.
"""
from __future__ import annotations

from typing import List

from .config import EsConfig


def _payload_columns(columns, drop_fields) -> List[str]:
    """The columns that go into `_source`, in DataFrame order: everything except `drop_fields`.

    The `id_field` is deliberately KEPT (the document stays self-describing; `_id` is derived
    separately for the header), mirroring `transform.to_es_source`. Pure so it is unit-testable.
    """
    drop = set(drop_fields or ())
    return [c for c in columns if c not in drop]


def build_ndjson(df, cfg: EsConfig):
    """Return a one-column DataFrame (`_ndjson`) of complete `_bulk` action lines, built in Spark.

    Each row becomes "header\\nsource" with NO trailing newline (elastic_transport's NdjsonSerializer
    adds exactly one when the shipper forwards the line). Index/upsert only: `cfg.has_deletes` is
    rejected upstream in EsWriteConfig for this mode.

    Preconditions: `df` has been through `sanitize_for_arrow` + `normalize_timestamps_for_utc`, so
    `df.schema` is safe and types are Arrow-friendly / normalized.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, FloatType

    payload = _payload_columns(df.columns, cfg.drop_fields)
    field_types = {f.name: f.dataType for f in df.schema.fields}

    # Non-finite guard: turn NaN / +inf / -inf in top-level float/double columns into null, so
    # to_json cannot emit the bare NaN/Infinity tokens ES rejects (which would fail the WHOLE bulk
    # request, not just one doc). Nested non-finite floats are not reached here (documented limit).
    out = df
    for name in payload:
        dt = field_types.get(name)
        if isinstance(dt, (DoubleType, FloatType)):
            col = F.col(name)
            out = out.withColumn(name, F.when(
                col.isNull() | F.isnan(col) | (col == float("inf")) | (col == float("-inf")),
                F.lit(None).cast(dt)).otherwise(col))

    # ignoreNullFields=false keeps explicit null fields, matching the default path (coerce_value
    # keeps a null as JSON null rather than dropping the key).
    source = F.to_json(F.struct(*[F.col(c) for c in payload]), {"ignoreNullFields": "false"})

    # Bulk action header. index/upsert only; _id from id_field when set (else ES assigns one).
    # ignoreNullFields=false so a null id renders `"_id": null` (surfaced by ES) instead of being
    # dropped, which would silently fall back to an ES-assigned id (a replay-duplicate risk).
    index_meta = [F.lit(cfg.index).alias("_index")]
    if cfg.id_field:
        index_meta.append(F.col(cfg.id_field).cast("string").alias("_id"))
    header = F.to_json(F.struct(F.struct(*index_meta).alias("index")), {"ignoreNullFields": "false"})

    ndjson = F.concat(header, F.lit("\n"), source)
    return out.select(ndjson.alias("_ndjson"))
