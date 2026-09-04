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
differ (rendering verified live via a to_json probe):
  - Non-finite floats: `to_json` renders NaN/inf as the quoted STRINGS "NaN"/"Infinity"/"-Infinity",
    which a numeric ES field rejects. So this builder replaces them with null at ANY nesting depth
    (recursive walk over struct/array/map, mirroring spark_prep._rewrite_timestamps) to match the
    default path's NaN/inf -> null; unlike that path it is NOT counted (`coerced_nonfinite` is always
    0 in this mode).
  - decimal: rendered at FULL precision (e.g. 1.000000000000000001), more faithful than the default
    path's decimal -> double.
  - float (32-bit): rendered as its short decimal repr (0.1), not the default path's exact widened
    double (0.10000000149...).
  - A null or non-finite `id_field` value FAILS CLOSED: the row's action line is emitted as null and
    make_ndjson_partition_writer RAISES on it, failing the write UNCONDITIONALLY (like the default
    path's _require_id, which raises regardless of raise_on_error) -- rather than shipping
    `"_id": null` and trusting ES not to auto-assign a random id (which would duplicate on replay).
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


def _type_has_float(dt) -> bool:
    """True if `dt` is a Float/Double or contains one at any nesting depth (struct/array/map).

    Pure logic (only touches pyspark type objects), so build_ndjson only walks columns that can
    actually carry a non-finite float. Mirrors spark_prep._type_has_timestamp.
    """
    from pyspark.sql.types import ArrayType, DoubleType, FloatType, MapType, StructType

    if isinstance(dt, (FloatType, DoubleType)):
        return True
    if isinstance(dt, StructType):
        return any(_type_has_float(f.dataType) for f in dt.fields)
    if isinstance(dt, ArrayType):
        return _type_has_float(dt.elementType)
    if isinstance(dt, MapType):
        return _type_has_float(dt.keyType) or _type_has_float(dt.valueType)
    return False


def _null_nonfinite(col, dt):
    """Return a Column that rebuilds `col` with every non-finite float (NaN/±inf) replaced by null,
    at any nesting depth, preserving struct/array/map structure. `to_json` renders a non-finite float
    as the string "NaN"/"Infinity" (a numeric ES field rejects it), so nulling it here matches the
    default path's NaN/inf -> null for nested values too. Mirrors spark_prep._rewrite_timestamps.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, DoubleType, FloatType, MapType, StructType

    if isinstance(dt, (FloatType, DoubleType)):
        # A null float has isnan()/==inf evaluate to null, so `when` falls through to otherwise(col)
        # and a genuine null stays null; only actual NaN/±inf become null.
        return F.when(F.isnan(col) | (col == float("inf")) | (col == float("-inf")),
                      F.lit(None).cast(dt)).otherwise(col)
    if isinstance(dt, StructType):
        rebuilt = F.struct(*[
            (_null_nonfinite(col[f.name], f.dataType) if _type_has_float(f.dataType)
             else col[f.name]).alias(f.name)
            for f in dt.fields
        ])
        return F.when(col.isNull(), col).otherwise(rebuilt)   # keep a null struct null
    if isinstance(dt, ArrayType):
        return F.transform(col, lambda e: _null_nonfinite(e, dt.elementType))
    if isinstance(dt, MapType):
        # Values only: a map KEY cannot be nulled (a null map key is invalid / collapses the entry),
        # so a non-finite float KEY is left as-is and to_json renders it as the string "NaN"/"Infinity".
        # This is a documented limitation (README): a map keyed by a raw float is pathological anyway;
        # use string/int keys. Finite float keys are unaffected.
        return F.transform_values(col, lambda k, v: _null_nonfinite(v, dt.valueType))
    return col


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

    # Non-finite guard at ANY depth: NaN/±inf floats -> null so to_json never emits a "NaN"/
    # "Infinity" value that a numeric ES field would reject. Only walk columns that can carry a float
    # (top-level or nested); everything else is left untouched.
    out = df
    for name in payload:
        dt = field_types.get(name)
        if _type_has_float(dt):
            out = out.withColumn(name, _null_nonfinite(F.col(name), dt))

    # ignoreNullFields=false keeps explicit null fields, matching the default path (coerce_value
    # keeps a null as JSON null rather than dropping the key).
    source = F.to_json(F.struct(*[F.col(c) for c in payload]), {"ignoreNullFields": "false"})

    # Bulk action header. index/upsert only; _id from id_field when set (else ES assigns one).
    index_meta = [F.lit(cfg.index).alias("_index")]
    id_col = None
    if cfg.id_field:
        # Guard the id column for non-finite SEPARATELY from `payload`: id_field may be in
        # drop_fields (excluded from payload, so the loop above never guards it), and a NaN/±inf id
        # would otherwise cast to the string "NaN" -- not null -- evading the fail-closed check below
        # and colliding every non-finite id onto one _id. Turning it to null here routes it into that
        # check. Only float/double ids can be non-finite; other id types pass through unchanged.
        id_dt = field_types.get(cfg.id_field)
        id_col = F.col(cfg.id_field)
        if isinstance(id_dt, (DoubleType, FloatType)):
            id_col = _null_nonfinite(id_col, id_dt)
        index_meta.append(id_col.cast("string").alias("_id"))
    header = F.to_json(F.struct(F.struct(*index_meta).alias("index")), {"ignoreNullFields": "false"})

    ndjson = F.concat(header, F.lit("\n"), source)
    # Fail CLOSED on a null (or non-finite, nulled above) id value: emit a null action line. The
    # writer (make_ndjson_partition_writer) RAISES on a null line, failing the write unconditionally
    # -- mirroring the default path's _require_id, which raises regardless of raise_on_error -- rather
    # than shipping `"_id": null` (ES might auto-assign a random id and duplicate the row on replay).
    # Note: a numeric id_field is rendered here by Spark `cast(string)`, which can differ from the
    # default path's Python `str()` for float/decimal ids (e.g. scientific notation); use a string id
    # if you mix the two write paths for the same data and rely on _id equality.
    if cfg.id_field:
        ndjson = F.when(id_col.isNull(), F.lit(None).cast("string")).otherwise(ndjson)
    return out.select(ndjson.alias("_ndjson"))
