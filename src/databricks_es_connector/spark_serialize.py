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
`to_json` sees only Arrow-friendly, already-normalized types. The other two temporal types,
`DateType` and `TimestampNTZType`, are NOT touched by `normalize_timestamps_for_utc` (on the default
path they cross Arrow as native date/datetime objects and `coerce_value` converts them); this path
never runs `coerce_value`, so `build_ndjson` converts them itself to the same epoch-millis long (see
`_rewrite_date_ntz`), or `to_json` would emit ISO strings and break the read_coerce round-trip.

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
Everything else (nested structs/arrays/map VALUES, binary as base64, timestamp/date/timestamp_ntz all
as epoch-millis, kept null fields) matches. The one carve-out is a map KEY of a non-string type
(temporal/decimal/binary): to_json stringifies it to its own form (a temporal key -> its ISO string)
rather than the default path's _coerce_key rendering (epoch-millis) -- see the MapType branch of
_rewrite_date_ntz. A map keyed by a raw temporal value is pathological; use string/int map keys.

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


def _type_has_date_or_ntz(dt) -> bool:
    """True if `dt` is a DateType/TimestampNTZType or contains one at any nesting depth.

    These are the two temporal types the shared `spark_prep.normalize_timestamps_for_utc` does NOT
    convert (it handles only TimestampType). On the default path they cross Arrow as native
    date/datetime objects and `transform.coerce_value` turns them into epoch-millis; the Spark path
    never runs coerce_value, so `build_ndjson` must convert them itself (see `_rewrite_date_ntz`).
    Pure logic (pyspark type objects only); mirrors spark_prep._type_has_timestamp.
    """
    from pyspark.sql.types import ArrayType, DateType, MapType, StructType, TimestampNTZType

    if isinstance(dt, (DateType, TimestampNTZType)):
        return True
    if isinstance(dt, StructType):
        return any(_type_has_date_or_ntz(f.dataType) for f in dt.fields)
    if isinstance(dt, ArrayType):
        return _type_has_date_or_ntz(dt.elementType)
    if isinstance(dt, MapType):
        # Map KEY intentionally NOT walked: keys are not temporally converted on this path (see the
        # MapType branch of _rewrite_date_ntz); only the VALUE side is rewritten.
        return _type_has_date_or_ntz(dt.valueType)
    return False


def _epoch_type(dt):
    """The post-rewrite type of `dt` with DateType/TimestampNTZType -> LongType (recursively), used to
    type a null struct literal so `when(null)` keeps the rewritten schema. Mirrors
    spark_prep._epoch_struct_type (TimestampType is already a Long by the time build_ndjson runs, so
    only date/ntz need mapping here)."""
    from pyspark.sql.types import (ArrayType, DateType, LongType, MapType, StructField, StructType,
                                   TimestampNTZType)

    if isinstance(dt, (DateType, TimestampNTZType)):
        return LongType()
    if isinstance(dt, StructType):
        return StructType([StructField(f.name, _epoch_type(f.dataType), f.nullable) for f in dt.fields])
    if isinstance(dt, ArrayType):
        return ArrayType(_epoch_type(dt.elementType), dt.containsNull)
    if isinstance(dt, MapType):
        # keyType left unchanged (map keys are not temporally rewritten), so this null-branch literal
        # type matches the rebuilt map, whose keys are untouched and only values converted.
        return MapType(dt.keyType, _epoch_type(dt.valueType), dt.valueContainsNull)
    return dt


def _date_ntz_leaf_to_epoch_millis(col, dt):
    """A LongType Column: the epoch-millis for a DateType or TimestampNTZType leaf, reproducing
    EXACTLY what transform._to_epoch_millis stores on the default path (so read_coerce inverts it):

      - DateType -> midnight-UTC epoch millis. `unix_date` is days-since-1970-01-01 (a calendar count,
        zone-free), so `* 86400000` is midnight UTC regardless of spark.sql.session.timeZone.
      - TimestampNTZType -> the zoneless wall-clock interpreted LITERALLY as UTC (no session tz, no
        DST). Rebuilt from the value's own wall-clock fields anchored explicitly at 'UTC' via
        make_timestamp, then `unix_millis` (which floors toward -inf, matching _to_epoch_millis'
        timedelta // for pre-epoch and sub-millisecond values). `date_part('SECOND', ...)` carries the
        sub-second fraction so it is floored, not truncated. Casting the ntz to a timestamp (or to a
        string then a timestamp) would localize the wall-clock with the session tz (verified wrong
        by exactly the session offset under Asia/Kolkata), so this path avoids any tz-bearing cast.

    A null date/ntz yields null (unix_date/make_timestamp propagate null), so nulls stay null.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DateType, TimestampNTZType

    if isinstance(dt, DateType):
        return F.unix_date(col).cast("long") * F.lit(86400000).cast("long")
    if isinstance(dt, TimestampNTZType):
        return F.unix_millis(F.make_timestamp(
            F.year(col), F.month(col), F.dayofmonth(col),
            F.hour(col), F.minute(col), F.date_part(F.lit("SECOND"), col),
            F.lit("UTC")))
    return col


def _rewrite_date_ntz(col, dt):
    """Return a Column that rebuilds `col` with every DateType/TimestampNTZType node replaced by its
    epoch-millis long (see `_date_ntz_leaf_to_epoch_millis`), at any nesting depth, preserving
    struct/array/map structure. Mirrors spark_prep._rewrite_timestamps; the caller only invokes it for
    columns that `_type_has_date_or_ntz`, and prunes clean subtrees below.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, DateType, MapType, StructType, TimestampNTZType

    if isinstance(dt, (DateType, TimestampNTZType)):
        return _date_ntz_leaf_to_epoch_millis(col, dt)
    if isinstance(dt, StructType):
        rebuilt = F.struct(*[
            (_rewrite_date_ntz(col[f.name], f.dataType) if _type_has_date_or_ntz(f.dataType)
             else col[f.name]).alias(f.name)
            for f in dt.fields
        ])
        return F.when(col.isNull(), F.lit(None).cast(_epoch_type(dt))).otherwise(rebuilt)  # keep null null
    if isinstance(dt, ArrayType):
        return F.transform(col, lambda e: _rewrite_date_ntz(e, dt.elementType))
    if isinstance(dt, MapType):
        # VALUES only. Map KEYS are deliberately NOT temporally converted on the serialize_in_spark
        # path: `to_json` stringifies a temporal map key to its ISO form, which differs from the
        # default path's `transform._coerce_key` (epoch-millis) -- a documented divergence (README).
        # Rewriting keys with F.transform_keys was tried and rejected: it (a) RAISES on two
        # sub-millisecond-distinct keys that floor to the same epoch (the default path silently keeps
        # the last), and (b) still would not cover TimestampType keys (spark_prep leaves map keys
        # untouched), so it cannot make the class consistent anyway. A map keyed by a raw temporal
        # value is pathological; use string/int map keys with serialize_in_spark for exact key parity.
        return F.transform_values(col, lambda k, v: _rewrite_date_ntz(v, dt.valueType))
    return col


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

    An index/upsert row becomes "header\\nsource"; a delete row (when `cfg.has_deletes` and its
    `delete_flag_column` is true) becomes just the delete-by-id "header" with NO source line. Neither
    has a trailing newline (elastic_transport's NdjsonSerializer adds exactly one when the shipper
    forwards the line). Deletes require `delete_flag_column` to be a real BooleanType column, enforced
    in bulk._preflight; the flag column itself is dropped from `_source` so it is never indexed.

    Preconditions: `df` has been through `sanitize_for_arrow` + `normalize_timestamps_for_utc`, so
    `df.schema` is safe and types are Arrow-friendly / normalized.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, FloatType

    # Drop the delete flag from _source alongside drop_fields: it is control data, not document data
    # (mirrors the default path's build_action, which adds delete_flag_column to its drop set).
    drop = tuple(cfg.drop_fields or ())
    if cfg.has_deletes and cfg.delete_flag_column:
        drop = drop + (cfg.delete_flag_column,)
    payload = _payload_columns(df.columns, drop)
    field_types = {f.name: f.dataType for f in df.schema.fields}

    # Build each payload column's _source expression from the ORIGINAL column, applying two rewrites
    # where the type calls for them (both preserve struct/array/map structure and field names, so they
    # compose on the same expression):
    #   1. date/ntz -> epoch-millis, matching the default path. The shared normalize_timestamps_for_utc
    #      only converts TimestampType (already a Long here); DateType/TimestampNTZType would otherwise
    #      reach to_json as ISO strings and break the read_coerce round-trip (which expects epoch-millis
    #      for those declared types). Runs FIRST so the float guard below sees the rewritten structure.
    #   2. non-finite floats -> null at any depth, so to_json never emits a "NaN"/"Infinity" value a
    #      numeric ES field would reject (matches the default path's NaN/inf -> null).
    def _source_expr(name):
        dt = field_types.get(name)
        c = F.col(name)
        if _type_has_date_or_ntz(dt):
            c = _rewrite_date_ntz(c, dt)
        if _type_has_float(dt):
            c = _null_nonfinite(c, dt)
        return c.alias(name)

    # ignoreNullFields=false keeps explicit null fields, matching the default path (coerce_value
    # keeps a null as JSON null rather than dropping the key).
    source = F.to_json(F.struct(*[_source_expr(c) for c in payload]), {"ignoreNullFields": "false"})

    # Index/upsert action header; _id from id_field when set (else ES assigns one).
    index_meta = [F.lit(cfg.index).alias("_index")]
    id_col = None
    if cfg.id_field:
        # Guard the id column for non-finite SEPARATELY from `payload`: id_field may be in
        # drop_fields (excluded from payload, so the loop above never guards it), and a NaN/±inf id
        # would otherwise cast to the string "NaN" -- not null -- evading the fail-closed check below
        # and colliding every non-finite id onto one _id. Turning it to null here routes it into that
        # check. Only float/double ids can be non-finite; other id types pass through unchanged.
        # id_col reads the ORIGINAL column (the _source rewrites above are built as separate
        # expressions, not applied to df), so a date/ntz id_field renders its raw calendar/wall-clock
        # string here (a human-readable value close to the default path's str(value)) rather than the
        # epoch-millis we store in _source. Only its _source copy is epoch; the _id stays the
        # human-readable value. The exact string is Spark's cast, which is NOT guaranteed identical to
        # Python str() for a non-string id -- see the fail-closed note below.
        id_dt = field_types.get(cfg.id_field)
        id_col = F.col(cfg.id_field)
        if isinstance(id_dt, (DoubleType, FloatType)):
            id_col = _null_nonfinite(id_col, id_dt)
        index_meta.append(id_col.cast("string").alias("_id"))
    index_header = F.to_json(F.struct(F.struct(*index_meta).alias("index")), {"ignoreNullFields": "false"})
    index_line = F.concat(index_header, F.lit("\n"), source)

    if cfg.has_deletes:
        # Delete-by-id action: id-only, NO source line. has_deletes requires id_field (config guard),
        # so id_col is always set here. A row whose delete_flag_column is true routes to the delete
        # line; every other row indexes. `flag === true` is null-safe: a null flag yields null (not
        # true), so it falls through to the index line -- matching the default path's "a null flag is
        # not a delete". delete_flag_column is a real BooleanType (bulk._preflight enforces it), so no
        # string parsing happens here; there is no Catalyst equivalent of the per-row
        # AmbiguousDeleteFlag raise, which is why the boolean type is required at this seam.
        delete_meta = [F.lit(cfg.index).alias("_index"), id_col.cast("string").alias("_id")]
        delete_header = F.to_json(F.struct(F.struct(*delete_meta).alias("delete")),
                                  {"ignoreNullFields": "false"})
        ndjson = F.when(F.col(cfg.delete_flag_column) == F.lit(True), delete_header).otherwise(index_line)
    else:
        ndjson = index_line

    # Fail CLOSED on a null (or non-finite, nulled above) id value: emit a null action line. The
    # writer (make_ndjson_partition_writer) RAISES on a null line, failing the write unconditionally
    # -- mirroring the default path's _require_id, which raises regardless of raise_on_error -- rather
    # than shipping `"_id": null` (ES might auto-assign a random id and duplicate the row on replay).
    # Applies to delete rows too: a delete needs a non-null _id to target, so a null-id delete fails
    # closed the same way. Note: a NON-STRING id_field is rendered here by Spark `cast(string)`, which
    # is NOT guaranteed to match the default path's Python `str()`: a float/decimal id can differ (e.g.
    # scientific notation), and a sub-second timestamp/timestamp_ntz id can differ in trailing-zero/
    # fraction rendering. Only a STRING id_field is byte-identical across the two paths; use one if you
    # mix the two write paths for the same data and rely on _id equality. (The _source epoch-millis DO
    # match across paths -- this caveat is about the human-readable _id only.)
    if cfg.id_field:
        ndjson = F.when(id_col.isNull(), F.lit(None).cast("string")).otherwise(ndjson)
    return df.select(ndjson.alias("_ndjson"))
