"""Spark-side prep for types that cannot cross Arrow into mapInPandas.

Most Spark types survive the Arrow -> pandas conversion inside `bulk_write` and are made
JSON-safe by `transform.coerce_value`. A few types fail Arrow conversion *before* any Python
code runs, so they cannot be fixed on the executor: they must be rewritten in Spark first:

  - VARIANT. Arrow has no VARIANT type. A column whose type contains VARIANT at ANY nesting
    depth (e.g. `variant`, `struct<...,v:variant>`, `array<struct<...,v:variant>>`) cannot be
    passed to `mapInPandas` and raises `[UNSUPPORTED_OPERATION] data type ... is not supported`.
  - INTERVAL (YearMonth and DayTime). Year-month intervals raise
    `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION`.

`sanitize_for_arrow(df)` rewrites those columns to a string so the whole DataFrame can be exported,
picking the serialization by type: a struct/array/map/variant goes through `to_json` (a JSON
string), while a scalar (top-level) INTERVAL is `cast(... as string)`, because Spark's `to_json`
REJECTS a scalar interval (`[DATATYPE_MISMATCH.INVALID_JSON_SCHEMA] Input schema must be a struct,
an array, a map or a variant`). It is called automatically by `bulk_write`, so callers do NOT need
to pre-process anything, any valid Spark DataFrame just works. Only Arrow-hostile columns are
touched; every other column is left exactly as-is (and handled downstream by `coerce_value`).

Serverless / Spark Connect constraint (this is why the implementation looks the way it does):
On Spark Connect, ANY schema accessor that builds Python type objects throws on a VARIANT column:
`df.schema`, `df.schema.jsonValue()`, `df.dtypes`, and even `df.columns` all raise
`UNSUPPORTED_OPERATION`. The ONLY way to read the schema of a VARIANT-bearing DataFrame is a SQL
`DESCRIBE` over a temp view. So we register the DataFrame as a temp view and DESCRIBE it to get
the column type strings, rather than touching `df.schema`.

Because DESCRIBE gives us top-level column *type strings* (not a navigable typed schema), a column
with a hostile type nested inside a struct/array is serialized *whole* to a JSON string, we cannot
surgically cast just the inner field. Data is preserved; the column lands in ES as a string.

This module imports pyspark lazily (inside functions) so the pure-Python transform/config layers
stay importable without Spark for local unit testing.
"""
from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame
    from pyspark.sql.types import DataType

# Arrow-hostile type names to find inside a column's DESCRIBE type string. VARIANT/INTERVAL have no
# Arrow representation, so a column whose type CONTAINS one (top-level or nested) must be
# JSON-serialized before export.
#
# The match is in TYPE POSITION, not a bare substring. A DESCRIBE type string interleaves type
# names with struct FIELD NAMES (`struct<field_name:field_type,...>`), so a plain `"variant" in
# text` check false-positives on innocuous field names like `polling_interval` or `invariant_score`
# and would silently JSON-stringify a valid struct/array column. So:
#   - the token must be bounded by non-identifier chars (not part of a longer word like
#     `polling_interval` / `invariant`), and
#   - it must NOT be immediately followed by ':', a field name is always `name:type`, so a token
#     followed by ':' is a field name, whereas a real type is followed by '>', ',', whitespace, or
#     end of string.
# Types appear at start-of-string or after '<' ':' ',' (array element / struct field type / map
# key-value), all of which are non-identifier chars, so the left boundary needs no special casing.
_ARROW_HOSTILE_RE = re.compile(r"(?<![a-z0-9_])(?:variant|interval)(?![a-z0-9_:])")


def _type_is_arrow_hostile(type_text: str) -> bool:
    """True if a DESCRIBE type string contains VARIANT or INTERVAL in type position (any depth)."""
    return bool(_ARROW_HOSTILE_RE.search((type_text or "").lower()))


def _is_scalar_interval_type(type_text: str) -> bool:
    """True if the column's TOP-LEVEL type is itself an interval (not a struct/array/map that merely
    contains one). Scalar interval type strings begin with `interval` (e.g. `interval day to second`,
    `interval year to month`).

    Why this matters for serialization: Spark's `to_json` REJECTS a scalar interval,
    `to_json(<interval>)` raises `[DATATYPE_MISMATCH.INVALID_JSON_SCHEMA] Input schema must be a
    struct, an array, a map or a variant`. So a top-level interval must be serialized with
    `cast(... as string)` instead. A struct/array/map that *contains* an interval has a top-level
    type of `struct<...>` / `array<...>` / `map<...>`, which `to_json` accepts, so those still go
    through `to_json`.
    """
    return (type_text or "").strip().lower().startswith("interval")


def _hostile_columns_from_describe(describe_rows) -> List[Tuple[str, str]]:
    """Pure logic: given DESCRIBE output rows (each a mapping with 'col_name'/'data_type'), return
    (col_name, data_type) pairs for top-level columns whose type text contains an Arrow-hostile type.

    Returns the type text alongside the name because the caller must decide serialization per column
    (scalar interval -> cast string; everything else -> to_json; see sanitize_for_arrow).

    Split out from the Spark call so it is unit-testable without a session. DESCRIBE lists nested
    subfields and partition/metadata rows after the column list; the column rows come first and end
    at the first blank or '#'-prefixed row (e.g. '# Partition Information'), so we stop there.
    """
    hostile = []
    for r in describe_rows:
        col = r["col_name"]
        if not col or col.startswith("#"):
            break  # end of the column section
        if _type_is_arrow_hostile(r["data_type"]):
            hostile.append((col, r["data_type"]))
    return hostile


def _hostile_columns(df: "DataFrame") -> List[Tuple[str, str]]:
    """Return (name, type_text) for top-level columns whose type contains an Arrow-hostile type.

    Uses DESCRIBE over a temp view, the only schema read that survives a VARIANT column on Spark
    Connect (see module docstring).
    """
    view = "_es_connector_sanitize_" + uuid.uuid4().hex
    df.createOrReplaceTempView(view)
    try:
        rows = df.sparkSession.sql(f"DESCRIBE `{view}`").collect()
    finally:
        df.sparkSession.catalog.dropTempView(view)
    return _hostile_columns_from_describe(rows)


def sanitize_for_arrow(df: "DataFrame") -> "DataFrame":
    """Return `df` with every Arrow-incompatible column (VARIANT / INTERVAL, at any depth) rewritten
    to a string, so the DataFrame can be exported through `mapInPandas`.

    Serialization is per-column by type:
      - a scalar (top-level) INTERVAL -> `cast(... as string)` (Spark's `to_json` rejects intervals),
        e.g. `INTERVAL '1 02:03:04' DAY TO SECOND`.
      - everything else hostile (VARIANT at any depth, or a struct/array/map containing one)
        -> `to_json`, landing as a JSON string.

    Called automatically by `bulk_write`; callers do not need to invoke it. Idempotent: a DataFrame
    with no hostile columns is returned unchanged. A column serialized here lands in Elasticsearch
    as a string (map it as `keyword`/`text`, not `object`).
    """
    from pyspark.sql import functions as F

    out = df
    for name, type_text in _hostile_columns(df):
        if _is_scalar_interval_type(type_text):
            out = out.withColumn(name, F.col(name).cast("string"))
        else:
            out = out.withColumn(name, F.to_json(F.col(name)))
    return out


# --- timestamp -> epoch-millis in Spark (timezone-safe) --------------------------------------
# Why this exists: Spark's Arrow export converts a `TimestampType` (an instant) to a naive pandas
# Timestamp using `spark.sql.session.timeZone`, i.e. the session-LOCAL wall-clock with the zone
# dropped. Under any non-UTC session, `transform.coerce_value` (which reads a naive datetime as
# UTC) then computes an epoch offset by the session's UTC offset -- silent timestamp corruption,
# present since 0.1.0 and only surfaced by running the datatype test under a non-UTC session.
#
# Fix, mirroring how the elasticsearch-hadoop connector stays tz-safe: convert every `TimestampType`
# to its epoch-millis long IN SPARK via `unix_millis`, which operates on the instant and is therefore
# independent of session.timeZone (verified on serverless under UTC / America/New_York /
# Asia/Kolkata; correct for pre-epoch and sub-millisecond flooring too). The value then reaches the
# executor already an integer, so coerce_value passes it straight through and the READ path is
# unchanged (it already reconstructs a `timestamp` from epoch-millis).
#
# Deliberately NOT touched:
#   - TimestampNTZType: a zoneless wall-clock. The connector's contract is to read it AS UTC, which
#     is exactly what the current path already produces (verified). `unix_millis` also REJECTS ntz.
#   - DateType: has no time-of-day, converts to midnight-UTC epoch correctly already (verified).
# Only `TimestampType` at any nesting depth is rewritten.


def _type_has_timestamp(dt: "DataType") -> bool:
    """True if `dt` is a TimestampType or contains one at any nesting depth (struct/array/map).

    TimestampNTZType is intentionally NOT matched (different, already-correct handling)."""
    from pyspark.sql.types import (ArrayType, MapType, StructType, TimestampType)

    if isinstance(dt, TimestampType):
        return True
    if isinstance(dt, StructType):
        return any(_type_has_timestamp(f.dataType) for f in dt.fields)
    if isinstance(dt, ArrayType):
        return _type_has_timestamp(dt.elementType)
    if isinstance(dt, MapType):
        # A timestamp map KEY can't occur in JSON output distinctly, but rewrite defensively so the
        # value side is always handled; keys are coerced to strings downstream regardless.
        return _type_has_timestamp(dt.keyType) or _type_has_timestamp(dt.valueType)
    return False


def _rewrite_timestamps(col: "Any", dt: "DataType") -> "Any":
    """Return a Spark Column expression that rebuilds `col` with every TimestampType node replaced by
    its `unix_millis` epoch-millis long, preserving struct/array/map structure. Non-timestamp leaves
    and whole subtrees with no timestamp are returned unchanged (the caller only invokes this for
    columns that _type_has_timestamp, and prunes clean subtrees below)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (ArrayType, MapType, StructType, TimestampType)

    if isinstance(dt, TimestampType):
        return F.unix_millis(col)
    if isinstance(dt, StructType):
        # Rebuild the struct field-by-field; convert only fields that contain a timestamp, keep the
        # rest as-is (and preserve field names/order). A null struct stays null: guard with when().
        rebuilt = F.struct(*[
            (_rewrite_timestamps(col[f.name], f.dataType) if _type_has_timestamp(f.dataType)
             else col[f.name]).alias(f.name)
            for f in dt.fields
        ])
        return F.when(col.isNull(), F.lit(None).cast(_epoch_struct_type(dt))).otherwise(rebuilt)
    if isinstance(dt, ArrayType):
        return F.transform(col, lambda e: _rewrite_timestamps(e, dt.elementType))
    if isinstance(dt, MapType):
        return F.transform_values(col, lambda k, v: _rewrite_timestamps(v, dt.valueType))
    return col


def _epoch_struct_type(dt: "DataType") -> "DataType":
    """The post-rewrite type of `dt` (TimestampType -> LongType, recursively), used to type a null
    struct literal so `when(null)` keeps the column's rewritten schema instead of collapsing it."""
    from pyspark.sql.types import (ArrayType, LongType, MapType, StructField, StructType,
                                   TimestampType)

    if isinstance(dt, TimestampType):
        return LongType()
    if isinstance(dt, StructType):
        return StructType([StructField(f.name, _epoch_struct_type(f.dataType), f.nullable)
                           for f in dt.fields])
    if isinstance(dt, ArrayType):
        return ArrayType(_epoch_struct_type(dt.elementType), dt.containsNull)
    if isinstance(dt, MapType):
        return MapType(_epoch_struct_type(dt.keyType), _epoch_struct_type(dt.valueType),
                       dt.valueContainsNull)
    return dt


def normalize_timestamps_for_utc(df: "DataFrame") -> "DataFrame":
    """Return `df` with every `TimestampType` column (at any nesting depth) converted to its
    epoch-millis long via `unix_millis`, so the epoch is correct regardless of
    `spark.sql.session.timeZone`. See the section comment above for the why.

    Does NOT mutate session state. Leaves TimestampNTZType, DateType, and all other types unchanged.
    Idempotent-ish: a DataFrame with no TimestampType columns is returned unchanged. Must run AFTER
    sanitize_for_arrow (VARIANT columns break df.schema on Spark Connect; sanitize removes them
    first), which is how bulk_write orders the two.
    """
    out = df
    for f in df.schema.fields:
        if _type_has_timestamp(f.dataType):
            from pyspark.sql import functions as F
            out = out.withColumn(f.name, _rewrite_timestamps(F.col(f.name), f.dataType))
    return out
