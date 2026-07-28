"""Spark-side prep for types that cannot cross Arrow into mapInPandas.

Most Spark types survive the Arrow -> pandas conversion inside `bulk_write` and are made
JSON-safe by `transform.coerce_value`. A few types fail Arrow conversion *before* any Python
code runs, so they cannot be fixed on the executor — they must be rewritten in Spark first:

  - VARIANT. Arrow has no VARIANT type. A column whose type contains VARIANT at ANY nesting
    depth (e.g. `variant`, `struct<...,v:variant>`, `array<struct<...,v:variant>>`) cannot be
    passed to `mapInPandas` and raises `[UNSUPPORTED_OPERATION] data type ... is not supported`.
  - INTERVAL (YearMonth and DayTime). Year-month intervals raise
    `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION`.

`sanitize_for_arrow(df)` rewrites those columns to a string so the whole DataFrame can be exported,
picking the serialization by type: a struct/array/map/variant goes through `to_json` (a JSON
string), while a scalar (top-level) INTERVAL is `cast(... as string)` — because Spark's `to_json`
REJECTS a scalar interval (`[DATATYPE_MISMATCH.INVALID_JSON_SCHEMA] Input schema must be a struct,
an array, a map or a variant`). It is called automatically by `bulk_write`, so callers do NOT need
to pre-process anything — any valid Spark DataFrame just works. Only Arrow-hostile columns are
touched; every other column is left exactly as-is (and handled downstream by `coerce_value`).

Serverless / Spark Connect constraint (this is why the implementation looks the way it does):
On Spark Connect, ANY schema accessor that builds Python type objects throws on a VARIANT column
— `df.schema`, `df.schema.jsonValue()`, `df.dtypes`, and even `df.columns` all raise
`UNSUPPORTED_OPERATION`. The ONLY way to read the schema of a VARIANT-bearing DataFrame is a SQL
`DESCRIBE` over a temp view. So we register the DataFrame as a temp view and DESCRIBE it to get
the column type strings, rather than touching `df.schema`.

Because DESCRIBE gives us top-level column *type strings* (not a navigable typed schema), a column
with a hostile type nested inside a struct/array is serialized *whole* to a JSON string — we cannot
surgically cast just the inner field. Data is preserved; the column lands in ES as a string.

This module imports pyspark lazily (inside functions) so the pure-Python transform/config layers
stay importable without Spark for local unit testing.
"""
from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

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
#   - it must NOT be immediately followed by ':' — a field name is always `name:type`, so a token
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

    Why this matters for serialization: Spark's `to_json` REJECTS a scalar interval —
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

    Uses DESCRIBE over a temp view — the only schema read that survives a VARIANT column on Spark
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
