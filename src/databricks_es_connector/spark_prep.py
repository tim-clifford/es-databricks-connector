"""Spark-side prep for types that cannot cross Arrow into mapInPandas.

Most Spark types survive the Arrow -> pandas conversion inside `bulk_write` and are made
JSON-safe by `transform.coerce_value`. A few types fail Arrow conversion *before* any Python
code runs, so they cannot be fixed on the executor — they must be cast in Spark first:

  - INTERVAL (YearMonth and DayTime). Year-month intervals raise
    `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION` outright.

`cast_unsupported_to_string(df)` rewrites those columns to their string form so a caller can
export 100% of a table's columns without hand-casting each one. It only touches the Arrow-
hostile columns; everything else is left exactly as-is (and handled downstream by coerce_value).

This module imports pyspark lazily (inside the function) so the pure-Python transform/config
layers stay importable without Spark for local unit testing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


def cast_unsupported_to_string(df: "DataFrame") -> "DataFrame":
    """Return `df` with Arrow-incompatible columns cast to string.

    Currently targets INTERVAL types (year-month / day-time), which Spark cannot convert to
    Arrow and therefore cannot pass to `mapInPandas`. Nested occurrences (a struct/array/map
    field whose element type is an interval) are also cast, via a recursive schema walk that
    rebuilds the column with `to_json`/`cast` only where needed.

    All other columns are returned unchanged. Idempotent: a DataFrame with no interval columns
    comes back untouched.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DayTimeIntervalType,
        YearMonthIntervalType,
        StructType,
        ArrayType,
        MapType,
    )

    def _has_interval(dt) -> bool:
        if isinstance(dt, (DayTimeIntervalType, YearMonthIntervalType)):
            return True
        if isinstance(dt, StructType):
            return any(_has_interval(f.dataType) for f in dt.fields)
        if isinstance(dt, ArrayType):
            return _has_interval(dt.elementType)
        if isinstance(dt, MapType):
            return _has_interval(dt.keyType) or _has_interval(dt.valueType)
        return False

    out = df
    for field in df.schema.fields:
        dt = field.dataType
        if isinstance(dt, (DayTimeIntervalType, YearMonthIntervalType)):
            # Top-level interval column: a plain cast to string is well-defined and readable.
            out = out.withColumn(field.name, F.col(field.name).cast("string"))
        elif _has_interval(dt):
            # Interval nested inside a struct/array/map: no per-field cast expression is
            # practical, so serialize the whole column to a JSON string. This preserves the
            # data (nothing dropped) at the cost of that column landing as a JSON string in ES.
            out = out.withColumn(field.name, F.to_json(F.col(field.name)))
    return out
