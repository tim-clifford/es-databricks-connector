"""Pure-Python row shaping for Elasticsearch. No Spark, no ES client — unit-testable.

`coerce_value` makes any value that `mapInPandas` can hand us JSON/ES-serializable, so
EVERY exportable column lands as usable data (field pruning via `drop_fields` is a client
opt-out to shrink payload — never the reason a column can't be exported). It handles every
Spark type that survives Arrow -> pandas conversion:
  - ES `date` fields reject str(pandas.Timestamp) (no 'T'); send epoch-millis instead.
  - Naive datetimes are treated as UTC so epochs are deterministic across executors
    (a naive value's local-tz epoch would differ per worker).
  - pandas nulls (NaT for datetimes, NaN for numerics, pd.NA) must become JSON null:
    NaT.timestamp() raises, and NaN serializes to the token `NaN` which ES rejects.
  - numpy scalars/arrays (from Arrow) -> Python scalars / lists.
  - Decimal -> float (numeric + queryable in ES; note: >~15-17 sig-fig precision is lost).
  - bytes/bytearray (binary) -> base64 str (reversible, ES-safe).
  - Nested structs/maps arrive as dicts and lists pass through as JSON objects/arrays.
  - Total fallback: any other non-JSON-native object -> str(v), so an unforeseen type can
    never silently pass through and crash `helpers.bulk` (it lands as a string, not a crash).

NOT handled here (cannot be): types that fail Spark's Arrow conversion BEFORE reaching this
code — notably INTERVAL (year-month intervals raise UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION).
Cast those to string in Spark first; see `spark_prep.cast_unsupported_to_string`.
"""
from __future__ import annotations

import base64 as _base64
import datetime as _dt
from decimal import Decimal as _Decimal
from typing import Any, Iterable, Optional


def _is_null(v: Any) -> bool:
    """True for pandas nulls (NaT, pd.NA) and float NaN, without importing pandas.

    - float('nan') != itself.
    - pandas.NaT / pd.NA are not floats but also compare unequal to themselves,
      so the `v != v` identity catches them too; guarded to avoid arrays/objects
      whose __ne__ returns non-bool.
    """
    if v is None:
        return True
    # pandas.NA: `NA != NA` returns NA, and bool(NA) raises — identify it by type name
    # (avoids a hard pandas import in this pure-Python module).
    if type(v).__name__ in ("NAType", "NaTType"):
        return True
    try:
        return bool(v != v)   # float NaN (and NaT) satisfy v != v with a real bool
    except (ValueError, TypeError):
        return False          # e.g. numpy arrays — not a scalar null


def _to_epoch_millis(v: Any) -> int:
    """Coerce a datetime/date to epoch milliseconds. Naive values are assumed UTC."""
    if isinstance(v, _dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=_dt.timezone.utc)
        return int(v.timestamp() * 1000)
    # date without time -> midnight UTC
    d = _dt.datetime(v.year, v.month, v.day, tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


def coerce_value(v: Any) -> Any:
    """Recursively make a value JSON/ES-serializable.

    - nulls (None, NaN, NaT, pd.NA) -> None
    - bool -> unchanged (before int/Decimal checks; bool is an int subclass)
    - datetimes/dates -> epoch millis (naive treated as UTC)
    - dict/list/tuple -> recurse (preserves nested structs/arrays)
    - numpy scalars/arrays -> Python scalars/lists (then recurse)
    - Decimal -> float
    - bytes/bytearray -> base64 str
    - int/float/str -> unchanged (JSON-native)
    - anything else -> str(v)  (total fallback: never let an unknown type crash the write)
    """
    if _is_null(v):                                   # must precede datetime: NaT is a datetime
        return None
    if isinstance(v, bool):                           # bool is a subclass of int — keep as-is
        return v
    if isinstance(v, (_dt.datetime, _dt.date)):
        return _to_epoch_millis(v)
    if isinstance(v, dict):
        return {k: coerce_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [coerce_value(x) for x in v]
    # numpy values from Arrow / mapInPandas: array<...> columns arrive as np.ndarray
    # and numeric columns as numpy scalars (np.int64, np.float64, ...). Neither is
    # JSON-serializable and neither is caught by the list/dict branches above, so a
    # DataFrame with an array field (array<struct>) would break helpers.bulk.
    # .tolist() converts an ndarray to a (possibly nested) Python list and a numpy
    # scalar to a plain Python scalar; recurse on containers so inner values coerce too.
    if type(v).__module__ == "numpy" and hasattr(v, "tolist"):
        converted = v.tolist()
        return coerce_value(converted) if isinstance(converted, (list, dict)) else converted
    if isinstance(v, _Decimal):
        # float => numeric + queryable in ES. Precision beyond ~15-17 sig figs is lost;
        # cast the source column to string in Spark first if exactness is required.
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        return _base64.b64encode(bytes(v)).decode("ascii")
    if isinstance(v, (int, float, str)):              # JSON-native scalars
        return v
    # Total fallback: an unforeseen type (e.g. a Spark-side object that slipped through
    # Arrow) becomes its string form rather than crashing helpers.bulk downstream.
    return str(v)


def to_es_source(
    row: dict,
    *,
    id_field: Optional[str] = None,
    drop_fields: Iterable[str] = (),
) -> dict:
    """Turn one row (dict) into the ES _source document.

    - drops `drop_fields` (egress pruning; also drops id_field from _source is NOT done —
      the id is kept in _source by default so the doc is self-describing).
    - coerces values (timestamps -> epoch millis, nested structs preserved).

    Returns the _source dict. The _id is extracted separately by the caller
    (see build_action) so this stays a pure value transform.
    """
    drop = set(drop_fields)
    return {k: coerce_value(v) for k, v in row.items() if k not in drop}


def build_action(
    row: dict,
    *,
    index: str,
    id_field: Optional[str] = None,
    drop_fields: Iterable[str] = (),
) -> dict:
    """Build one Elasticsearch bulk action dict from a row.

    Deterministic _id (from id_field) gives idempotent upserts: replays/backfills
    overwrite the same doc instead of duplicating.
    """
    if not index:
        raise ValueError("index is required to build a bulk action")
    source = to_es_source(row, id_field=id_field, drop_fields=drop_fields)
    action = {"_index": index, "_source": source}
    if id_field is not None:
        if id_field not in row or row[id_field] is None:
            raise KeyError(f"id_field '{id_field}' missing/None in row")
        action["_id"] = str(row[id_field])
    return action
