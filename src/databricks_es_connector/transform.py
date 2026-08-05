"""Pure-Python row shaping for Elasticsearch. No Spark, no ES client: unit-testable.

`coerce_value` makes any value that `mapInPandas` can hand us JSON/ES-serializable, so
EVERY exportable column lands as usable data (field pruning via `drop_fields` is a client
opt-out to shrink payload, never the reason a column can't be exported). It handles every
Spark type that survives Arrow -> pandas conversion:
  - ES `date` fields reject str(pandas.Timestamp) (no 'T'); send epoch-millis instead.
  - Naive datetimes are treated as UTC so epochs are deterministic across executors
    (a naive value's local-tz epoch would differ per worker).
  - pandas nulls (NaT for datetimes, NaN for numerics, pd.NA) must become JSON null:
    NaT.timestamp() raises, and NaN serializes to the token `NaN` which ES rejects.
  - non-finite floats (inf, -inf) likewise have no JSON form (they serialize to `Infinity` /
    `-Infinity`, which ES rejects) and become JSON null.
  - numpy scalars/arrays (from Arrow) -> Python scalars / lists.
  - Decimal -> float (numeric + queryable in ES; note: >~15-17 sig-fig precision is lost).
  - bytes/bytearray (binary) -> base64 str (reversible, ES-safe).
  - Nested structs/maps arrive as dicts and lists pass through as JSON objects/arrays.
  - Total fallback: any other non-JSON-native object -> str(v), so an unforeseen type can
    never silently pass through and crash `helpers.bulk` (it lands as a string, not a crash).

NOT handled here (cannot be): types that fail Spark's Arrow conversion BEFORE reaching this
code: VARIANT and INTERVAL. `spark_prep.sanitize_for_arrow` serializes those to a JSON string
in Spark before the export (called automatically by `bulk_write`), so they never reach here.
"""
from __future__ import annotations

import base64 as _base64
import datetime as _dt
import math as _math
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
    # pandas.NA: `NA != NA` returns NA, and bool(NA) raises, identify it by type name
    # (avoids a hard pandas import in this pure-Python module).
    if type(v).__name__ in ("NAType", "NaTType"):
        return True
    try:
        return bool(v != v)   # float NaN (and NaT) satisfy v != v with a real bool
    except (ValueError, TypeError):
        return False          # e.g. numpy arrays, not a scalar null


_EPOCH_UTC = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def _to_epoch_millis(v: Any) -> int:
    """Coerce a datetime/date to epoch milliseconds. Naive values are assumed UTC.

    IMPORTANT: a Spark `TimestampType` (an instant) does NOT reach this function as a datetime.
    Spark's Arrow export converts a `timestamp` to the session-LOCAL wall-clock (naive) using
    `spark.sql.session.timeZone`, so treating that naive value as UTC would store an epoch shifted
    by the session's offset -- a real bug (verified on serverless under America/New_York /
    Asia/Kolkata; present since 0.1.0). It is fixed UPSTREAM in Spark by
    `spark_prep.normalize_timestamps_for_utc`, which converts every `TimestampType` to its true
    epoch-millis long via `unix_millis` (instant-based, session-tz-independent) BEFORE the Arrow
    export -- so a `timestamp` arrives here already an int and skips this function entirely.
    What still reaches this function as a naive datetime is genuinely UTC: a `date` (no
    time-of-day) and a `timestamp_ntz` (a zoneless wall-clock the connector defines as UTC). For
    those, naive==UTC is correct. The integration datatype test runs under a non-UTC session to
    keep the whole guarantee red-able.

    Floors to the containing millisecond (sub-millisecond precision is dropped, ES `date` is
    millisecond-resolution by default). Uses integer `timedelta` arithmetic rather than
    `v.timestamp() * 1000`: the float multiply loses sub-ms resolution at large magnitudes and
    drifts by ~1ms for far-future dates (first divergence ~year 2106). Python's `//` floors toward
    negative infinity, which keeps the flooring consistent across the epoch boundary (pre-epoch
    negatives round the same direction as post-epoch) and matches Spark/Java `unix_millis`, the
    reason we floor rather than truncate toward zero.
    """
    if isinstance(v, _dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=_dt.timezone.utc)
    else:
        # date without time -> midnight UTC
        v = _dt.datetime(v.year, v.month, v.day, tzinfo=_dt.timezone.utc)
    delta = v - _EPOCH_UTC
    # timedelta // timedelta is exact integer division, floored toward -inf.
    return delta // _dt.timedelta(milliseconds=1)


def _coerce_key(k: Any) -> str:
    """Render a map/dict key to a JSON-safe string.

    JSON object keys must be strings. Spark `map<K,V>` allows non-string key types (int, date,
    decimal, binary, ...); `json.dumps` silently stringifies int/bool/float/None keys but RAISES on
    date/decimal/bytes/struct keys, crashing helpers.bulk on the executor. So coerce the key with
    the SAME value transform (date -> epoch-millis, decimal -> float, bytes -> base64, ...) and then
    render it to a string. Spark maps are homogeneously typed, so distinct keys stay distinct.
    """
    ck = coerce_value(k)
    if isinstance(ck, str):
        return ck
    if ck is None:
        return "null"           # mirror json.dumps' rendering of a None key
    if isinstance(ck, bool):
        return "true" if ck else "false"
    return str(ck)


def coerce_value(v: Any, stats: Optional[dict] = None) -> Any:
    """Recursively make a value JSON/ES-serializable.

    - nulls (None, NaN, NaT, pd.NA) -> None
    - non-finite floats (inf, -inf) -> None (see below)
    - bool -> unchanged (before int/Decimal checks; bool is an int subclass)
    - datetimes/dates -> epoch millis (naive treated as UTC)
    - dict/list/tuple -> recurse (preserves nested structs/arrays)
    - numpy scalars/arrays -> Python scalars/lists (then recurse)
    - Decimal -> float
    - bytes/bytearray -> base64 str
    - int/float/str -> unchanged (JSON-native)
    - anything else -> str(v)  (total fallback: never let an unknown type crash the write)

    `stats`, if given, is a mutable dict this function increments to make otherwise-invisible
    coercions countable by the caller. Currently one key: `coerced_nonfinite`, bumped for every
    inf/-inf/NaN turned into JSON null. Those are unrepresentable in JSON and MUST become null (ES
    rejects the bare `NaN`/`Infinity` tokens), but an upstream divide-by-zero would otherwise land
    in ES as a null with no error, no sample, and no count. NaN is indistinguishable from a genuine
    null by the time it reaches here, so it is counted where it is detected (_is_null's float
    branch), not guessed at afterwards.
    """
    if v is not None and isinstance(v, float) and _math.isnan(v) and stats is not None:
        # Count NaN before the generic null branch swallows it (a NaN is a real value that became
        # null, unlike a None that was already null).
        stats["coerced_nonfinite"] = stats.get("coerced_nonfinite", 0) + 1
    if _is_null(v):                                   # must precede datetime: NaT is a datetime
        return None
    if isinstance(v, bool):                           # bool is a subclass of int, keep as-is
        return v
    if isinstance(v, (_dt.datetime, _dt.date)):
        return _to_epoch_millis(v)
    if isinstance(v, dict):
        return {_coerce_key(k): coerce_value(val, stats) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [coerce_value(x, stats) for x in v]
    # numpy values from Arrow / mapInPandas: array<...> columns arrive as np.ndarray
    # and numeric columns as numpy scalars (np.int64, np.float64, ...). Neither is
    # JSON-serializable and neither is caught by the list/dict branches above, so a
    # DataFrame with an array field (array<struct>) would break helpers.bulk.
    # .tolist() converts an ndarray to a (possibly nested) Python list and a numpy
    # scalar to a plain Python scalar; recurse UNCONDITIONALLY so an unwrapped scalar
    # (e.g. numpy inf -> Python inf) re-enters and hits the non-finite guard below.
    if type(v).__module__ == "numpy" and hasattr(v, "tolist"):
        return coerce_value(v.tolist(), stats)
    if isinstance(v, _Decimal):
        # float => numeric + queryable in ES. Precision beyond ~15-17 sig figs is lost;
        # cast the source column to string in Spark first if exactness is required.
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        return _base64.b64encode(bytes(v)).decode("ascii")
    if isinstance(v, float) and not _math.isfinite(v):
        # inf / -inf have no JSON representation: json.dumps emits the bare tokens
        # `Infinity` / `-Infinity`, which Elasticsearch's strict JSON parser rejects, so the
        # whole document fails to index and is silently lost to the error count. Coerce to
        # None (JSON null), mirroring the NaN handling in _is_null. (NaN is already caught by
        # _is_null above; this branch is reached only for inf/-inf.)
        if stats is not None:
            stats["coerced_nonfinite"] = stats.get("coerced_nonfinite", 0) + 1
        return None
    if isinstance(v, (int, float, str)):              # JSON-native scalars
        return v
    # Total fallback: an unforeseen type (e.g. a Spark-side object that slipped through
    # Arrow) becomes its string form rather than crashing helpers.bulk downstream.
    return str(v)


def to_es_source(
    row: dict,
    *,
    drop_fields: Iterable[str] = (),
    stats: Optional[dict] = None,
) -> dict:
    """Turn one row (dict) into the ES _source document.

    - drops `drop_fields` (egress pruning).
    - coerces values (timestamps -> epoch millis, nested structs preserved).

    The id column is deliberately KEPT in `_source`, so the document is self-describing. The `_id`
    itself is extracted separately by the caller (see build_action), which is why this function
    takes no id argument and stays a pure value transform.

    `stats`, if given, is forwarded to `coerce_value` to count invisible coercions.
    """
    drop = set(drop_fields)
    return {k: coerce_value(v, stats) for k, v in row.items() if k not in drop}


_DELETE_TRUE_STRINGS = ("true", "t", "1", "yes", "y")
_DELETE_FALSE_STRINGS = ("false", "f", "0", "no", "n")


class AmbiguousDeleteFlag(ValueError):
    """A delete-flag string that is neither clearly true nor clearly false.

    Raised rather than defaulted because both defaults are silently wrong: reading it as False
    leaves a doc in ES that should have been deleted (stale data, no error), and reading it as True
    deletes a doc that should have stayed. The caller must cast the column to a real boolean in
    Spark, or use one of the recognized string forms.
    """


def _is_delete_flagged(v: Any) -> bool:
    """True if a delete-flag column value means 'delete this row'.

    The flag arrives from Spark via mapInPandas, so it may be a Python bool, a numpy
    bool_, 0/1, or a string. Nulls (None/NaN/NaT/pd.NA) mean 'not a delete', a missing
    flag must never be read as a delete.

    Strings are parsed against an explicit allow-list in BOTH directions
    ('true'/'t'/'1'/'yes'/'y' vs 'false'/'f'/'0'/'no'/'n', case-insensitive, whitespace-trimmed).
    An unrecognized non-empty string RAISES AmbiguousDeleteFlag instead of quietly meaning "not a
    delete": values like 'on', 'enabled', 'delete', '2' and '-1' read as truthy to a human but would
    have left the document in Elasticsearch forever with no error and no count. An empty/whitespace
    string is treated as absent (not a delete), matching the null rule.
    """
    if _is_null(v):
        return False
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return False                 # empty string == absent == not a delete
        if s in _DELETE_TRUE_STRINGS:
            return True
        if s in _DELETE_FALSE_STRINGS:
            return False
        raise AmbiguousDeleteFlag(
            f"delete flag {v!r} is neither true-like {_DELETE_TRUE_STRINGS} nor false-like "
            f"{_DELETE_FALSE_STRINGS}. Cast the delete_flag_column to boolean in Spark rather than "
            "relying on string parsing, so the intent is unambiguous.")
    # numpy bool_/int and Python bool/int both respond correctly to bool(); coerce numpy
    # scalars to a Python value first so bool() is well-defined.
    if type(v).__module__ == "numpy" and hasattr(v, "item"):
        v = v.item()
    return bool(v)


def _require_id(row: dict, id_field: str) -> str:
    """Derive the deterministic _id from a row, rejecting null-ish ids.

    Guards `_is_null` (None, NaN, NaT, pd.NA), not just `is None`: a pandas NaN in a
    numeric id column is a float, so `is None` lets it through and `str(nan)` -> "nan"
    would make EVERY NaN-id row collide on _id="nan" and silently overwrite into one
    document (no error, counts still reconcile). Rejecting it fails the partition loudly
    instead of losing rows. The id column must be non-null in every row (see EsWriteConfig).
    """
    if id_field not in row or _is_null(row[id_field]):
        raise KeyError(f"id_field '{id_field}' missing/null (None/NaN/NaT) in row")
    return str(row[id_field])


def build_action(
    row: dict,
    *,
    index: str,
    id_field: Optional[str] = None,
    drop_fields: Iterable[str] = (),
    has_deletes: bool = False,
    delete_flag_column: Optional[str] = None,
    stats: Optional[dict] = None,
) -> dict:
    """Build one Elasticsearch bulk action dict from a row.

    Deterministic _id (from id_field) gives idempotent upserts: replays/backfills
    overwrite the same doc instead of duplicating.

    When has_deletes is True and the row's delete_flag_column is truthy, emit a
    delete-by-id action ({"_op_type": "delete", "_index", "_id"}, no _source) instead
    of an index. Deletes require id_field. The flag column itself is dropped from the
    _source of kept (non-delete) rows so it never pollutes the indexed document.
    """
    if not index:
        raise ValueError("index is required to build a bulk action")

    if has_deletes:
        if not delete_flag_column:
            raise ValueError("has_deletes=True requires delete_flag_column")
        if id_field is None:
            raise ValueError("has_deletes=True requires id_field (deletes target a doc _id)")
        if _is_delete_flagged(row.get(delete_flag_column)):
            # Delete action: id-only, no body. Idempotent, deleting an absent doc is a no-op
            # (the 404 is suppressed in the bulk layer).
            return {"_op_type": "delete", "_index": index, "_id": _require_id(row, id_field)}

    # Index path. Drop the delete flag from the body so it isn't indexed as data.
    drop = tuple(drop_fields) + ((delete_flag_column,) if (has_deletes and delete_flag_column) else ())
    source = to_es_source(row, drop_fields=drop, stats=stats)
    action = {"_index": index, "_source": source}
    if id_field is not None:
        action["_id"] = _require_id(row, id_field)
    return action
