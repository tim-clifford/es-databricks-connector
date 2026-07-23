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
    delete: bool = False,
) -> dict:
    """Build one Elasticsearch bulk action dict from a row.

    Deterministic _id (from id_field) gives idempotent upserts: replays/backfills
    overwrite the same doc instead of duplicating.

    delete=True builds a delete action ({"_op_type": "delete", "_id": ...}) with no _source
    — used for Change Data Feed `delete` rows. A delete requires id_field.
    """
    if not index:
        raise ValueError("index is required to build a bulk action")
    if delete:
        if id_field is None:
            raise ValueError("delete action requires id_field")
        if id_field not in row or row[id_field] is None:
            raise KeyError(f"id_field '{id_field}' missing/None in row")
        return {"_op_type": "delete", "_index": index, "_id": str(row[id_field])}
    source = to_es_source(row, id_field=id_field, drop_fields=drop_fields)
    action = {"_index": index, "_source": source}
    if id_field is not None:
        if id_field not in row or row[id_field] is None:
            raise KeyError(f"id_field '{id_field}' missing/None in row")
        action["_id"] = str(row[id_field])
    return action


# --- Change Data Feed (CDC) handling -----------------------------------------------------
# CDF metadata columns emitted by Delta's readChangeFeed. They describe the change but must
# never land in the ES _source, so they are always pruned from the indexed document.
CDF_METADATA_FIELDS = ("_change_type", "_commit_version", "_commit_timestamp")

# _change_type values that mean "this row is the new/current state of the doc" (-> index).
_CDF_UPSERT_TYPES = ("insert", "update_postimage")
# The value that means "remove this doc" (-> delete).
_CDF_DELETE_TYPE = "delete"
# update_preimage is the OLD state before an update; it carries no new information and is
# always dropped (the matching update_postimage row carries the new state).
_CDF_DROP_TYPES = ("update_preimage",)


def collapse_cdf_changes(
    rows: list,
    *,
    id_field: str,
    change_type_field: str = "_change_type",
    commit_version_field: str = "_commit_version",
) -> list:
    """Collapse a batch of CDF change rows to one net action per id.

    Pure function (no Spark, no ES) so every corner case is unit-testable. Given a list of
    row dicts (each carrying a _change_type and a commit version), for each id keep only the
    LAST change by commit version (ties broken by original order), and drop update_preimage
    rows entirely. Returns a list of `(row, is_delete)` tuples in a deterministic order (first
    appearance of each id), where is_delete is True for a `delete` net-change.

    Semantics this produces:
      - insert then update    -> index the update_postimage        (last wins)
      - update then delete     -> delete                            (last wins)
      - delete then re-insert  -> index the insert                 (last wins)
      - N updates              -> index the highest-version postimage
    Unknown _change_type values are treated as upserts (fail-open to indexing, not dropping).
    """
    # winner[id] = (commit_version, seq, row, is_delete); seq preserves input order for ties.
    winner: dict = {}
    order: list = []  # first-appearance order of ids, for deterministic output
    for seq, row in enumerate(rows):
        ct = row.get(change_type_field)
        if ct in _CDF_DROP_TYPES:
            continue
        rid = row.get(id_field)
        if rid is None:
            raise KeyError(f"id_field '{id_field}' missing/None in a CDF row")
        rid = str(rid)
        # commit version may be absent/None; treat as -1 so any real version wins over it.
        cv = row.get(commit_version_field)
        try:
            cv = int(cv)
        except (TypeError, ValueError):
            cv = -1
        is_delete = (ct == _CDF_DELETE_TYPE)
        prev = winner.get(rid)
        if prev is None:
            order.append(rid)
            winner[rid] = (cv, seq, row, is_delete)
        elif (cv, seq) >= (prev[0], prev[1]):   # newer version, or same version later in batch
            winner[rid] = (cv, seq, row, is_delete)
    return [(winner[rid][2], winner[rid][3]) for rid in order]
