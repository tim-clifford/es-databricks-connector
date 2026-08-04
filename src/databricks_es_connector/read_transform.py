"""Pure-Python inverse of `transform.coerce_value`: ES `_source` value -> the value Spark expects
for a declared target type. No Spark, no ES client: unit-testable, and the single coercion oracle
shared by the driver-side reader (read.py) and the future distributed reader.

The read path requires the caller to declare a Spark schema (v0.4.0). That is deliberate: several
write transforms are documented as one-way and are NOT invertible from `_source` alone:
  - a `date`/`timestamp` is stored as an epoch-millis integer (indistinguishable from a plain long),
  - a `decimal` is stored as a float,
  - `binary` is stored as a base64 string,
  - `variant`/`interval` are stored as strings,
so only the caller's declared type tells us how to turn the stored value back. Each branch here is
the exact inverse of a `transform.coerce_value` branch; the accepted round-trip deltas are precisely
those the README datatype table documents (decimal precision, sub-millisecond timestamp, float32
widening): reads introduce no new lossiness.

Target types are given as type *tokens* rather than pyspark objects, so this module stays importable
without Spark for local unit testing. read.py maps a pyspark DataType to a token before calling in.
A token is either:
  - a scalar: a lowercase Spark type string ("string", "long", "timestamp", "decimal(10,2)", ...), or
  - a container: a tuple carrying its already-parsed sub-token(s), so no string parsing happens here:
      ("array", elem_token)
      ("map",   key_token, value_token)
      ("struct", [(field_name, field_token), ...])
read.py builds these tuples directly by walking the declared pyspark DataType (which is already a
navigable tree), so this module never has to re-parse a `struct<...>` DDL string.
"""
from __future__ import annotations

import base64 as _base64
import datetime as _dt
import math as _math
from decimal import Decimal as _Decimal
from typing import Any


class ReadSchemaMismatch(TypeError):
    """The stored ES value does not fit the Spark type the caller declared for it.

    Raised instead of coercing, because every available fallback produces plausible-looking WRONG
    data rather than an obvious failure: `str(["a","b"])` yields the literal `"['a', 'b']"`, and
    truncating 3.7 to an `int` yields 3. Both flow into a DataFrame that looks fine.

    The usual causes: (1) the field is multi-valued in ES but declared as a scalar (ES has no array
    type, so ANY field can hold multiple values under the same mapping, and nothing in the mapping
    reveals it: declare `array<T>`); (2) the declared numeric width is narrower than the stored
    value; (3) the field is an object but was declared as a scalar (declare a struct).
    """


def _is_null(v: Any) -> bool:
    """True for JSON null / missing. (ES omits nothing we send as explicit null; a missing field
    also arrives as None from the reader.) Kept simple, the read side never sees NaN/NaT."""
    return v is None


def _reject_non_scalar(value: Any, target: Any) -> None:
    """Raise ReadSchemaMismatch if `value` is a container but a SCALAR type was declared.

    A list means the ES field is multi-valued; a dict means it is an object. Either way, coercing to
    the declared scalar produces plausible-looking nonsense (a Python repr string, or a TypeError
    deep inside int()), so stop here with a message that names the fix.
    """
    if isinstance(value, (list, tuple)):
        raise ReadSchemaMismatch(
            f"field holds multiple values ({value!r}) but was declared as scalar {target!r}. "
            "Elasticsearch has no array type: any field can hold a list under the same mapping. "
            f"Declare array<{target}> to read it faithfully.")
    if isinstance(value, dict):
        raise ReadSchemaMismatch(
            f"field holds an object ({value!r}) but was declared as scalar {target!r}. "
            "Declare a matching struct<...> type.")


_EPOCH_UTC = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def _epoch_millis_to_datetime(v: Any) -> _dt.datetime:
    """Inverse of transform._to_epoch_millis for a timestamp: epoch-millis int -> aware UTC datetime.

    Mirrors the write side's UTC treatment: writes floor to the millisecond in UTC, so we read back
    in UTC. Sub-millisecond precision was dropped on write (documented), so this is exact to the ms.

    Uses integer timedelta arithmetic (epoch + timedelta(milliseconds=ms)) rather than
    `fromtimestamp(ms / 1000)`: the float division loses sub-ms resolution at large magnitudes and
    introduces a spurious ~1µs error for far-future dates (~year 2245+). timedelta keeps it exact.
    """
    return _EPOCH_UTC + _dt.timedelta(milliseconds=int(v))


def _epoch_millis_to_naive_datetime(v: Any) -> _dt.datetime:
    """Inverse of the write for a `timestamp_ntz`: epoch-millis int -> NAIVE datetime.

    A Spark `timestamp_ntz` is a zoneless wall-clock; the write side reads that wall-clock as UTC
    to pick a deterministic epoch (transform._to_epoch_millis). The symmetric read reconstructs the
    same wall-clock and drops the zone, so the value declared `timestamp_ntz` comes back naive (as
    Spark expects) rather than tz-aware. Exact to the ms, same as the aware path."""
    return _epoch_millis_to_datetime(v).replace(tzinfo=None)


def _epoch_millis_to_date(v: Any) -> _dt.date:
    """Inverse for a date: epoch-millis int -> date (UTC). Writes store a date as midnight-UTC
    epoch-millis, so reading the UTC date component round-trips the original date."""
    return _epoch_millis_to_datetime(v).date()


def read_coerce(value: Any, target: Any) -> Any:
    """Coerce one ES `_source` value to the Python value Spark expects for `target`.

    `target` is a type token (see the module docstring):
      - a scalar string: "string", "boolean", "byte"/"short"/"int"/"long", "float"/"double",
        "decimal(10,2)", "date", "timestamp", "timestamp_ntz", "binary", "variant", "interval ...".
      - a container tuple: ("array", elem), ("map", key, val), ("struct", [(name, sub), ...]).
        read.py builds these from the declared pyspark DataType, so no string parsing happens here.

    null/missing -> None for every type. Each non-null branch is the inverse of a coerce_value
    transform; anything already JSON-native (string/number/bool) passes through with the target's
    interpretation applied.
    """
    if _is_null(value):
        return None

    # --- container tokens: a pre-parsed tuple, recurse against the sub-token(s) ---
    if isinstance(target, tuple):
        kind = target[0]
        if kind == "array":
            elem = target[1]
            seq = value if isinstance(value, (list, tuple)) else [value]   # ES scalar-as-1-elem
            return [read_coerce(x, elem) for x in seq]
        if kind == "map":
            # JSON object keys are always strings; coerce values by the value sub-token.
            valtype = target[2]
            return {k: read_coerce(v, valtype) for k, v in dict(value).items()}
        if kind == "struct":
            src = dict(value)
            return {name: read_coerce(src.get(name), sub) for name, sub in target[1]}
        # Unknown container kind: passthrough rather than guess (mirrors the scalar fallback).
        return value

    t = target.strip().lower()

    # --- temporal: stored as epoch-millis integers, reconstructed per the declared type ---
    if t == "timestamp":
        _reject_non_scalar(value, target)
        return _epoch_millis_to_datetime(value)
    if t == "timestamp_ntz":
        # Zoneless wall-clock: write read it as UTC to pick the epoch, read hands it back naive.
        _reject_non_scalar(value, target)
        return _epoch_millis_to_naive_datetime(value)
    if t == "date":
        _reject_non_scalar(value, target)
        return _epoch_millis_to_date(value)

    # --- binary: stored base64, decode back to bytes ---
    if t == "binary":
        _reject_non_scalar(value, target)
        return _base64.b64decode(value)

    # --- decimal: stored as a float; precision beyond ~15-17 sig figs was already lost on write.
    # Decimal(str(v)) reconstructs a Decimal from the stored (possibly-rounded) float value. ---
    if t.startswith("decimal"):
        _reject_non_scalar(value, target)
        return _Decimal(str(value))

    # --- numeric scalars: JSON number -> the declared width. ES/JSON has one number type, so the
    # declared type decides int vs float; we coerce rather than trust the incoming Python type. ---
    if t in ("byte", "short", "int", "integer", "long", "bigint"):
        _reject_non_scalar(value, target)
        # A non-integral float declared as an integer type would silently truncate (3.7 -> 3),
        # losing data with no signal. int(3.0) is exact and allowed; int(3.7) is not.
        if isinstance(value, float):
            if not value.is_integer():
                raise ReadSchemaMismatch(
                    f"stored value {value!r} is not an integer but the declared type is {target!r}; "
                    "int() would silently truncate it. Declare a float/double/decimal type, or "
                    "round upstream if truncation is genuinely intended.")
            return int(value)
        return int(value)
    if t in ("float", "double"):
        _reject_non_scalar(value, target)
        return float(value)

    if t == "boolean":
        _reject_non_scalar(value, target)
        return bool(value)

    # --- string family, incl. the variant/interval-as-string round-trip (passthrough).
    # A variant is read back as its JSON string; reconstructing a Spark VARIANT is a caller-side
    # parse_json step (documented), symmetric with how writes serialize it. ---
    if t == "string" or t == "variant" or t.startswith("interval"):
        if isinstance(value, str):
            return value
        # str() on a list/dict yields Python repr ("['a', 'b']", "{'k': 1}"), which looks like data
        # but is garbage. Reject it; scalars (numbers/bools) still stringify meaningfully.
        _reject_non_scalar(value, target)
        return str(value)

    # Unknown token: leave the value as-is (a plain passthrough) rather than guess.
    return value
