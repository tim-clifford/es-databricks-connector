"""Pure-Python inverse of `transform.coerce_value`: ES `_source` value -> the value Spark expects
for a declared target type. No Spark, no ES client: unit-testable, and the single coercion oracle
shared by the distributed reader (read.py).

The read path requires the caller to declare a Spark schema. That is deliberate: several
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


# Recognized boolean spellings for a `boolean`-declared field whose stored value is a STRING. Kept
# symmetric with transform._DELETE_TRUE_STRINGS / _DELETE_FALSE_STRINGS so the write and read sides
# agree on what a boolean-ish string means. Anything outside both lists raises rather than defaulting.
_BOOL_TRUE_STRINGS = ("true", "t", "1", "yes", "y")
_BOOL_FALSE_STRINGS = ("false", "f", "0", "no", "n")

# Declared Spark types the reader cannot honor, mapped to the fix. Their simpleString() tokens match
# no branch below, so before this check they fell through the unknown-token passthrough with the
# value UNTOUCHED -- skipping even the _reject_non_scalar guard that "string" applies. Spark then
# rejects the column from mapInPandas with `Invalid return type`, naming neither the field nor the
# real cause. Failing here names both.
_UNSUPPORTED_SCALAR_TOKENS = {
    "char": "CharType is not supported by read_index; declare StringType instead",
    "varchar": "VarcharType is not supported by read_index; declare StringType instead",
    "void": "NullType is not supported by read_index; declare the column's real type",
}


def _reject_unsupported_token(t: str, target: Any) -> None:
    """Raise ReadSchemaMismatch for a declared type read_index cannot carry through mapInPandas."""
    base = t.split("(", 1)[0].strip()
    fix = _UNSUPPORTED_SCALAR_TOKENS.get(base)
    if fix is not None:
        raise ReadSchemaMismatch(f"declared type {target!r} cannot be read: {fix}.")


# Inclusive value range of each signed integer width Spark carries. A stored value outside the
# declared width's range cannot be represented in that Spark type: mapInPandas casts the returned
# value to the declared Arrow type, and that cast either raises an opaque `ArrowInvalid` (Spark 4.1+,
# where spark.sql.execution.pandas.convertToArrowArraySafely defaults True) or SILENTLY WRAPS on the
# runtimes where it defaults False (all of DBR <=15.x / Spark 3.5): 10_000_000_000 declared `int`
# comes back 1410065408. Both are checked here so the read fails closed with a message that names the
# field and the fix, rather than corrupting data or surfacing a cause-less Arrow error downstream.
# `bigint` is bounded too: a JSON number larger than int64 (ES stores such values as a double or
# unsigned long) would overflow Spark's LongType the same way.
#
# KEYED ON THE TOKENS read.py ACTUALLY PRODUCES. read._spark_type_token renders a scalar via
# pyspark `DataType.simpleString()`, which emits `tinyint`/`smallint`/`int`/`bigint` for
# ByteType/ShortType/IntegerType/LongType -- NOT `byte`/`short`/`integer`/`long`. Keying on the
# latter left the two NARROWEST widths (tinyint/smallint), where overflow is most likely, matching
# no branch at all: a ByteType column fell through to the unknown-token passthrough with the value
# untouched, defeating this guard AND the truncation guard AND the int() coercion itself. The
# `byte`/`short`/`integer`/`long` spellings are kept as aliases so a hand-built token (tests, a
# caller invoking read_coerce directly) still works, but tinyint/smallint/int/bigint are the ones
# that occur in production. tests/test_silent_failure_hardening.py
# ::test_int_width_bounds_cover_every_production_token asserts these keys cover every real
# simpleString(), so a future width added with the wrong spelling fails a test instead of leaking.
_INT_WIDTH_BOUNDS = {
    "tinyint": (-(2 ** 7), 2 ** 7 - 1),
    "byte": (-(2 ** 7), 2 ** 7 - 1),           # alias: not emitted by simpleString(), hand-built only
    "smallint": (-(2 ** 15), 2 ** 15 - 1),
    "short": (-(2 ** 15), 2 ** 15 - 1),        # alias
    "int": (-(2 ** 31), 2 ** 31 - 1),
    "integer": (-(2 ** 31), 2 ** 31 - 1),      # alias
    "long": (-(2 ** 63), 2 ** 63 - 1),         # alias
    "bigint": (-(2 ** 63), 2 ** 63 - 1),
}


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

    A declared type the reader cannot honor (char/varchar/void) raises ReadSchemaMismatch. That is
    checked BEFORE the null branch: the type is wrong whatever the value happens to be, so a column
    that is null in the first document read must not look supported.
    """
    if not isinstance(target, tuple):
        _reject_unsupported_token(target.strip().lower(), target)
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
    # declared type decides int vs float; we coerce rather than trust the incoming Python type.
    # Membership is the keys of _INT_WIDTH_BOUNDS (not a re-typed tuple) so the branch and the bounds
    # table cannot drift apart: adding/renaming a width in one place changes both at once. This is
    # what makes the token set the SINGLE source of truth after the tinyint/smallint miss. ---
    if t in _INT_WIDTH_BOUNDS:
        _reject_non_scalar(value, target)
        # A non-integral float declared as an integer type would silently truncate (3.7 -> 3),
        # losing data with no signal. int(3.0) is exact and allowed; int(3.7) is not.
        if isinstance(value, float):
            if not value.is_integer():
                raise ReadSchemaMismatch(
                    f"stored value {value!r} is not an integer but the declared type is {target!r}; "
                    "int() would silently truncate it. Declare a float/double/decimal type, or "
                    "round upstream if truncation is genuinely intended.")
            value = int(value)
        else:
            value = int(value)
        # Value-magnitude check: the SIBLING of the truncation guard above and the second half of the
        # width failure the class docstring promises. A value that exceeds the declared width's range
        # cannot survive the mapInPandas cast to that Arrow type: it either raises a cause-less
        # ArrowInvalid (safe cast) or silently wraps (unsafe cast, the pre-Spark-4.1 default). Fail
        # closed here with a message naming the wider type to declare, so neither outcome reaches the
        # caller. Only integer widths have a bound; float/double/decimal branches handle range their
        # own way (float -> inf, decimal -> exact).
        lo, hi = _INT_WIDTH_BOUNDS[t]
        if not (lo <= value <= hi):
            raise ReadSchemaMismatch(
                f"stored value {value!r} is outside the range of the declared type {target!r} "
                f"([{lo}, {hi}]). Reading it into that Spark type would overflow the mapInPandas "
                "Arrow cast, which either raises an opaque error or silently wraps the value "
                "(e.g. 10000000000 -> 1410065408) depending on the runtime. Declare a wider integer "
                "type (short/int/long) or a decimal/string type wide enough to hold it.")
        return value
    if t in ("float", "double"):
        _reject_non_scalar(value, target)
        return float(value)

    if t == "boolean":
        _reject_non_scalar(value, target)
        if isinstance(value, str):
            # bool() on a string is truthiness, so a stored "false"/"0"/"no" would read back as
            # True: the value inverts, silently, and a boolean column is exactly where nobody
            # re-checks. Parse an explicit allow-list in BOTH directions and refuse anything else,
            # for the same reason `transform._is_delete_flagged` does on the write side: both
            # possible defaults are wrong. Strings arise when reading an index the connector did
            # not write; a connector round-trip stores real JSON booleans and skips this branch.
            #
            # It is NOT a mirror of that function on one input: an empty/whitespace string. There it
            # means "no flag present", and absent must mean "not a delete", so it returns False.
            # Here the caller has DECLARED this column boolean and ES stored a string, so "" is a
            # value that does not parse, not an absence -- a genuine null reads as None several
            # branches above and never reaches here. Returning False would invent a datum the source
            # does not contain, in the one column type where nobody re-checks. The asymmetry is
            # deliberate: the two functions answer different questions about the same characters.
            s = value.strip().lower()
            if s in _BOOL_TRUE_STRINGS:
                return True
            if s in _BOOL_FALSE_STRINGS:
                return False
            raise ReadSchemaMismatch(
                f"stored value {value!r} is a string that is neither true-like "
                f"{_BOOL_TRUE_STRINGS} nor false-like {_BOOL_FALSE_STRINGS}, but the declared type "
                f"is {target!r}. bool() would read ANY non-empty string as True, so 'false' would "
                "come back True. Declare StringType and convert in Spark, or fix the source data.")
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
