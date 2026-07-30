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

Target types are given as Spark type *tokens* (short lowercase strings like "timestamp", "long",
"struct<...>") rather than pyspark objects, so this module stays importable without Spark for local
unit testing. read.py maps a pyspark DataType to its token before calling in.
"""
from __future__ import annotations

import base64 as _base64
import datetime as _dt
import math as _math
from decimal import Decimal as _Decimal
from typing import Any


def _is_null(v: Any) -> bool:
    """True for JSON null / missing. (ES omits nothing we send as explicit null; a missing field
    also arrives as None from the reader.) Kept simple, the read side never sees NaN/NaT."""
    return v is None


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


def _epoch_millis_to_date(v: Any) -> _dt.date:
    """Inverse for a date: epoch-millis int -> date (UTC). Writes store a date as midnight-UTC
    epoch-millis, so reading the UTC date component round-trips the original date."""
    return _epoch_millis_to_datetime(v).date()


def read_coerce(value: Any, target: str) -> Any:
    """Coerce one ES `_source` value to the Python value Spark expects for `target`.

    `target` is a Spark type token (lowercase), e.g. "string", "boolean", "byte"/"short"/"int"/
    "long", "float"/"double", "decimal(10,2)", "date", "timestamp", "binary", "variant",
    "interval ...", or a nested "struct<...>" / "array<...>" / "map<...>". Nested tokens recurse.

    null/missing -> None for every type. Each non-null branch is the inverse of a coerce_value
    transform; anything already JSON-native (string/number/bool) passes through with the target's
    interpretation applied.
    """
    if _is_null(value):
        return None

    t = target.strip().lower()

    # --- temporal: stored as epoch-millis integers, reconstructed per the declared type ---
    if t == "timestamp":
        return _epoch_millis_to_datetime(value)
    if t == "date":
        return _epoch_millis_to_date(value)

    # --- binary: stored base64, decode back to bytes ---
    if t == "binary":
        return _base64.b64decode(value)

    # --- decimal: stored as a float; precision beyond ~15-17 sig figs was already lost on write.
    # Decimal(str(v)) reconstructs a Decimal from the stored (possibly-rounded) float value. ---
    if t.startswith("decimal"):
        return _Decimal(str(value))

    # --- numeric scalars: JSON number -> the declared width. ES/JSON has one number type, so the
    # declared type decides int vs float; we coerce rather than trust the incoming Python type. ---
    if t in ("byte", "short", "int", "integer", "long", "bigint"):
        return int(value)
    if t in ("float", "double"):
        return float(value)

    if t == "boolean":
        return bool(value)

    # --- string family, incl. the variant/interval-as-string round-trip (passthrough).
    # A variant is read back as its JSON string; reconstructing a Spark VARIANT is a caller-side
    # parse_json step (documented), symmetric with how writes serialize it. ---
    if t == "string" or t == "variant" or t.startswith("interval"):
        return value if isinstance(value, str) else str(value)

    # --- nested containers: recurse against the element/field/value sub-type ---
    if t.startswith("array<"):
        elem = _inner(t, "array<")
        seq = value if isinstance(value, (list, tuple)) else [value]   # ES scalar-as-1-elem
        return [read_coerce(x, elem) for x in seq]

    if t.startswith("map<"):
        # map<keytype,valtype>: JSON object keys are always strings; coerce values by valtype.
        _k, valtype = _split_map(t)
        return {k: read_coerce(v, valtype) for k, v in dict(value).items()}

    if t.startswith("struct<"):
        fields = _struct_fields(t)
        src = dict(value)
        return {name: read_coerce(src.get(name), ftype) for name, ftype in fields}

    # Unknown token: leave the value as-is (a plain passthrough) rather than guess.
    return value


# --- token parsing helpers (a minimal Spark DDL-type-string parser, depth-aware) -----------------

def _inner(token: str, prefix: str) -> str:
    """The single inner type of array<INNER>, strips the prefix and the trailing '>'."""
    return token[len(prefix):-1].strip()


def _split_top_level(body: str):
    """Split a struct/map body on top-level commas only.

    Ignores commas nested inside `<...>` (nested struct/array/map) AND inside `(...)`: the latter
    matters because a `decimal(p,s)` token carries a comma between its precision and scale, e.g.
    `struct<a:decimal(10,2),b:int>`. Tracking angle-bracket depth alone would wrongly split on that
    inner comma; we track paren depth too.
    """
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i]); start = i + 1
    parts.append(body[start:])
    return [p.strip() for p in parts if p.strip()]


def _split_map(token: str):
    """map<keytype,valtype> -> (keytype, valtype), splitting on the top-level comma."""
    body = token[len("map<"):-1]
    parts = _split_top_level(body)
    return parts[0], parts[1]


def _struct_fields(token: str):
    """struct<name:type,name:type,...> -> [(name, type), ...]. The field split (_split_top_level) is
    depth-aware over both <...> and (...) so a decimal(p,s)'s inner comma doesn't split a field. The
    name/type split is a plain first-':' partition: a Spark field name contains no ':' (or brackets),
    so the first ':' always separates name from type, even when the type is a decimal or a struct."""
    body = token[len("struct<"):-1]
    out = []
    for field in _split_top_level(body):
        name, _colon, ftype = field.partition(":")
        out.append((name.strip(), ftype.strip()))
    return out
