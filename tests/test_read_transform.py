"""Unit tests for the read coercion layer (read_transform.read_coerce). No Spark, no ES.

The headline test is the ROUND-TRIP oracle: for every datatype, read_coerce(coerce_value(x), type)
must reproduce x, except the deltas the README documents as one-way (decimal precision, sub-ms
timestamp, float32 widening). This is the acceptance bar for the read path: it proves the read
inverse matches the write transform we already test in test_transform.py.
"""
import base64
import datetime as dt
from decimal import Decimal

import pytest

from databricks_es_connector.transform import coerce_value
from databricks_es_connector.read_transform import read_coerce


def _roundtrip(x, target):
    """write transform, then read transform: the value a client would get back for `target`."""
    return read_coerce(coerce_value(x), target)


# Container token constructors. read.py builds these tuples by walking the declared pyspark
# DataType; here we build them directly so the read-inverse is tested without Spark. Scalars stay
# plain strings ("int", "decimal(10,2)"), matching read.py's simpleString() leaves.
def _arr(elem):
    return ("array", elem)

def _map(key, val):
    return ("map", key, val)

def _struct(*fields):        # fields: (name, sub_token) pairs
    return ("struct", list(fields))


# --- scalars: exact round-trip ---------------------------------------------------------------

def test_string_roundtrip():
    assert _roundtrip("hello ☃", "string") == "hello ☃"

def test_bool_roundtrip():
    assert _roundtrip(True, "boolean") is True
    assert _roundtrip(False, "boolean") is False

def test_integer_widths_roundtrip():
    for tok in ("byte", "short", "int", "long"):
        assert _roundtrip(7, tok) == 7
    assert _roundtrip(9223372036854775807, "long") == 9223372036854775807   # max long, exact

def test_double_roundtrip():
    assert _roundtrip(1.5, "double") == 1.5

def test_null_roundtrips_to_none_for_every_type():
    for tok in ("string", "boolean", "long", "double", "timestamp", "date", "binary",
                "decimal(10,2)", _struct(("a", "int")), _arr("int"), _map("string", "int")):
        assert read_coerce(None, tok) is None


# --- temporal: epoch-millis <-> datetime/date ------------------------------------------------

def test_timestamp_roundtrip_utc():
    ts = dt.datetime(2021, 1, 1, 12, 30, 0, tzinfo=dt.timezone.utc)
    out = _roundtrip(ts, "timestamp")
    assert out == ts                       # exact to the millisecond
    assert out.tzinfo is not None          # aware, UTC

def test_date_roundtrip():
    d = dt.date(2021, 6, 15)
    assert _roundtrip(d, "date") == d

def test_preepoch_timestamp_roundtrip():
    # A pre-epoch instant floors to -1000ms on write; read must reconstruct the same instant.
    ts = dt.datetime(1969, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    assert _roundtrip(ts, "timestamp") == ts

def test_timestamp_subms_is_documented_loss():
    # Sub-millisecond precision is dropped on write (documented). Round-trip equals the value
    # FLOORED to the ms, not the original microseconds, assert the documented delta, not equality.
    ts = dt.datetime(2021, 1, 1, 0, 0, 0, 123_456, tzinfo=dt.timezone.utc)  # 123456 us
    out = _roundtrip(ts, "timestamp")
    assert out == dt.datetime(2021, 1, 1, 0, 0, 0, 123_000, tzinfo=dt.timezone.utc)  # floored to ms

def test_timestamp_ntz_roundtrip_is_naive():
    # REGRESSION: timestamp_ntz had no read branch, so it fell through to the unknown-token
    # passthrough and returned the raw epoch-millis INT. It must invert to a NAIVE datetime
    # (the wall-clock Spark expects for timestamp_ntz), symmetric with the UTC-read write side.
    wall = dt.datetime(2021, 1, 1, 12, 30, 0)          # naive wall-clock
    out = _roundtrip(wall, "timestamp_ntz")
    assert out == wall                                  # exact to the ms
    assert out.tzinfo is None                           # naive, not tz-aware
    assert isinstance(out, dt.datetime)                 # not the raw epoch-millis int

def test_timestamp_ntz_preepoch_naive():
    wall = dt.datetime(1969, 12, 31, 23, 59, 59)
    assert _roundtrip(wall, "timestamp_ntz") == wall


# --- binary: base64 <-> bytes ----------------------------------------------------------------

def test_binary_roundtrip():
    assert _roundtrip(b"\x01\x02\x03", "binary") == b"\x01\x02\x03"
    # and the stored form really was base64
    assert coerce_value(b"\x01\x02\x03") == base64.b64encode(b"\x01\x02\x03").decode("ascii")


# --- decimal: documented precision loss ------------------------------------------------------

def test_decimal_roundtrip_within_double_precision():
    # A decimal that fits in a double round-trips exactly (as a Decimal again).
    out = _roundtrip(Decimal("1.50"), "decimal(10,2)")
    assert out == Decimal("1.5")

def test_decimal_precision_loss_is_documented():
    # 18 sig figs: write->float loses the low digits (README). Read reconstructs from the lossy
    # float, so it does NOT equal the original, assert the documented lossy value.
    out = _roundtrip(Decimal("123456789012345678"), "decimal(38,0)")
    assert out == Decimal("123456789012345680")   # low digits gone, per the write contract
    assert out != Decimal("123456789012345678")


# --- variant / interval: string passthrough --------------------------------------------------

def test_variant_reads_back_as_json_string():
    # A variant is stored as a JSON string on write; read v1 returns that string (caller parse_json).
    stored = coerce_value("{\"k\":1}")   # sanitize_for_arrow already produced the JSON string
    assert read_coerce(stored, "variant") == "{\"k\":1}"

def test_interval_reads_back_as_string():
    assert read_coerce("INTERVAL '1 02:03:04' DAY TO SECOND", "interval day to second") \
        == "INTERVAL '1 02:03:04' DAY TO SECOND"


# --- containers: recurse against sub-types ---------------------------------------------------

def test_array_roundtrip():
    assert _roundtrip([1, 2, 3], _arr("int")) == [1, 2, 3]

def test_array_of_timestamps_roundtrip():
    ts = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    assert _roundtrip([ts], _arr("timestamp")) == [ts]

def test_nested_array_of_arrays_roundtrip():
    # array<array<int>>: the container tuple carries the inner array token, and read_coerce recurses
    # both levels. Empty inner array preserved; null inner element preserved.
    assert _roundtrip([[1, 2], [3], []], _arr(_arr("int"))) == [[1, 2], [3], []]
    assert read_coerce([[1, 2], None], _arr(_arr("int"))) == [[1, 2], None]

def test_empty_string_roundtrip():
    assert _roundtrip("", "string") == ""

def test_es_scalar_read_as_single_element_array():
    # ES has no array type: a field declared array<int> may come back as a bare scalar. Wrap it.
    assert read_coerce(5, _arr("int")) == [5]

def test_null_element_in_array_preserved():
    assert read_coerce([1, None, 3], _arr("int")) == [1, None, 3]

def test_struct_roundtrip_with_mixed_types():
    row = {"id": 7, "ts": dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc), "payload": b"\x00\x01"}
    stored = coerce_value(row)
    out = read_coerce(stored, _struct(("id", "int"), ("ts", "timestamp"), ("payload", "binary")))
    assert out == {"id": 7, "ts": dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
                   "payload": b"\x00\x01"}

def test_struct_missing_field_is_none():
    # A field absent from _source (ES omits some) reads as None, not a KeyError.
    out = read_coerce({"a": 1}, _struct(("a", "int"), ("b", "string")))
    assert out == {"a": 1, "b": None}

def test_empty_struct_reads_as_empty_dict():
    # A zero-field struct has no fields to fill: an empty _source object reads back as {}.
    assert read_coerce({}, _struct()) == {}

def test_nested_struct_roundtrip():
    row = {"inner": {"x": 1, "ts": dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)}}
    stored = coerce_value(row)
    out = read_coerce(stored, _struct(("inner", _struct(("x", "int"), ("ts", "timestamp")))))
    assert out == {"inner": {"x": 1, "ts": dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)}}

def test_map_values_coerced_by_valtype():
    ts = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    stored = coerce_value({"a": ts, "b": ts})   # map<string,timestamp>
    out = read_coerce(stored, _map("string", "timestamp"))
    assert out == {"a": ts, "b": ts}

def test_array_of_structs_roundtrip():
    rows = [{"k": "a", "v": 1}, {"k": "b", "v": 2}]
    stored = coerce_value(rows)
    assert read_coerce(stored, _arr(_struct(("k", "string"), ("v", "int")))) == rows

def test_map_non_string_keys_stay_stringified():
    # Non-string map keys are a DOCUMENTED one-way transform: writes stringify keys (JSON object
    # keys must be strings), so the read side can only ever see string keys and MUST keep them as
    # such, even when the declared key type is int. This is not lossy re-parsing; it's the contract.
    stored = coerce_value({1: "x", 2: "y"})     # map<int,string> -> {"1": "x", "2": "y"} on write
    assert stored == {"1": "x", "2": "y"}
    out = read_coerce(stored, _map("int", "string"))
    assert out == {"1": "x", "2": "y"}           # keys stay strings; values coerced by valtype
    assert all(isinstance(k, str) for k in out)


# --- token dispatch edge cases ---------------------------------------------------------------

def test_unknown_scalar_token_passes_value_through():
    # An unrecognized SCALAR type token must not crash: the value passes through unchanged (a
    # defensive fallback; the caller declared the schema, so this is a last resort, not normal).
    assert read_coerce({"anything": 1}, "somefuturetype") == {"anything": 1}
    assert read_coerce(42, "geo_point") == 42

def test_unknown_container_kind_passes_value_through():
    # A tuple whose kind isn't array/map/struct is also a defensive passthrough, not a crash.
    assert read_coerce({"x": 1}, ("somefuturecontainer", "int")) == {"x": 1}


def test_struct_with_nested_container_fields():
    # A struct whose fields are themselves containers: each field carries its own container tuple,
    # so nesting is just tuple recursion (no string parsing, no comma-splitting to get wrong).
    tok = _struct(("m", _map("string", "int")), ("a", _arr("int")), ("n", "long"))
    src = {"m": {"x": 1}, "a": [1, 2], "n": 5}
    out = read_coerce(src, tok)
    assert out == {"m": {"x": 1}, "a": [1, 2], "n": 5}

def test_struct_with_nested_decimal_field():
    # A decimal(p,s) field: the precision/scale comma used to be a parser hazard (it lived inside a
    # struct<...> string). With pre-parsed tuples the decimal is just an opaque scalar token, so the
    # comma is a non-issue. Corresponds to Spark simpleString() struct<a:decimal(10,2),b:int>.
    tok = _struct(("a", "decimal(10,2)"), ("b", "int"))
    out = read_coerce({"a": 1.5, "b": 3}, tok)
    assert out == {"a": Decimal("1.5"), "b": 3}
    assert set(out.keys()) == {"a", "b"}

def test_map_value_decimal():
    out = read_coerce({"k": 2.5}, _map("string", "decimal(10,2)"))
    assert out == {"k": Decimal("2.5")}

def test_array_of_decimal():
    out = read_coerce([1.5, 2.5], _arr("decimal(10,2)"))
    assert out == [Decimal("1.5"), Decimal("2.5")]

def test_struct_of_array_of_decimal():
    tok = _struct(("vals", _arr("decimal(5,2)")), ("n", "int"))
    out = read_coerce({"vals": [1.1, 2.2], "n": 7}, tok)
    assert out == {"vals": [Decimal("1.1"), Decimal("2.2")], "n": 7}

def test_far_future_timestamp_exact_to_ms():
    # REGRESSION: fromtimestamp(ms/1000) float division introduced a spurious ~1us error for
    # far-future dates (~2245+). Integer timedelta arithmetic must round-trip exact-to-the-ms.
    far = dt.datetime(2250, 5, 16, 4, 36, 8, 915000, tzinfo=dt.timezone.utc)
    ms = int(far.timestamp() * 1000)
    back = read_coerce(ms, "timestamp")
    assert back == far
    assert back.microsecond == 915000    # not 915001
