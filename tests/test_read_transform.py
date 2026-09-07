"""Unit tests for the read coercion layer (read_transform.read_coerce). No Spark, no ES.

read_coerce turns an ES `_source` value back into the value Spark expects for the declared type. Each
test feeds it the STORED form a document actually holds (epoch-millis for temporal, base64 for binary,
a float for decimal, a JSON string for variant) and asserts the reconstructed Python value, including
the deltas the README documents as one-way (decimal precision, sub-ms timestamp, float32 widening).

The stored epoch-millis are computed here from a datetime the independent way (int(ts.timestamp()*
1000)), NOT via the connector's write code, so this file tests the read inverse on its own. The full
write->read round-trip (Spark to_json build_ndjson -> ES -> read_index) is proven end-to-end in the
integration tier (test_datatype_coverage / test_read_roundtrip), which is the only place Spark's
serializer actually runs; deleting the old pure-Python write transform moved that oracle there.
"""
import base64
import datetime as dt
from decimal import Decimal

import pytest

from databricks_es_connector.read_transform import read_coerce


def _epoch_ms(dt_obj: dt.datetime) -> int:
    """Ground-truth epoch-millis for an aware datetime, independent of the connector's write code.
    (Float multiply is exact for the millisecond-aligned instants used here.)"""
    return int(dt_obj.timestamp() * 1000)


# Container token constructors. read.py builds these tuples by walking the declared pyspark
# DataType; here we build them directly so the read-inverse is tested without Spark. Scalars stay
# plain strings ("int", "decimal(10,2)"), matching read.py's simpleString() leaves.
def _arr(elem):
    return ("array", elem)

def _map(key, val):
    return ("map", key, val)

def _struct(*fields):        # fields: (name, sub_token) pairs
    return ("struct", list(fields))


# --- scalars: exact read-back ----------------------------------------------------------------

def test_string_read():
    assert read_coerce("hello ☃", "string") == "hello ☃"

def test_bool_read():
    assert read_coerce(True, "boolean") is True
    assert read_coerce(False, "boolean") is False

def test_integer_widths_read():
    for tok in ("byte", "short", "int", "long"):
        assert read_coerce(7, tok) == 7
    assert read_coerce(9223372036854775807, "long") == 9223372036854775807   # max long, exact

def test_double_read():
    assert read_coerce(1.5, "double") == 1.5

def test_null_reads_to_none_for_every_type():
    for tok in ("string", "boolean", "long", "double", "timestamp", "date", "binary",
                "decimal(10,2)", _struct(("a", "int")), _arr("int"), _map("string", "int")):
        assert read_coerce(None, tok) is None


# --- temporal: epoch-millis -> datetime/date -------------------------------------------------

def test_timestamp_read_utc():
    ts = dt.datetime(2021, 1, 1, 12, 30, 0, tzinfo=dt.timezone.utc)
    out = read_coerce(_epoch_ms(ts), "timestamp")
    assert out == ts                       # exact to the millisecond
    assert out.tzinfo is not None          # aware, UTC

def test_date_read():
    d = dt.date(2021, 6, 15)
    stored = _epoch_ms(dt.datetime(2021, 6, 15, tzinfo=dt.timezone.utc))   # date stored as midnight UTC
    assert read_coerce(stored, "date") == d

def test_preepoch_timestamp_read():
    # A pre-epoch instant is stored as -1000ms; read must reconstruct the same instant.
    ts = dt.datetime(1969, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    assert read_coerce(-1000, "timestamp") == ts

def test_timestamp_subms_read_is_exact_to_ms():
    # Sub-millisecond precision is dropped on WRITE (documented one-way delta, proven in the
    # integration tier). The read side reconstructs exactly from the stored ms: 123 ms -> 123000 us.
    stored = _epoch_ms(dt.datetime(2021, 1, 1, 0, 0, 0, 123_000, tzinfo=dt.timezone.utc))
    out = read_coerce(stored, "timestamp")
    assert out == dt.datetime(2021, 1, 1, 0, 0, 0, 123_000, tzinfo=dt.timezone.utc)

def test_timestamp_ntz_read_is_naive():
    # timestamp_ntz must invert to a NAIVE datetime (the wall-clock Spark expects), symmetric with the
    # write side reading the wall-clock as UTC to pick the epoch. Not the raw epoch-millis int.
    wall = dt.datetime(2021, 1, 1, 12, 30, 0)          # naive wall-clock
    stored = _epoch_ms(wall.replace(tzinfo=dt.timezone.utc))   # wall-clock read as UTC on write
    out = read_coerce(stored, "timestamp_ntz")
    assert out == wall                                  # exact to the ms
    assert out.tzinfo is None                           # naive, not tz-aware
    assert isinstance(out, dt.datetime)                 # not the raw epoch-millis int

def test_timestamp_ntz_preepoch_naive():
    wall = dt.datetime(1969, 12, 31, 23, 59, 59)
    assert read_coerce(-1000, "timestamp_ntz") == wall


# --- binary: base64 -> bytes -----------------------------------------------------------------

def test_binary_read():
    stored = base64.b64encode(b"\x01\x02\x03").decode("ascii")   # the form ES holds
    assert read_coerce(stored, "binary") == b"\x01\x02\x03"


# --- decimal: documented precision loss ------------------------------------------------------

def test_decimal_read_within_double_precision():
    # A decimal stored as a float that fits in a double reads back exactly (as a Decimal again).
    assert read_coerce(1.5, "decimal(10,2)") == Decimal("1.5")

def test_decimal_precision_loss_is_documented():
    # 18 sig figs: the value is stored as a lossy float (write->double, README one-way delta). Read
    # reconstructs from that stored float, so it does NOT equal the original 18-digit decimal.
    stored = float(Decimal("123456789012345678"))       # the lossy double ES holds
    out = read_coerce(stored, "decimal(38,0)")
    assert out == Decimal("123456789012345680")          # low digits gone, per the write contract
    assert out != Decimal("123456789012345678")


# --- variant / interval: string passthrough --------------------------------------------------

def test_variant_reads_back_as_json_string():
    # A variant is stored as a JSON string on write; read v1 returns that string (caller parse_json).
    assert read_coerce("{\"k\":1}", "variant") == "{\"k\":1}"

def test_interval_reads_back_as_string():
    assert read_coerce("INTERVAL '1 02:03:04' DAY TO SECOND", "interval day to second") \
        == "INTERVAL '1 02:03:04' DAY TO SECOND"


# --- containers: recurse against sub-types ---------------------------------------------------

def test_array_read():
    assert read_coerce([1, 2, 3], _arr("int")) == [1, 2, 3]

def test_array_of_timestamps_read():
    ts = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    assert read_coerce([_epoch_ms(ts)], _arr("timestamp")) == [ts]

def test_nested_array_of_arrays_read():
    # array<array<int>>: the container tuple carries the inner array token, and read_coerce recurses
    # both levels. Empty inner array preserved; null inner element preserved.
    assert read_coerce([[1, 2], [3], []], _arr(_arr("int"))) == [[1, 2], [3], []]
    assert read_coerce([[1, 2], None], _arr(_arr("int"))) == [[1, 2], None]

def test_empty_string_read():
    assert read_coerce("", "string") == ""

def test_es_scalar_read_as_single_element_array():
    # ES has no array type: a field declared array<int> may come back as a bare scalar. Wrap it.
    assert read_coerce(5, _arr("int")) == [5]

def test_null_element_in_array_preserved():
    assert read_coerce([1, None, 3], _arr("int")) == [1, None, 3]

def test_struct_read_with_mixed_types():
    ts = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    stored = {"id": 7, "ts": _epoch_ms(ts), "payload": base64.b64encode(b"\x00\x01").decode("ascii")}
    out = read_coerce(stored, _struct(("id", "int"), ("ts", "timestamp"), ("payload", "binary")))
    assert out == {"id": 7, "ts": ts, "payload": b"\x00\x01"}

def test_struct_missing_field_is_none():
    # A field absent from _source (ES omits some) reads as None, not a KeyError.
    out = read_coerce({"a": 1}, _struct(("a", "int"), ("b", "string")))
    assert out == {"a": 1, "b": None}

def test_empty_struct_reads_as_empty_dict():
    # A zero-field struct has no fields to fill: an empty _source object reads back as {}.
    assert read_coerce({}, _struct()) == {}

def test_nested_struct_read():
    ts = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    stored = {"inner": {"x": 1, "ts": _epoch_ms(ts)}}
    out = read_coerce(stored, _struct(("inner", _struct(("x", "int"), ("ts", "timestamp")))))
    assert out == {"inner": {"x": 1, "ts": ts}}

def test_map_values_coerced_by_valtype():
    ts = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    stored = {"a": _epoch_ms(ts), "b": _epoch_ms(ts)}   # map<string,timestamp> as stored
    out = read_coerce(stored, _map("string", "timestamp"))
    assert out == {"a": ts, "b": ts}

def test_array_of_structs_read():
    rows = [{"k": "a", "v": 1}, {"k": "b", "v": 2}]
    assert read_coerce(rows, _arr(_struct(("k", "string"), ("v", "int")))) == rows

def test_map_non_string_keys_stay_stringified():
    # Non-string map keys are a DOCUMENTED one-way transform: writes stringify keys (JSON object keys
    # must be strings), so the read side only ever sees string keys and MUST keep them as such, even
    # when the declared key type is int. This is not lossy re-parsing; it's the contract. The stored
    # form for a map<int,string> is therefore {"1": "x", "2": "y"}.
    stored = {"1": "x", "2": "y"}
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
    # far-future dates (~2245+). Integer timedelta arithmetic must read back exact-to-the-ms.
    far = dt.datetime(2250, 5, 16, 4, 36, 8, 915000, tzinfo=dt.timezone.utc)
    ms = int(far.timestamp() * 1000)
    back = read_coerce(ms, "timestamp")
    assert back == far
    assert back.microsecond == 915000    # not 915001
