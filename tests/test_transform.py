"""Unit tests for the pure-Python transform layer. No Spark, no ES needed."""
import datetime as dt
import json
import math

import pytest

from databricks_es_connector.transform import coerce_value, to_es_source, build_action


# --- null handling: pandas emits NaT for null datetimes and NaN for null numerics ---
# Both must become JSON null, or they crash the partition / produce ES-invalid JSON.

def test_nat_becomes_none():
    # pandas.NaT is a datetime subclass; .timestamp() raises. Must coerce to None.
    pd = pytest.importorskip("pandas")
    assert coerce_value(pd.NaT) is None


def test_nan_becomes_none():
    # float('nan') serializes to the literal token NaN, which ES's strict JSON rejects.
    assert coerce_value(float("nan")) is None


def test_pd_na_becomes_none():
    pd = pytest.importorskip("pandas")
    assert coerce_value(pd.NA) is None


def test_nested_and_listed_nulls_coerced():
    assert coerce_value({"a": float("nan")}) == {"a": None}
    assert coerce_value([float("nan")]) == [None]


def test_finite_floats_preserved():
    assert coerce_value(3.14) == 3.14
    assert coerce_value(0.0) == 0.0


# --- naive datetimes must be treated as UTC (deterministic across executor TZs) ---

def test_naive_datetime_treated_as_utc():
    naive = dt.datetime(2024, 1, 26, 12, 0, 0)                       # no tzinfo
    aware = dt.datetime(2024, 1, 26, 12, 0, 0, tzinfo=dt.timezone.utc)
    assert coerce_value(naive) == coerce_value(aware) == 1706270400000


# --- timestamp coercion (the gotcha: ES date fields reject str(Timestamp)) ---

def test_datetime_becomes_epoch_millis():
    d = dt.datetime(2024, 1, 26, 11, 55, 23, tzinfo=dt.timezone.utc)
    out = coerce_value(d)
    assert out == 1706270123000
    assert isinstance(out, int)


def test_date_becomes_epoch_millis():
    out = coerce_value(dt.date(2024, 1, 26))
    assert isinstance(out, int)


def test_nested_struct_timestamps_are_coerced():
    row = {"meta": {"ts": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)}}
    out = coerce_value(row)
    assert out["meta"]["ts"] == 1704067200000


def test_list_of_datetimes_coerced():
    out = coerce_value([dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)])
    assert out == [1704067200000]


def test_plain_values_unchanged():
    assert coerce_value("tcp") == "tcp"
    assert coerce_value(443) == 443
    assert coerce_value(None) is None


# --- numpy values from Arrow / mapInPandas ---
# array<...> columns arrive as np.ndarray and numeric columns as numpy scalars; both
# must coerce to JSON-serializable Python or helpers.bulk fails (an array<struct> column
# is the common trigger).

def test_numpy_array_of_structs_coerced():
    np = pytest.importorskip("numpy")
    import json
    answers = np.array(
        [{"type": "A", "rdata": "203.0.113.10", "ttl": 300},
         {"type": "A", "rdata": "203.0.113.11", "ttl": 300}],
        dtype=object,
    )
    out = coerce_value(answers)
    assert isinstance(out, list)
    assert out[0] == {"type": "A", "rdata": "203.0.113.10", "ttl": 300}
    json.dumps(out)   # must be JSON-serializable (would raise on a bare ndarray)


def test_empty_numpy_array_becomes_empty_list():
    np = pytest.importorskip("numpy")
    out = coerce_value(np.array([], dtype=object))
    assert out == []


def test_numpy_scalars_coerced_to_python():
    np = pytest.importorskip("numpy")
    import json
    assert coerce_value(np.int64(300)) == 300
    assert coerce_value(np.float64(1.5)) == 1.5
    json.dumps({"a": coerce_value(np.int64(1))})   # np scalars are not JSON-serializable


def test_numpy_nan_scalar_becomes_none():
    np = pytest.importorskip("numpy")
    assert coerce_value(np.float64("nan")) is None


def test_numpy_array_nested_nulls_and_timestamps_coerced():
    np = pytest.importorskip("numpy")
    arr = np.array([{"ttl": np.int64(300), "ts": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)}], dtype=object)
    out = coerce_value(arr)
    assert out == [{"ttl": 300, "ts": 1704067200000}]


# --- explicit coverage: every Spark datatype that reaches coerce_value must become
# JSON-serializable and land as usable data. The Python object each case uses is what
# mapInPandas actually produces for that Spark type (verified empirically on serverless).
# The contract under test is json.dumps() succeeding — that is exactly what breaks in
# helpers.bulk when a type is not coerced.

def _json_roundtrip(v):
    """coerce, then prove the result is JSON-serializable; return the parsed value."""
    return json.loads(json.dumps(coerce_value(v)))


def test_dtype_integers():            # Spark byte/short/int/long -> Python int
    assert _json_roundtrip(1) == 1
    assert _json_roundtrip(-2**40) == -2**40


def test_dtype_floats():              # Spark float/double -> Python float
    assert _json_roundtrip(1.5) == 1.5


def test_dtype_bool_preserved():      # Spark boolean -> bool (NOT coerced to 1/0)
    assert _json_roundtrip(True) is True
    assert _json_roundtrip(False) is False


def test_dtype_string():              # Spark string -> str
    assert _json_roundtrip("hello") == "hello"


def test_dtype_null():                # Spark null -> None
    assert _json_roundtrip(None) is None


def test_dtype_date_and_timestamp():  # Spark date/timestamp -> epoch millis (int)
    assert _json_roundtrip(dt.date(2024, 1, 26)) == 1706227200000
    assert _json_roundtrip(dt.datetime(2024, 1, 26, 11, 55, 23, tzinfo=dt.timezone.utc)) == 1706270123000


def test_dtype_decimal_to_float():    # Spark decimal -> float (numeric + queryable in ES)
    from decimal import Decimal
    out = _json_roundtrip(Decimal("1.25"))
    assert out == 1.25 and isinstance(out, float)


def test_dtype_binary_to_base64():    # Spark binary (bytes) -> base64 str (reversible)
    import base64
    out = _json_roundtrip(b"abc")
    assert out == "YWJj"
    assert base64.b64decode(out) == b"abc"   # round-trips back to the original bytes
    assert _json_roundtrip(bytearray(b"abc")) == "YWJj"


def test_dtype_array():               # Spark array<int> -> list
    assert _json_roundtrip([1, 2, 3]) == [1, 2, 3]


def test_dtype_map_and_struct():      # Spark map/struct -> dict (both arrive as dict)
    assert _json_roundtrip({"k": 1}) == {"k": 1}
    assert _json_roundtrip({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}


def test_dtype_interval_as_string():  # INTERVAL can't cross Arrow; caller casts to string
    # After a Spark-side cast (see cast_unsupported_to_string) it arrives as a plain str.
    assert _json_roundtrip("INTERVAL '1 02:03:04' DAY TO SECOND") == "INTERVAL '1 02:03:04' DAY TO SECOND"


def test_dtype_unknown_object_falls_back_to_string():
    # Total fallback: an unforeseen non-JSON-native object must not crash the write —
    # it lands as its string form. (Guards the helpers.bulk failure mode generically.)
    class Weird:
        def __str__(self):
            return "weird-repr"
    out = _json_roundtrip(Weird())
    assert out == "weird-repr"


def test_nested_struct_mixes_all_dtypes():
    # A struct carrying decimal + binary + timestamp + array, as a real nested row would.
    from decimal import Decimal
    row = {
        "id": 7,
        "score": Decimal("0.5"),
        "payload": b"\x00\x01",
        "ts": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        "tags": ["a", "b"],
    }
    out = _json_roundtrip(row)
    assert out == {"id": 7, "score": 0.5, "payload": "AAE=", "ts": 1704067200000, "tags": ["a", "b"]}


# --- field pruning (egress lever) ---

def test_to_es_source_drops_fields():
    row = {"a": 1, "raw_data": "big", "unmapped": "big2"}
    out = to_es_source(row, drop_fields=("raw_data", "unmapped"))
    assert out == {"a": 1}


def test_to_es_source_preserves_nested_struct():
    row = {"src_endpoint": {"ip": "10.0.0.1", "port": 443}}
    out = to_es_source(row)
    assert out["src_endpoint"] == {"ip": "10.0.0.1", "port": 443}


# --- bulk action building (deterministic id => idempotency) ---

def test_build_action_sets_deterministic_id():
    row = {"doc_id": "abc123", "x": 1}
    action = build_action(row, index="my-index", id_field="doc_id")
    assert action["_index"] == "my-index"
    assert action["_id"] == "abc123"
    assert action["_source"]["x"] == 1
    # id is kept in _source so the doc is self-describing
    assert action["_source"]["doc_id"] == "abc123"


def test_build_action_without_id_field():
    action = build_action({"x": 1}, index="idx")
    assert "_id" not in action
    assert action["_source"] == {"x": 1}


def test_build_action_requires_index():
    with pytest.raises(ValueError):
        build_action({"x": 1}, index="")


def test_build_action_missing_id_raises():
    with pytest.raises(KeyError):
        build_action({"x": 1}, index="idx", id_field="doc_id")


def test_build_action_coerces_timestamp_in_source():
    row = {"doc_id": "1", "time": dt.datetime(2024, 1, 26, 11, 55, 23, tzinfo=dt.timezone.utc)}
    action = build_action(row, index="idx", id_field="doc_id")
    assert action["_source"]["time"] == 1706270123000
