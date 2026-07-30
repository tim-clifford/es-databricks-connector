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


# --- non-finite floats: inf/-inf have no JSON representation. json.dumps emits the bare
# tokens `Infinity`/`-Infinity`, which ES's strict parser rejects, so the doc fails to index
# and is silently lost to the error count. Must coerce to None, mirroring NaN. The strict
# json.dumps(allow_nan=False) assertions below are what fail RED without the fix (default
# json.dumps permits Infinity and would hide the bug).

def _strict_json(v):
    """coerce, then serialize with the same strictness ES's parser enforces (no NaN/Infinity)."""
    return json.loads(json.dumps(coerce_value(v), allow_nan=False))


def test_positive_infinity_becomes_none():
    assert coerce_value(float("inf")) is None
    assert _strict_json(float("inf")) is None


def test_negative_infinity_becomes_none():
    assert coerce_value(float("-inf")) is None
    assert _strict_json(float("-inf")) is None


def test_numpy_infinity_scalar_becomes_none():
    np = pytest.importorskip("numpy")
    assert coerce_value(np.float64("inf")) is None
    assert coerce_value(np.float64("-inf")) is None
    assert coerce_value(np.float32("inf")) is None   # 32-bit FLOAT column path


def test_nested_and_listed_infinity_coerced():
    # inf anywhere in a struct/array must also be neutralized, or the whole doc's JSON is invalid.
    assert coerce_value({"a": float("inf")}) == {"a": None}
    assert coerce_value([float("inf"), float("-inf"), 1.5]) == [None, None, 1.5]
    assert _strict_json({"nested": {"x": float("inf")}}) == {"nested": {"x": None}}


def test_numpy_array_with_infinity_coerced():
    np = pytest.importorskip("numpy")
    arr = np.array([1.0, np.inf, -np.inf, 2.0])
    assert coerce_value(arr) == [1.0, None, None, 2.0]


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
# The contract under test is json.dumps() succeeding, that is exactly what breaks in
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


def test_epoch_millis_floors_consistently_across_epoch():
    # epoch-millis must FLOOR to the containing millisecond, not truncate toward zero, otherwise
    # pre-epoch (negative) timestamps round the wrong direction vs post-epoch, an inconsistency.
    # Matches Spark/Java unix_millis floor semantics.
    # 0.5 ms AFTER epoch -> floors to 0.
    post = dt.datetime(1970, 1, 1, 0, 0, 0, 500, tzinfo=dt.timezone.utc)  # +500 microseconds
    assert _json_roundtrip(post) == 0
    # 0.5 ms BEFORE epoch (-500 microseconds) -> floors to -1, NOT truncated toward zero (0).
    pre = dt.datetime(1969, 12, 31, 23, 59, 59, 999_500, tzinfo=dt.timezone.utc)  # -500 microseconds
    assert _json_roundtrip(pre) == -1
    # A whole pre-epoch second is exact under both floor and truncate; guards the common case.
    assert _json_roundtrip(dt.datetime(1969, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)) == -1000


def test_epoch_millis_exact_for_far_future_dates():
    # REGRESSION: v.timestamp() * 1000 loses ms precision at large magnitudes, a far-future
    # instant drifts by 1ms (first divergence ~year 2106). Integer arithmetic must be exact.
    # Compute the expected millis by exact integer math and assert the connector matches.
    from databricks_es_connector.transform import _to_epoch_millis
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    for v in (
        dt.datetime(2106, 6, 15, 12, 34, 56, 789_000, tzinfo=dt.timezone.utc),
        dt.datetime(2200, 1, 1, 0, 0, 0, 123_000, tzinfo=dt.timezone.utc),
        dt.datetime(2299, 12, 31, 23, 59, 59, 999_000, tzinfo=dt.timezone.utc),
    ):
        expected = (v - epoch) // dt.timedelta(milliseconds=1)   # exact integer millis, floored
        assert _to_epoch_millis(v) == expected, f"{v}: {_to_epoch_millis(v)} != {expected}"
        assert _json_roundtrip(v) == expected                    # and via the public coerce path


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


def test_dtype_empty_map():           # Spark map() -> {} (sibling of the empty-array edge)
    assert _json_roundtrip({}) == {}


# --- nulls INSIDE containers (not just a fully-null row) ---
# A client with sparse nested data relies on a null map value / array element / struct field
# landing as JSON null, recursively, the _is_null guard must fire inside the dict/list recursion,
# not only at the top level.

def test_null_map_value_becomes_json_null():
    assert _json_roundtrip({"k": None}) == {"k": None}
    assert _json_roundtrip({"k": float("nan")}) == {"k": None}   # pandas null inside a map


def test_null_array_element_becomes_json_null():
    assert _json_roundtrip([1, None, 3]) == [1, None, 3]
    assert _json_roundtrip([1, float("nan"), 3]) == [1, None, 3]  # pandas null inside an array


def test_partially_null_struct_keeps_nulls():
    # A struct where some fields are null: present fields kept, null fields -> JSON null.
    assert _json_roundtrip({"a": 1, "b": None, "c": "x"}) == {"a": 1, "b": None, "c": "x"}


def test_deeply_nested_nulls_coerced():
    # array<struct> where an inner field and an inner array element are both null.
    row = {"items": [{"v": None, "tags": ["a", None]}]}
    assert _json_roundtrip(row) == {"items": [{"v": None, "tags": ["a", None]}]}


# --- decimal precision loss at the documented boundary ---
# README documents "precision lost beyond ~15-17 sig figs" for decimal -> float. Prove it: an
# 18-sig-fig decimal does NOT round-trip to the same integer once widened to a double. This makes
# the documented lossy contract a *tested* fact (a client exporting a high-precision decimal must
# cast it to string in Spark first if exactness matters).

def test_decimal_precision_lost_beyond_double():
    from decimal import Decimal
    d = Decimal("123456789012345678")            # 18 significant digits
    out = _json_roundtrip(d)
    assert isinstance(out, float)
    assert int(out) == 123456789012345680         # NOT ...678: the low digits are lost to float64
    assert int(out) != int(d)                     # explicit: the exact integer did not survive


# --- non-string map keys (Spark map<K,V> where K is not a string) ---
# json.dumps stringifies int/bool/float/None keys but RAISES on date/decimal/bytes/tuple keys,
# which would crash helpers.bulk on the executor (uncounted). coerce_value must render every key
# to a JSON-safe string so any map<K,V> exports without a crash. Spark maps are homogeneously
# typed, so distinct keys always render distinctly (no collision to detect).

def test_map_int_keys_become_strings():
    # map<int,V> keys must be JSON strings, deterministically.
    assert _json_roundtrip({1: "a", 2: "b"}) == {"1": "a", "2": "b"}


def test_map_date_keys_do_not_crash():
    # THE crash case: a date key makes json.dumps raise TypeError without coercion.
    out = _json_roundtrip({dt.date(2024, 1, 26): "x"})
    assert out == {"1706227200000": "x"}   # key coerced the same way a date value would be


def test_map_decimal_and_bytes_keys_do_not_crash():
    from decimal import Decimal
    assert _json_roundtrip({Decimal("1.5"): "a"}) == {"1.5": "a"}
    assert _json_roundtrip({b"\x01\x02": "a"}) == {"AQI=": "a"}


def test_map_none_key_becomes_json_null_key():
    # A None key stringifies to the literal "null" (JSON object keys are always strings).
    assert _json_roundtrip({None: "a"}) == {"null": "a"}


def test_map_bool_key_becomes_true_false_string():
    # A bool map key renders to "true"/"false" (not "True"/"False"), matching JSON key semantics.
    assert _json_roundtrip({True: "a", False: "b"}) == {"true": "a", "false": "b"}


def test_nested_map_with_nonstring_keys_coerced():
    # A struct holding a map<int,timestamp>: keys AND values both need coercion, recursively.
    row = {"counts": {10: dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)}}
    assert _json_roundtrip(row) == {"counts": {"10": 1704067200000}}


def test_dtype_interval_as_string():  # INTERVAL can't cross Arrow; sanitize_for_arrow serializes it
    # sanitize_for_arrow (called by bulk_write) turns it into a JSON string before it reaches here.
    assert _json_roundtrip("INTERVAL '1 02:03:04' DAY TO SECOND") == "INTERVAL '1 02:03:04' DAY TO SECOND"


def test_dtype_unknown_object_falls_back_to_string():
    # Total fallback: an unforeseen non-JSON-native object must not crash the write,
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


def test_build_action_none_id_raises():
    # id present but None must raise, not become _id="None".
    with pytest.raises(KeyError):
        build_action({"doc_id": None, "x": 1}, index="idx", id_field="doc_id")


def test_build_action_nan_id_raises():
    # REGRESSION: a pandas NaN in a numeric id column is a float, not None. It must raise,
    # not slip past the guard and become _id="nan" (which would collapse every NaN-id row
    # onto one document, silently overwriting them). Guards the whole null-ish class.
    with pytest.raises(KeyError):
        build_action({"doc_id": float("nan"), "x": 1}, index="idx", id_field="doc_id")


def test_build_action_numpy_nan_id_raises():
    np = pytest.importorskip("numpy")
    with pytest.raises(KeyError):
        build_action({"doc_id": np.float64("nan"), "x": 1}, index="idx", id_field="doc_id")


def test_build_action_nat_id_raises():
    # A NaT in a timestamp id column is likewise null-ish and must raise.
    pd = pytest.importorskip("pandas")
    with pytest.raises(KeyError):
        build_action({"doc_id": pd.NaT, "x": 1}, index="idx", id_field="doc_id")


def test_delete_flagged_row_nan_id_raises():
    # Same guard on the delete path: a NaN id can't silently target _id="nan".
    with pytest.raises(KeyError):
        build_action({"doc_id": float("nan"), "d": True}, index="i", id_field="doc_id",
                     has_deletes=True, delete_flag_column="d")


def test_build_action_coerces_timestamp_in_source():
    row = {"doc_id": "1", "time": dt.datetime(2024, 1, 26, 11, 55, 23, tzinfo=dt.timezone.utc)}
    action = build_action(row, index="idx", id_field="doc_id")
    assert action["_source"]["time"] == 1706270123000


# --- delete routing (has_deletes + delete_flag_column) -----------------------------------

def test_flagged_row_becomes_delete_action():
    # A truthy flag routes the row to a delete-by-id action: id-only, no _source body.
    row = {"doc_id": "abc", "x": 1, "is_deleted": True}
    action = build_action(row, index="idx", id_field="doc_id",
                          has_deletes=True, delete_flag_column="is_deleted")
    assert action == {"_op_type": "delete", "_index": "idx", "_id": "abc"}
    assert "_source" not in action


def test_unflagged_row_is_index_and_flag_dropped_from_source():
    # A falsy flag keeps the row as an index, and the flag column must NOT be indexed as data.
    row = {"doc_id": "abc", "x": 1, "is_deleted": False}
    action = build_action(row, index="idx", id_field="doc_id",
                          has_deletes=True, delete_flag_column="is_deleted")
    assert action.get("_op_type") != "delete"               # indexed, not deleted
    assert action["_index"] == "idx" and action["_id"] == "abc"
    assert action["_source"] == {"doc_id": "abc", "x": 1}   # is_deleted pruned


def test_null_flag_is_not_a_delete():
    # A missing/null flag must never be read as a delete.
    for null in (None, float("nan")):
        row = {"doc_id": "abc", "is_deleted": null}
        action = build_action(row, index="idx", id_field="doc_id",
                              has_deletes=True, delete_flag_column="is_deleted")
        assert action.get("_op_type") != "delete"


def test_string_flag_values_parsed_leniently():
    truthy = ["true", "True", "t", "1", "yes"]
    falsy = ["false", "f", "0", "no", ""]
    for v in truthy:
        a = build_action({"doc_id": "x", "d": v}, index="i", id_field="doc_id",
                         has_deletes=True, delete_flag_column="d")
        assert a.get("_op_type") == "delete", v
    for v in falsy:
        a = build_action({"doc_id": "x", "d": v}, index="i", id_field="doc_id",
                         has_deletes=True, delete_flag_column="d")
        assert a.get("_op_type") != "delete", v


def test_numpy_bool_flag_is_a_delete():
    np = pytest.importorskip("numpy")
    a = build_action({"doc_id": "x", "d": np.bool_(True)}, index="i", id_field="doc_id",
                     has_deletes=True, delete_flag_column="d")
    assert a.get("_op_type") == "delete"


def test_delete_requires_id_field():
    with pytest.raises(ValueError):
        build_action({"d": True}, index="i", has_deletes=True, delete_flag_column="d")


def test_delete_flagged_row_missing_id_raises():
    with pytest.raises(KeyError):
        build_action({"d": True}, index="i", id_field="doc_id",
                     has_deletes=True, delete_flag_column="d")


def test_has_deletes_without_flag_column_raises():
    with pytest.raises(ValueError):
        build_action({"doc_id": "x"}, index="i", id_field="doc_id", has_deletes=True)


def test_deletes_off_ignores_flag_column():
    # With has_deletes=False, the flag column is just ordinary data and is indexed.
    row = {"doc_id": "abc", "is_deleted": True}
    action = build_action(row, index="idx", id_field="doc_id")
    assert action.get("_op_type") != "delete"
    assert action["_source"]["is_deleted"] is True
