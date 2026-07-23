"""Unit tests for the pure-Python transform layer. No Spark, no ES needed."""
import datetime as dt
import json
import math

import pytest

from databricks_es_connector.transform import (
    coerce_value, to_es_source, build_action, collapse_cdf_changes, CDF_METADATA_FIELDS,
)
from databricks_es_connector import EsConfig


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


# ==========================================================================================
# Change Data Feed (CDC) support
# ==========================================================================================

# --- EsConfig: change_feed is opt-in and requires id_field; defaults are unchanged ---

def test_esconfig_defaults_have_change_feed_off():
    cfg = EsConfig(hosts="h", api_key="k")
    assert cfg.change_feed is False
    assert cfg.change_type_field == "_change_type"
    assert cfg.commit_version_field == "_commit_version"


def test_esconfig_change_feed_requires_id_field():
    with pytest.raises(ValueError):
        EsConfig(hosts="h", api_key="k", index="i", change_feed=True)   # no id_field
    # with id_field it constructs fine
    cfg = EsConfig(hosts="h", api_key="k", index="i", change_feed=True, id_field="doc_id")
    assert cfg.change_feed is True


# --- build_action delete path (CDF `delete` rows) ---

def test_build_action_delete_builds_delete_op_no_source():
    action = build_action({"doc_id": "abc", "x": 1}, index="idx", id_field="doc_id", delete=True)
    assert action == {"_op_type": "delete", "_index": "idx", "_id": "abc"}
    assert "_source" not in action


def test_build_action_delete_requires_id_field():
    with pytest.raises(ValueError):
        build_action({"x": 1}, index="idx", delete=True)         # no id_field
    with pytest.raises(KeyError):
        build_action({"x": 1}, index="idx", id_field="doc_id", delete=True)  # id missing in row


# --- collapse_cdf_changes: one net action per id, latest version wins ---

def _cdf(doc_id, change_type, version, **extra):
    return {"doc_id": doc_id, "_change_type": change_type, "_commit_version": version, **extra}


def test_collapse_single_insert():
    out = collapse_cdf_changes([_cdf("a", "insert", 1, val=1)], id_field="doc_id")
    assert len(out) == 1
    row, is_delete = out[0]
    assert is_delete is False and row["val"] == 1


def test_collapse_update_postimage_indexes():
    out = collapse_cdf_changes([_cdf("a", "update_postimage", 5, val=9)], id_field="doc_id")
    assert out == [({"doc_id": "a", "_change_type": "update_postimage", "_commit_version": 5, "val": 9}, False)]


def test_collapse_preimage_is_dropped():
    # A preimage-only change carries no new state and must be dropped entirely.
    out = collapse_cdf_changes([_cdf("a", "update_preimage", 5, val=0)], id_field="doc_id")
    assert out == []


def test_collapse_delete_is_delete():
    out = collapse_cdf_changes([_cdf("a", "delete", 3)], id_field="doc_id")
    assert len(out) == 1 and out[0][1] is True


def test_collapse_insert_then_update_last_wins():
    rows = [_cdf("a", "insert", 1, val=1), _cdf("a", "update_postimage", 2, val=2)]
    out = collapse_cdf_changes(rows, id_field="doc_id")
    assert len(out) == 1
    row, is_delete = out[0]
    assert is_delete is False and row["val"] == 2   # highest version wins


def test_collapse_update_then_delete_nets_to_delete():
    rows = [_cdf("a", "update_postimage", 1, val=1), _cdf("a", "delete", 2)]
    out = collapse_cdf_changes(rows, id_field="doc_id")
    assert len(out) == 1 and out[0][1] is True       # delete wins (later version)


def test_collapse_delete_then_reinsert_nets_to_index():
    rows = [_cdf("a", "delete", 1), _cdf("a", "insert", 2, val=7)]
    out = collapse_cdf_changes(rows, id_field="doc_id")
    assert len(out) == 1
    row, is_delete = out[0]
    assert is_delete is False and row["val"] == 7    # re-insert wins


def test_collapse_multiple_updates_highest_version():
    # Order-independent by construction: the highest commit version (val=30) must win for
    # EVERY input ordering. Checking all permutations means no positional-keep mutation
    # (keep-first, keep-last, keep-Nth) can pass — for any fixed position some permutation
    # puts a lower-version row there.
    import itertools
    base = [_cdf("a", "update_postimage", 1, val=10),
            _cdf("a", "update_postimage", 2, val=20),
            _cdf("a", "update_postimage", 3, val=30)]
    for perm in itertools.permutations(base):
        out = collapse_cdf_changes(list(perm), id_field="doc_id")
        assert len(out) == 1
        assert out[0][0]["val"] == 30, f"failed for input order {[r['val'] for r in perm]}"


def test_collapse_out_of_order_within_same_version_uses_input_order():
    # Same commit version (e.g. preimage/postimage share a version): later input row wins.
    rows = [_cdf("a", "update_postimage", 5, val="first"),
            _cdf("a", "update_postimage", 5, val="second")]
    out = collapse_cdf_changes(rows, id_field="doc_id")
    assert out[0][0]["val"] == "second"


def test_collapse_multiple_ids_independent_and_ordered():
    rows = [_cdf("a", "insert", 1, val="a1"),
            _cdf("b", "insert", 1, val="b1"),
            _cdf("a", "delete", 2),
            _cdf("c", "update_postimage", 1, val="c1")]
    out = collapse_cdf_changes(rows, id_field="doc_id")
    # first-appearance order: a, b, c
    ids = [r["doc_id"] for r, _ in out]
    assert ids == ["a", "b", "c"]
    net = {r["doc_id"]: is_del for r, is_del in out}
    assert net == {"a": True, "b": False, "c": False}   # a nets to delete


def test_collapse_unknown_change_type_treated_as_upsert():
    # Fail-open: an unrecognized _change_type indexes rather than silently dropping data.
    out = collapse_cdf_changes([_cdf("a", "something_new", 1, val=1)], id_field="doc_id")
    assert len(out) == 1 and out[0][1] is False


def test_collapse_missing_id_raises():
    with pytest.raises(KeyError):
        collapse_cdf_changes([{"_change_type": "insert", "_commit_version": 1}], id_field="doc_id")


def test_collapse_missing_version_still_collapses():
    # commit version absent/None -> treated as -1, so a real-versioned row wins over it.
    rows = [_cdf("a", "insert", None, val="v_none"), _cdf("a", "update_postimage", 0, val="v0")]
    out = collapse_cdf_changes(rows, id_field="doc_id")
    assert out[0][0]["val"] == "v0"


def test_cdf_metadata_fields_constant():
    # These must be pruned from _source; guard the exact set the writer strips.
    assert set(CDF_METADATA_FIELDS) == {"_change_type", "_commit_version", "_commit_timestamp"}


def test_cdf_metadata_stripped_from_source_via_drop_fields():
    # build_action with the CDF metadata in drop_fields must not leak them into _source.
    row = {"doc_id": "a", "val": 1, "_change_type": "insert", "_commit_version": 2,
           "_commit_timestamp": "2024-01-01"}
    action = build_action(row, index="idx", id_field="doc_id", drop_fields=CDF_METADATA_FIELDS)
    assert action["_source"] == {"doc_id": "a", "val": 1}   # metadata gone, doc intact
