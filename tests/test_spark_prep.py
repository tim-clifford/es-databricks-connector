"""Unit tests for spark_prep's Arrow-hostile-column detection.

The Spark-touching parts (createOrReplaceTempView / DESCRIBE / to_json) need a session and are
exercised by the OCSF round-trip demo on a live cluster. Here we test the PURE logic,
_hostile_columns_from_describe, which is where the detection rules and edge cases live, using
synthetic DESCRIBE output. This is the regression guard for the VARIANT/INTERVAL export bug:
before the fix, a DataFrame with a VARIANT column crashed bulk_write; the fix detects such columns
here and serializes them.
"""
import pytest

from databricks_es_connector.spark_prep import (
    _hostile_columns_from_describe,
    _is_scalar_interval_type,
    _type_is_arrow_hostile,
)


def _rows(*pairs):
    """Build DESCRIBE-style rows from (col_name, data_type) pairs."""
    return [{"col_name": n, "data_type": t} for n, t in pairs]


def _names(describe_rows):
    """Just the detected column names: most tests assert on names, not the carried type text."""
    return [name for name, _type in _hostile_columns_from_describe(describe_rows)]


# --- _type_is_arrow_hostile: the type-string classifier ---------------------------------------
# A DESCRIBE type string interleaves type names with struct FIELD names, so detection must match
# VARIANT/INTERVAL in TYPE position only, never as a substring of a field name.

_SHOULD_DETECT = [
    "variant",                                          # top-level variant
    "interval day to second",                           # day-time interval
    "interval year to month",                           # year-month interval
    "array<variant>",                                   # array of variant
    "map<string,variant>",                              # variant as map value
    "struct<uid:string,tags:variant>",                  # variant nested in struct
    "array<struct<data:variant,name:string>>",          # variant nested in array<struct>
    "struct<dur:interval day to second>",               # interval nested in struct
    "struct<variant:variant>",                          # field NAMED variant, of TYPE variant
    "VARIANT",                                           # case-insensitive
    "INTERVAL DAY TO SECOND",
]

_SHOULD_NOT_DETECT = [
    "string", "bigint", "boolean", "timestamp",
    "struct<ip:string,port:int>",                       # plain struct
    "array<string>",                                     # plain array
    "struct<polling_interval_ms:bigint>",               # field name CONTAINS 'interval'
    "struct<retry_interval:int,name:string>",           # field name IS 'retry_interval'
    "struct<interval_count:int>",                       # field name STARTS with 'interval'
    "struct<invariant_score:double>",                   # field name CONTAINS 'variant'
    "struct<covariant:double>",                         # field name ENDS with 'variant'
    "struct<variant_id:string>",                        # field name STARTS with 'variant'
    "struct<my_variant:string>",                        # field name ENDS with 'variant'
    "struct<variant:string>",                           # field NAMED variant, of TYPE string
]


@pytest.mark.parametrize("type_text", _SHOULD_DETECT)
def test_type_is_hostile_positive(type_text):
    assert _type_is_arrow_hostile(type_text) is True


@pytest.mark.parametrize("type_text", _SHOULD_NOT_DETECT)
def test_type_is_hostile_negative(type_text):
    assert _type_is_arrow_hostile(type_text) is False


def test_type_is_hostile_handles_none():
    assert _type_is_arrow_hostile(None) is False
    assert _type_is_arrow_hostile("") is False


def test_detects_top_level_variant():
    rows = _rows(("id", "string"), ("raw_data", "variant"), ("n", "int"))
    assert _names(rows) == ["raw_data"]


def test_detects_top_level_interval():
    rows = _rows(("id", "string"), ("dur", "interval day to second"))
    assert _names(rows) == ["dur"]


def test_detects_variant_nested_in_struct():
    rows = _rows(
        ("id", "string"),
        ("metadata", "struct<uid:string,tags:variant>"),
    )
    assert _names(rows) == ["metadata"]


def test_detects_variant_nested_in_array_of_struct():
    rows = _rows(
        ("id", "string"),
        ("enrichments", "array<struct<data:variant,name:string>>"),
    )
    assert _names(rows) == ["enrichments"]


def test_clean_schema_has_no_hostile_columns():
    rows = _rows(
        ("id", "string"), ("time", "timestamp"), ("n", "bigint"),
        ("endpoint", "struct<ip:string,port:int>"),
        ("tags", "array<string>"),
    )
    assert _names(rows) == []


def test_multiple_hostile_columns_all_detected():
    rows = _rows(
        ("id", "string"),
        ("raw_data", "variant"),
        ("metadata", "struct<uid:string,x:variant>"),
        ("ok", "string"),
        ("dur", "interval year to month"),
    )
    assert _names(rows) == ["raw_data", "metadata", "dur"]


def test_stops_at_partition_information_section():
    # DESCRIBE appends a '# Partition Information' block after the columns; a stray 'variant' word
    # in that section (or blank rows) must not be scanned as a column.
    rows = _rows(
        ("id", "string"),
        ("raw_data", "variant"),
        ("", ""),
        ("# Partition Information", ""),
        ("# col_name", "data_type"),
        ("part_variant_col", "variant"),  # after the marker, must be ignored
    )
    assert _names(rows) == ["raw_data"]


def test_case_insensitive_type_match():
    rows = _rows(("v", "VARIANT"), ("i", "INTERVAL DAY TO SECOND"))
    assert _names(rows) == ["v", "i"]


def test_null_data_type_is_safe():
    # A row with a None data_type (defensive) must not crash.
    rows = [{"col_name": "x", "data_type": None}, {"col_name": "v", "data_type": "variant"}]
    assert _names(rows) == ["v"]


def test_innocuous_field_names_are_not_false_positives():
    # Regression: a column whose type merely has a struct FIELD named 'polling_interval' /
    # 'invariant_score' must NOT be flagged (it is Arrow-crossable). Only the genuinely
    # variant/interval-TYPED columns are hostile. A naive substring match would wrongly flag
    # 'sched' and 'stats' here and silently JSON-stringify valid struct columns.
    rows = _rows(
        ("id", "string"),
        ("sched", "struct<polling_interval_ms:bigint,retry_interval:int>"),  # field names only
        ("stats", "struct<invariant_score:double,covariant:double>"),        # field names only
        ("raw_data", "variant"),                                             # genuinely hostile
        ("dur", "interval day to second"),                                   # genuinely hostile
    )
    assert _names(rows) == ["raw_data", "dur"]


# --- the detector carries each column's TYPE TEXT so the caller can pick a serialization ---------
# (scalar interval -> cast string; everything else -> to_json). See sanitize_for_arrow.

def test_detect_returns_name_and_type_pairs():
    rows = _rows(
        ("id", "string"),
        ("raw_data", "variant"),
        ("dur", "interval day to second"),
    )
    assert _hostile_columns_from_describe(rows) == [
        ("raw_data", "variant"),
        ("dur", "interval day to second"),
    ]


# --- _is_scalar_interval_type: which hostile columns need cast(string) instead of to_json ---------
# Spark's to_json REJECTS a scalar interval, but ACCEPTS a struct/array/map that contains one. So
# only a TOP-LEVEL interval type is a "scalar interval"; a container that merely holds an interval
# is not (its top-level type is struct/array/map, which to_json handles).

_SCALAR_INTERVAL = [
    "interval day to second",
    "interval year to month",
    "INTERVAL DAY TO SECOND",     # case-insensitive
    "  interval day to second ",  # surrounding whitespace tolerated
]

_NOT_SCALAR_INTERVAL = [
    "variant",                                  # variant -> to_json, not cast
    "struct<dur:interval day to second>",       # CONTAINS interval, but top-level is struct
    "array<interval day to second>",            # top-level is array
    "map<string,interval day to second>",       # top-level is map
    "string", "bigint",
    "struct<interval_count:int>",               # field name only, not an interval type at all
]


@pytest.mark.parametrize("type_text", _SCALAR_INTERVAL)
def test_scalar_interval_positive(type_text):
    assert _is_scalar_interval_type(type_text) is True


@pytest.mark.parametrize("type_text", _NOT_SCALAR_INTERVAL)
def test_scalar_interval_negative(type_text):
    assert _is_scalar_interval_type(type_text) is False


def test_scalar_interval_handles_none():
    assert _is_scalar_interval_type(None) is False
    assert _is_scalar_interval_type("") is False


# --- timestamp -> epoch-millis normalization (tz-safety) --------------------------------------
# _type_has_timestamp / _epoch_struct_type are pure functions over pyspark DataType objects
# (pyspark TYPES import without a JVM; only a live SESSION needs Java), so the detection + result-
# type logic is unit-testable here. The Spark rewrite itself (unix_millis + transform/struct
# rebuild) runs on a live session and is covered by the integration datatype test under a NON-UTC
# session, which is the red-able regression guard for the tz-corruption bug.

def _types():
    from pyspark.sql.types import (ArrayType, MapType, StructType, StructField, TimestampType,
                                   TimestampNTZType, DateType, LongType, StringType, IntegerType)
    return dict(ArrayType=ArrayType, MapType=MapType, StructType=StructType, StructField=StructField,
                TimestampType=TimestampType, TimestampNTZType=TimestampNTZType, DateType=DateType,
                LongType=LongType, StringType=StringType, IntegerType=IntegerType)


def test_type_has_timestamp_top_level():
    from databricks_es_connector.spark_prep import _type_has_timestamp
    t = _types()
    assert _type_has_timestamp(t["TimestampType"]()) is True
    # NTZ and Date are deliberately NOT matched: different, already-correct handling.
    assert _type_has_timestamp(t["TimestampNTZType"]()) is False
    assert _type_has_timestamp(t["DateType"]()) is False
    assert _type_has_timestamp(t["StringType"]()) is False


def test_type_has_timestamp_nested():
    from databricks_es_connector.spark_prep import _type_has_timestamp
    t = _types()
    struct_ts = t["StructType"]([t["StructField"]("a", t["IntegerType"]()),
                                 t["StructField"]("ts", t["TimestampType"]())])
    assert _type_has_timestamp(struct_ts) is True
    assert _type_has_timestamp(t["ArrayType"](t["TimestampType"]())) is True
    assert _type_has_timestamp(t["MapType"](t["StringType"](), t["TimestampType"]())) is True
    # a struct/array with NO timestamp (and only ntz/date) is not matched
    clean = t["StructType"]([t["StructField"]("n", t["TimestampNTZType"]()),
                             t["StructField"]("d", t["DateType"]())])
    assert _type_has_timestamp(clean) is False
    assert _type_has_timestamp(t["ArrayType"](t["StringType"]())) is False


def test_epoch_struct_type_maps_timestamp_to_long_recursively():
    from databricks_es_connector.spark_prep import _epoch_struct_type
    t = _types()
    # top-level timestamp -> long
    assert isinstance(_epoch_struct_type(t["TimestampType"]()), t["LongType"])
    # ntz/date unchanged
    assert isinstance(_epoch_struct_type(t["TimestampNTZType"]()), t["TimestampNTZType"])
    assert isinstance(_epoch_struct_type(t["DateType"]()), t["DateType"])
    # nested: struct<ts:timestamp, n:ntz> -> struct<ts:long, n:ntz>
    src = t["StructType"]([t["StructField"]("ts", t["TimestampType"]()),
                           t["StructField"]("n", t["TimestampNTZType"]())])
    out = _epoch_struct_type(src)
    fields = {f.name: type(f.dataType).__name__ for f in out.fields}
    assert fields == {"ts": "LongType", "n": "TimestampNTZType"}
    # array<timestamp> -> array<long>; map<string,timestamp> -> map<string,long>
    assert isinstance(_epoch_struct_type(t["ArrayType"](t["TimestampType"]())).elementType, t["LongType"])
    assert isinstance(_epoch_struct_type(t["MapType"](t["StringType"](), t["TimestampType"]())).valueType, t["LongType"])
