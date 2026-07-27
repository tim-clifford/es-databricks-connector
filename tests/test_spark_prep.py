"""Unit tests for spark_prep's Arrow-hostile-column detection.

The Spark-touching parts (createOrReplaceTempView / DESCRIBE / to_json) need a session and are
exercised by the OCSF round-trip demo on a live cluster. Here we test the PURE logic —
_hostile_columns_from_describe — which is where the detection rules and edge cases live, using
synthetic DESCRIBE output. This is the regression guard for the VARIANT/INTERVAL export bug:
before the fix, a DataFrame with a VARIANT column crashed bulk_write; the fix detects such columns
here and serializes them.
"""
import pytest

from databricks_es_connector.spark_prep import (
    _hostile_columns_from_describe,
    _type_is_arrow_hostile,
)


def _rows(*pairs):
    """Build DESCRIBE-style rows from (col_name, data_type) pairs."""
    return [{"col_name": n, "data_type": t} for n, t in pairs]


# --- _type_is_arrow_hostile: the type-string classifier ---------------------------------------
# A DESCRIBE type string interleaves type names with struct FIELD names, so detection must match
# VARIANT/INTERVAL in TYPE position only — never as a substring of a field name.

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
    assert _hostile_columns_from_describe(rows) == ["raw_data"]


def test_detects_top_level_interval():
    rows = _rows(("id", "string"), ("dur", "interval day to second"))
    assert _hostile_columns_from_describe(rows) == ["dur"]


def test_detects_variant_nested_in_struct():
    rows = _rows(
        ("id", "string"),
        ("metadata", "struct<uid:string,tags:variant>"),
    )
    assert _hostile_columns_from_describe(rows) == ["metadata"]


def test_detects_variant_nested_in_array_of_struct():
    rows = _rows(
        ("id", "string"),
        ("enrichments", "array<struct<data:variant,name:string>>"),
    )
    assert _hostile_columns_from_describe(rows) == ["enrichments"]


def test_clean_schema_has_no_hostile_columns():
    rows = _rows(
        ("id", "string"), ("time", "timestamp"), ("n", "bigint"),
        ("endpoint", "struct<ip:string,port:int>"),
        ("tags", "array<string>"),
    )
    assert _hostile_columns_from_describe(rows) == []


def test_multiple_hostile_columns_all_detected():
    rows = _rows(
        ("id", "string"),
        ("raw_data", "variant"),
        ("metadata", "struct<uid:string,x:variant>"),
        ("ok", "string"),
        ("dur", "interval year to month"),
    )
    assert _hostile_columns_from_describe(rows) == ["raw_data", "metadata", "dur"]


def test_stops_at_partition_information_section():
    # DESCRIBE appends a '# Partition Information' block after the columns; a stray 'variant' word
    # in that section (or blank rows) must not be scanned as a column.
    rows = _rows(
        ("id", "string"),
        ("raw_data", "variant"),
        ("", ""),
        ("# Partition Information", ""),
        ("# col_name", "data_type"),
        ("part_variant_col", "variant"),  # after the marker — must be ignored
    )
    assert _hostile_columns_from_describe(rows) == ["raw_data"]


def test_case_insensitive_type_match():
    rows = _rows(("v", "VARIANT"), ("i", "INTERVAL DAY TO SECOND"))
    assert _hostile_columns_from_describe(rows) == ["v", "i"]


def test_null_data_type_is_safe():
    # A row with a None data_type (defensive) must not crash.
    rows = [{"col_name": "x", "data_type": None}, {"col_name": "v", "data_type": "variant"}]
    assert _hostile_columns_from_describe(rows) == ["v"]


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
    assert _hostile_columns_from_describe(rows) == ["raw_data", "dur"]
