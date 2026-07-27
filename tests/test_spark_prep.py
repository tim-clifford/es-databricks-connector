"""Unit tests for spark_prep's Arrow-hostile-column detection.

The Spark-touching parts (createOrReplaceTempView / DESCRIBE / to_json) need a session and are
exercised by the OCSF round-trip demo on a live cluster. Here we test the PURE logic —
_hostile_columns_from_describe — which is where the detection rules and edge cases live, using
synthetic DESCRIBE output. This is the regression guard for the VARIANT/INTERVAL export bug:
before the fix, a DataFrame with a VARIANT column crashed bulk_write; the fix detects such columns
here and serializes them.
"""
from databricks_es_connector.spark_prep import _hostile_columns_from_describe


def _rows(*pairs):
    """Build DESCRIBE-style rows from (col_name, data_type) pairs."""
    return [{"col_name": n, "data_type": t} for n, t in pairs]


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
