# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: sanitize_for_arrow on real VARIANT / INTERVAL (live Spark)
# MAGIC The pure-Python suite (`tests/test_spark_prep.py`) only covers the regex/parsing helpers; the
# MAGIC actual Spark behavior of `sanitize_for_arrow`: `to_json` on VARIANT, `cast(string)` on scalar
# MAGIC INTERVAL, and the fact that `df.schema` THROWS on a VARIANT column under Spark Connect, can
# MAGIC only be exercised on a real serverless session. That is what this fixture does.
# MAGIC
# MAGIC No Elasticsearch needed: this tests only the Spark-side transform, not the write.

# COMMAND ----------
import json
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector.spark_prep import sanitize_for_arrow
from pyspark.sql import functions as F


class TestSanitizeForArrow(NotebookTestFixture):
    """sanitize_for_arrow rewrites Arrow-hostile columns to strings, in Spark, on serverless."""

    def run_setup(self):
        # A row exercising VARIANT (top-level, nested-in-struct, nested-in-array) and INTERVAL
        # (scalar day-time and year-month) alongside ordinary columns that must pass through
        # untouched. Built via SQL so the column TYPES are exactly the Spark types.
        self.df = spark.sql("""
            SELECT
              'r1'                                          AS id,
              42                                            AS plain_int,
              named_struct('ip','10.0.0.1','port',443)      AS plain_struct,
              parse_json('{"k":1,"nested":[2,3]}')          AS v_top,
              named_struct('v', parse_json('{"a":true}'),
                           'label','hi')                    AS v_in_struct,
              array(parse_json('{"n":1}'),
                    parse_json('{"n":2}'))                  AS v_in_array,
              INTERVAL '1 02:03:04' DAY TO SECOND           AS iv_daytime,
              INTERVAL '2-3' YEAR TO MONTH                  AS iv_yearmonth
        """)
        self.out = sanitize_for_arrow(self.df)
        # Materialize the sanitized rows once. If sanitize missed a hostile column, .collect()
        # (Arrow conversion) would raise here, so a clean collect is itself part of the contract.
        self.rows = self.out.collect()
        self.row = self.rows[0].asDict()

    def test_raw_variant_df_schema_throws_on_connect(self):
        # The constraint that forces sanitize_for_arrow to use DESCRIBE instead of df.schema:
        # touching .schema on a VARIANT-bearing DataFrame raises under Spark Connect. If this ever
        # STOPS throwing, the connector's DESCRIBE workaround could be simplified, so we assert it.
        err = None
        try:
            _ = self.df.schema  # VARIANT column present -> UNSUPPORTED_OPERATION on Connect
        except Exception as e:
            err = e
        assert err is not None, "df.schema unexpectedly succeeded on a VARIANT column, revisit spark_prep"
        # Assert it's specifically the unsupported-type error, not just any failure, a DIFFERENT
        # error would mean something else broke and this test should not quietly pass. Match on the
        # stable error-class token rather than a version-specific exception type.
        assert "UNSUPPORTED_OPERATION" in str(err), \
            f"df.schema raised, but not the expected UNSUPPORTED_OPERATION: {type(err).__name__}: {err}"

    def test_sanitized_df_is_arrow_collectable(self):
        # The whole point: after sanitize, every column is Arrow-safe and the df collects.
        assert len(self.rows) == 1

    def test_plain_columns_untouched(self):
        # Non-hostile columns must pass through unchanged (not stringified).
        assert self.row["id"] == "r1"
        assert self.row["plain_int"] == 42
        assert self.row["plain_struct"]["ip"] == "10.0.0.1"
        assert self.row["plain_struct"]["port"] == 443

    def test_top_level_variant_becomes_json_string(self):
        v = self.row["v_top"]
        assert isinstance(v, str), f"expected JSON string, got {type(v).__name__}"
        assert json.loads(v) == {"k": 1, "nested": [2, 3]}

    def test_variant_nested_in_struct_serializes_whole_column(self):
        # A struct CONTAINING a variant is serialized whole (DESCRIBE can't navigate to the inner
        # field), so the entire column lands as one JSON string, documented behavior.
        v = self.row["v_in_struct"]
        assert isinstance(v, str)
        assert json.loads(v) == {"v": {"a": True}, "label": "hi"}

    def test_variant_nested_in_array_serializes_whole_column(self):
        v = self.row["v_in_array"]
        assert isinstance(v, str)
        assert json.loads(v) == [{"n": 1}, {"n": 2}]

    def test_scalar_intervals_become_spark_string_form(self):
        # Scalar INTERVAL can't go through to_json (Spark rejects it), so sanitize uses cast(string).
        assert self.row["iv_daytime"] == "INTERVAL '1 02:03:04' DAY TO SECOND"
        assert self.row["iv_yearmonth"] == "INTERVAL '2-3' YEAR TO MONTH"

    def test_idempotent_on_already_clean_df(self):
        # A DataFrame with no hostile columns is returned unchanged (same columns, still collectable).
        clean = spark.sql("SELECT 'x' AS id, 1 AS n")
        out = sanitize_for_arrow(clean)
        assert out.columns == ["id", "n"]
        assert out.collect()[0].asDict() == {"id": "x", "n": 1}


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
