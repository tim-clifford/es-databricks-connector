# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: write → read round-trip (bulk_write then read_index) live on Spark + ES
# MAGIC The payoff test for the read path: take a Spark DataFrame, write it to ES with `bulk_write`,
# MAGIC read it back with `read_index` using the **same schema**, and assert the round-tripped rows
# MAGIC equal the originals — **except** the deltas the README documents as one-way (decimal
# MAGIC precision, sub-millisecond timestamp, float32 widening). This proves the read coercion layer
# MAGIC (`read_coerce`) is the true inverse of the write transform against real Elasticsearch, not
# MAGIC just in the offline unit oracle.
# MAGIC
# MAGIC v0.4.0 spike: uses the DRIVER-SIDE reader (Option A). The distributed sliced-scroll reader
# MAGIC (Option B) will reuse the same coercion layer, so this round-trip stays the acceptance bar.
# MAGIC Live ES + the `es_poc` scope required. Throwaway index, dropped per run.

# COMMAND ----------
import json, base64, datetime, requests, urllib3
urllib3.disable_warnings()
from decimal import Decimal
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, EsReadConfig, bulk_write, read_index
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, LongType, IntegerType, DoubleType,
    DecimalType, DateType, TimestampType, BinaryType, ArrayType,
)

SCOPE = "es_poc"
INDEX = "connector-integration-read-roundtrip"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# The declared schema the reader must be given (v0.4.0: no inference). Covers the invertible types
# and the documented-lossy ones. VARIANT/INTERVAL are excluded here — they read back as strings and
# are covered by test_datatype_coverage / test_sanitize_for_arrow; this fixture is about the typed
# write->read inverse.
SCHEMA = StructType([
    StructField("doc_id", StringType()),
    StructField("s_bool", BooleanType()),
    StructField("s_int", IntegerType()),
    StructField("s_long", LongType()),
    StructField("s_double", DoubleType()),
    StructField("s_decimal", DecimalType(10, 2)),
    StructField("s_date", DateType()),
    StructField("s_ts", TimestampType()),
    StructField("s_binary", BinaryType()),
    StructField("s_array", ArrayType(IntegerType())),
    StructField("s_struct", StructType([
        StructField("ip", StringType()),
        StructField("port", IntegerType()),
    ])),
])


class TestReadRoundtrip(NotebookTestFixture):
    """write(df) then read_index(df.schema) reproduces the original rows, modulo documented deltas."""

    def run_setup(self):
        self.write_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                  index=INDEX, id_field="doc_id", http_compress=True)

        # Explicit CASTs so the source column TYPES match SCHEMA exactly.
        self.src = spark.sql("""
            SELECT
              'r1' AS doc_id, true AS s_bool, CAST(70000 AS INT) AS s_int,
              CAST(9223372036854775807 AS BIGINT) AS s_long, CAST(1.5 AS DOUBLE) AS s_double,
              CAST(1.50 AS DECIMAL(10,2)) AS s_decimal, DATE'2021-01-01' AS s_date,
              TIMESTAMP'2021-01-01 12:30:00Z' AS s_ts, CAST(X'0102' AS BINARY) AS s_binary,
              array(1,2,3) AS s_array, named_struct('ip','10.0.0.1','port',443) AS s_struct
            UNION ALL
              SELECT 'r2', false, CAST(-5 AS INT), CAST(0 AS BIGINT), CAST(2.25 AS DOUBLE),
              CAST(99.99 AS DECIMAL(10,2)), DATE'1999-12-31',
              TIMESTAMP'2000-01-01 00:00:00Z', CAST(X'FF' AS BINARY),
              array(), named_struct('ip','192.168.0.1','port',8080)
        """)

        # Fresh index. Map the types ES needs: doc_id keyword, dates as epoch_millis, binary/struct
        # dynamic. (An explicit date mapping isn't strictly required — the connector sends
        # epoch-millis integers — but it makes the index self-describing.)
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {
                    "doc_id": {"type": "keyword"},
                    "s_date": {"type": "date", "format": "epoch_millis"},
                    "s_ts": {"type": "date", "format": "epoch_millis"},
                }}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        self.write_result = bulk_write(self.src, self.write_cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)

        # Read back with the SAME schema via the driver-side reader.
        self.out = read_index(spark, self.write_cfg, SCHEMA, read=EsReadConfig(batch_size=100))
        # Collect both sides into {doc_id: Row.asDict()} for order-independent comparison.
        self.src_rows = {r["doc_id"]: r.asDict(recursive=True) for r in self.src.collect()}
        self.out_rows = {r["doc_id"]: r.asDict(recursive=True) for r in self.out.collect()}

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- the write landed, the read returned the same rows/schema ---
    def test_write_clean(self):
        assert self.write_result["errors"] == 0, self.write_result
        assert self.write_result["written"] == 2, self.write_result

    def test_read_returns_both_rows(self):
        assert set(self.out_rows) == {"r1", "r2"}, self.out_rows

    def test_read_schema_matches_declared(self):
        # The returned DataFrame's schema is exactly the one we asked for.
        assert self.out.schema == SCHEMA, self.out.schema

    # --- exact round-trip for the invertible types ---
    def test_scalars_roundtrip_exactly(self):
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[did]
            for col in ("doc_id", "s_bool", "s_int", "s_long", "s_double"):
                assert o[col] == s[col], f"{did}.{col}: {o[col]!r} != {s[col]!r}"

    def test_date_and_timestamp_roundtrip(self):
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[did]
            assert o["s_date"] == s["s_date"], f"{did}.s_date: {o['s_date']!r} != {s['s_date']!r}"
            assert o["s_ts"] == s["s_ts"], f"{did}.s_ts: {o['s_ts']!r} != {s['s_ts']!r}"

    def test_binary_roundtrip(self):
        for did in ("r1", "r2"):
            assert self.out_rows[did]["s_binary"] == self.src_rows[did]["s_binary"], did

    def test_array_and_struct_roundtrip(self):
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[did]
            assert o["s_array"] == s["s_array"], f"{did}.s_array: {o['s_array']!r} != {s['s_array']!r}"
            assert o["s_struct"] == s["s_struct"], f"{did}.s_struct: {o['s_struct']!r} != {s['s_struct']!r}"

    # --- decimal: within DecimalType(10,2) there is no loss, so this round-trips exactly ---
    def test_decimal_roundtrip_within_scale(self):
        # 1.50 and 99.99 both fit in decimal(10,2); read back as Decimal, equal to source.
        assert self.out_rows["r1"]["s_decimal"] == self.src_rows["r1"]["s_decimal"]
        assert self.out_rows["r2"]["s_decimal"] == self.src_rows["r2"]["s_decimal"]


# COMMAND ----------
# Explicit fixture class (no-arg auto-discovery finds nothing through the wrapper).
dbutils.notebook.exit(json.dumps(run_notebook_tests(TestReadRoundtrip)))
