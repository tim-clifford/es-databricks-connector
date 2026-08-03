# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: write → read round-trip (bulk_write then read_index) live on Spark + ES
# MAGIC The payoff test for the read path: take a Spark DataFrame, write it to ES with `bulk_write`,
# MAGIC read it back with `read_index` using the **same schema**, and assert the round-tripped rows
# MAGIC equal the originals, **except** the deltas the README documents as one-way (decimal
# MAGIC precision, sub-millisecond timestamp, float32 widening). This proves the read coercion layer
# MAGIC (`read_coerce`) is the true inverse of the write transform against real Elasticsearch, not
# MAGIC just in the offline unit oracle.
# MAGIC
# MAGIC Exercises the distributed sliced-scroll `read_index` across three shapes: default fan-out,
# MAGIC multi-slice fan-out over a 3-shard index, and a single-slice (`num_slices=1`) multi-PAGE read
# MAGIC that drives the sliding-window PIT keep-alive against real ES. Live ES + the `es_poc` scope
# MAGIC required. Throwaway indices, dropped per run.

# COMMAND ----------
import json, base64, datetime, requests, urllib3
urllib3.disable_warnings()
from decimal import Decimal
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import (
    EsWriteConfig, EsReadConfig, bulk_write, read_index,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, LongType, IntegerType, DoubleType,
    DecimalType, DateType, TimestampType, TimestampNTZType, BinaryType, ArrayType,
)

SCOPE = "es_poc"
INDEX = "connector-integration-read-roundtrip"       # throwaway; recreated + dropped by the fixture
MULTI_INDEX = "connector-integration-read-multishard"  # 3-shard, to exercise sliced-scroll fan-out
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# The declared schema the reader must be given (v0.4.0: no inference). Covers the invertible types
# and the documented-lossy ones. VARIANT/INTERVAL are excluded here: they read back as strings and
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
    # A sub-millisecond timestamp: proves the documented microsecond->millisecond floor survives a
    # full live round-trip (the unit test asserts the floor offline; this asserts it through ES).
    StructField("s_ts_subms", TimestampType()),
    # timestamp_ntz: the read inverse added alongside this work. Reads back NAIVE (no tzinfo),
    # exercising read_coerce's timestamp_ntz branch end-to-end, not just in the unit oracle.
    StructField("s_ts_ntz", TimestampNTZType()),
    # A high-precision decimal CAST TO STRING in Spark: the documented workaround to preserve
    # exactness past double's ~15-17 sig figs. Declared StringType on read, must equal the digits.
    StructField("s_decimal_exact_str", StringType()),
    StructField("s_binary", BinaryType()),
    StructField("s_array", ArrayType(IntegerType())),
    StructField("s_struct", StructType([
        StructField("ip", StringType()),
        StructField("port", IntegerType()),
        # A DECIMAL nested inside a struct: its simpleString() is decimal(10,2), whose inner comma
        # regression-tests the DDL token splitter (a nested decimal previously corrupted struct
        # field parsing, see tests/test_read_transform.py::test_struct_with_nested_decimal...).
        StructField("weight", DecimalType(10, 2)),
    ])),
])


class TestReadRoundtrip(NotebookTestFixture):
    """write(df) then read_index(df.schema) reproduces the original rows, modulo documented deltas."""

    def run_setup(self):
        self.write_cfg = EsWriteConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                       index=INDEX, id_field="doc_id", http_compress=True)
        # Read config shares the same connection; carries the source index + paging.
        self.read_cfg = EsReadConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                     index=INDEX, id_field="doc_id", batch_size=100)

        # Explicit CASTs so the source column TYPES match SCHEMA exactly.
        self.src = spark.sql("""
            SELECT
              'r1' AS doc_id, true AS s_bool, CAST(70000 AS INT) AS s_int,
              CAST(9223372036854775807 AS BIGINT) AS s_long, CAST(1.5 AS DOUBLE) AS s_double,
              CAST(1.50 AS DECIMAL(10,2)) AS s_decimal, DATE'2021-01-01' AS s_date,
              TIMESTAMP'2021-01-01 12:30:00Z' AS s_ts,
              TIMESTAMP'2021-01-01 00:00:00.123456Z' AS s_ts_subms,
              TIMESTAMP_NTZ'2021-06-01 12:00:00' AS s_ts_ntz,
              CAST(CAST(123456789012345678 AS DECIMAL(38,0)) AS STRING) AS s_decimal_exact_str,
              CAST(X'0102' AS BINARY) AS s_binary,
              array(1,2,3) AS s_array,
              named_struct('ip','10.0.0.1','port',443,'weight',CAST(1.25 AS DECIMAL(10,2))) AS s_struct
            UNION ALL
              SELECT 'r2', false, CAST(-5 AS INT), CAST(0 AS BIGINT), CAST(2.25 AS DOUBLE),
              CAST(99.99 AS DECIMAL(10,2)), DATE'1999-12-31',
              TIMESTAMP'2000-01-01 00:00:00Z',
              TIMESTAMP'1969-12-31 23:59:59.999999Z',
              TIMESTAMP_NTZ'1999-12-31 23:59:58',
              CAST(CAST(-98765432109876543 AS DECIMAL(38,0)) AS STRING),
              CAST(X'FF' AS BINARY),
              array(), named_struct('ip','192.168.0.1','port',8080,'weight',CAST(9.99 AS DECIMAL(10,2)))
        """)

        # Fresh index. Map the types ES needs: doc_id keyword, dates as epoch_millis, binary/struct
        # dynamic. (An explicit date mapping isn't strictly required, the connector sends
        # epoch-millis integers, but it makes the index self-describing.)
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {
                    "doc_id": {"type": "keyword"},
                    "s_date": {"type": "date", "format": "epoch_millis"},
                    "s_ts": {"type": "date", "format": "epoch_millis"},
                    "s_ts_subms": {"type": "date", "format": "epoch_millis"},
                    "s_ts_ntz": {"type": "date", "format": "epoch_millis"},
                    "s_decimal_exact_str": {"type": "keyword"},   # exact-decimal-as-string
                }}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        self.write_result = bulk_write(self.src, self.write_cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)

        # Read back with the SAME schema via read_index (distributed mapInPandas fan-out); assert the
        # round-tripped rows against the source.
        self.out = read_index(spark, self.read_cfg, SCHEMA)
        self.src_rows = {r["doc_id"]: r.asDict(recursive=True) for r in self.src.collect()}
        self.out_rows = {r["doc_id"]: r.asDict(recursive=True) for r in self.out.collect()}

        # --- multi-shard index: exercise REAL sliced-scroll fan-out (>1 slice) ---
        # A 3-shard index so read_index defaults to 3 slices and each task reads a disjoint slice.
        requests.delete(f"{ES_HOSTS}/{MULTI_INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        requests.put(f"{ES_HOSTS}/{MULTI_INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"},
                     data=json.dumps({"settings": {"index": {"number_of_shards": 3,
                                                             "number_of_replicas": 0}},
                                      "mappings": {"properties": {"doc_id": {"type": "keyword"}}}}))
        many = spark.range(0, 50).selectExpr("concat('m', id) AS doc_id", "CAST(id AS INT) AS n")
        multi_write = EsWriteConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                    index=MULTI_INDEX, id_field="doc_id", http_compress=True)
        self.multi_write_result = bulk_write(many, multi_write)
        requests.post(f"{ES_HOSTS}/{MULTI_INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        multi_read = EsReadConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                  index=MULTI_INDEX, id_field="doc_id", pit_keep_alive="5m")
        multi_schema = StructType([StructField("doc_id", StringType()),
                                   StructField("n", IntegerType())])
        multi_out = read_index(spark, multi_read, multi_schema)
        self.multi_ids = {r["doc_id"] for r in multi_out.collect()}

        # --- multi-PAGE read: force search_after paging over ONE PIT (sliding-window keep_alive) ---
        # The reads above each fit in a single page (default/large batch_size), so they never page.
        # Read the same 50-doc index with batch_size=5 and a single slice: ~10 sequential pages over
        # one Point-in-Time. Each page re-sends pit_keep_alive and follows the refreshed pit_id, so a
        # complete, gap-free, duplicate-free result proves the paging + sliding-window PIT extension
        # (unit-guarded in tests/test_read.py) actually holds against real ES. A short keep_alive is
        # deliberate: it must survive across pages BECAUSE each page extends it, not because it's long.
        paged_read = EsReadConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                  index=MULTI_INDEX, id_field="doc_id",
                                  num_slices=1, batch_size=5, pit_keep_alive="1m")
        paged_out = read_index(spark, paged_read, multi_schema)
        self.paged_ids = [r["doc_id"] for r in paged_out.collect()]

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        requests.delete(f"{ES_HOSTS}/{MULTI_INDEX}", auth=ES_AUTH, verify=False, timeout=30)

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

    def test_timestamp_ntz_roundtrip_naive(self):
        # timestamp_ntz reads back NAIVE (no tzinfo) and equal to the source wall-clock. Proves the
        # read_coerce timestamp_ntz branch works against real ES, not just in the unit oracle.
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[did]
            assert o["s_ts_ntz"] == s["s_ts_ntz"], f"{did}.s_ts_ntz: {o['s_ts_ntz']!r} != {s['s_ts_ntz']!r}"
            assert o["s_ts_ntz"].tzinfo is None, f"{did}.s_ts_ntz should be naive, got {o['s_ts_ntz']!r}"

    def test_subms_timestamp_floors_to_ms(self):
        # DOCUMENTED delta: microsecond precision is floored to the millisecond on write. The round-
        # trip must equal the source FLOORED to ms (not the original micros), proven through live ES.
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[did]
            src_ts = s["s_ts_subms"]
            floored = src_ts.replace(microsecond=(src_ts.microsecond // 1000) * 1000)
            assert o["s_ts_subms"] == floored, \
                f"{did}.s_ts_subms: {o['s_ts_subms']!r} != floored {floored!r} (src {src_ts!r})"

    def test_high_precision_decimal_via_string_is_exact(self):
        # DOCUMENTED workaround: casting a high-precision decimal to STRING in Spark before writing
        # preserves exactness past double's ~15-17 sig figs. Read back as StringType, the 18-digit
        # value must be exact -- unlike s_decimal_hi in test_datatype_coverage, which loses low digits
        # because it goes through float. This proves the mitigation the README recommends.
        assert self.out_rows["r1"]["s_decimal_exact_str"] == "123456789012345678"
        assert self.out_rows["r2"]["s_decimal_exact_str"] == "-98765432109876543"

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

    # --- multi-shard: sliced-scroll fan-out reads every doc exactly once, no gaps/dupes ---
    def test_multishard_slices_cover_all_docs(self):
        # 50 docs across 3 shards => 3 slices; the union must be exactly the 50 ids, no loss or
        # duplication across slices (the core correctness property of a sliced read).
        assert self.multi_write_result["written"] == 50, self.multi_write_result
        expected = {f"m{i}" for i in range(50)}
        assert self.multi_ids == expected, (len(self.multi_ids), len(expected))

    # --- multi-page: search_after paging over one PIT reads every doc once, across ~10 pages ---
    def test_multipage_paging_reads_all_docs_once(self):
        # 50 docs at batch_size=5 => ~10 sequential pages on one PIT. The result must be exactly the
        # 50 ids with NO duplicates (a broken search_after or an expired PIT mid-read would drop or
        # repeat docs). len == set size asserts no duplication; the set equality asserts no loss.
        expected = {f"m{i}" for i in range(50)}
        assert len(self.paged_ids) == 50, f"expected 50 docs across pages, got {len(self.paged_ids)}"
        assert set(self.paged_ids) == expected, set(self.paged_ids) ^ expected


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
