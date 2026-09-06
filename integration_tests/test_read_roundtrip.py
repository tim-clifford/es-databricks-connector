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
import pytest
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

# The typed write->read round-trip is asserted for BOTH write paths with IDENTICAL expectations: the
# default per-row Python path and serialize_in_spark=True (JVM to_json). read_index reads _source and
# coerces to the declared schema, so the contract (value in == value out, modulo documented deltas)
# must hold whichever path produced the _source, including date/timestamp/timestamp_ntz, which the
# Spark path converts to epoch-millis in build_ndjson as of 0.8.1. Each path writes its own index.
PATHS = ["default", "spark"]
INDEX_BY_PATH = {"default": INDEX, "spark": INDEX + "-sis"}

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

    def _write_read_path(self, path):
        """Write self.src to this path's index (serialize_in_spark toggled), read it back with SCHEMA,
        and return (write_result, {doc_id: row_dict}). Same expectations hold for both paths."""
        index = INDEX_BY_PATH[path]
        write_cfg = EsWriteConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                  index=index, id_field="doc_id", http_compress=True,
                                  serialize_in_spark=(path == "spark"))
        read_cfg = EsReadConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                index=index, id_field="doc_id", batch_size=100)
        requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {
                    "doc_id": {"type": "keyword"},
                    "s_date": {"type": "date", "format": "epoch_millis"},
                    "s_ts": {"type": "date", "format": "epoch_millis"},
                    "s_ts_subms": {"type": "date", "format": "epoch_millis"},
                    "s_ts_ntz": {"type": "date", "format": "epoch_millis"},
                    "s_decimal_exact_str": {"type": "keyword"},   # exact-decimal-as-string
                }}}
        requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))
        result = bulk_write(self.src, write_cfg)
        requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        out = read_index(spark, read_cfg, SCHEMA)
        self.out_schema[path] = out.schema
        return result, {r["doc_id"]: r.asDict(recursive=True) for r in out.collect()}

    def run_setup(self):
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

        # Write + read back through BOTH paths (fresh per-path index each). The type mapping (doc_id
        # keyword, dates as epoch_millis, exact-decimal-as-string keyword) lives in _write_read_path.
        self.src_rows = {r["doc_id"]: r.asDict(recursive=True) for r in self.src.collect()}
        self.write_result = {}      # per path
        self.out_rows = {}          # per path: {doc_id: row_dict}
        self.out_schema = {}        # per path: read-back DataFrame schema
        for path in PATHS:
            self.write_result[path], self.out_rows[path] = self._write_read_path(path)

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
        for path in PATHS:
            requests.delete(f"{ES_HOSTS}/{INDEX_BY_PATH[path]}", auth=ES_AUTH, verify=False, timeout=30)
        requests.delete(f"{ES_HOSTS}/{MULTI_INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- the write landed, the read returned the same rows/schema (both write paths) ---
    @pytest.mark.parametrize("path", PATHS)
    def test_write_clean(self, path):
        assert self.write_result[path]["errors"] == 0, (path, self.write_result[path])
        assert self.write_result[path]["written"] == 2, (path, self.write_result[path])

    @pytest.mark.parametrize("path", PATHS)
    def test_read_returns_both_rows(self, path):
        assert set(self.out_rows[path]) == {"r1", "r2"}, (path, self.out_rows[path])

    @pytest.mark.parametrize("path", PATHS)
    def test_read_schema_matches_declared(self, path):
        # The returned DataFrame's schema is exactly the one we asked for.
        assert self.out_schema[path] == SCHEMA, (path, self.out_schema[path])

    # --- exact round-trip for the invertible types ---
    @pytest.mark.parametrize("path", PATHS)
    def test_scalars_roundtrip_exactly(self, path):
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[path][did]
            for col in ("doc_id", "s_bool", "s_int", "s_long", "s_double"):
                assert o[col] == s[col], f"[{path}] {did}.{col}: {o[col]!r} != {s[col]!r}"

    @pytest.mark.parametrize("path", PATHS)
    def test_date_and_timestamp_roundtrip(self, path):
        # The 0.8.1 lock at the CONTRACT level: date + timestamp round-trip identically whether the
        # _source was built by the default per-row path or serialize_in_spark's build_ndjson.
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[path][did]
            assert o["s_date"] == s["s_date"], f"[{path}] {did}.s_date: {o['s_date']!r} != {s['s_date']!r}"
            assert o["s_ts"] == s["s_ts"], f"[{path}] {did}.s_ts: {o['s_ts']!r} != {s['s_ts']!r}"

    @pytest.mark.parametrize("path", PATHS)
    def test_timestamp_ntz_roundtrip_naive(self, path):
        # timestamp_ntz reads back NAIVE (no tzinfo) and equal to the source wall-clock, on BOTH paths.
        # Before 0.8.1 the Spark path stored an ISO string here and this round-trip broke.
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[path][did]
            assert o["s_ts_ntz"] == s["s_ts_ntz"], f"[{path}] {did}.s_ts_ntz: {o['s_ts_ntz']!r} != {s['s_ts_ntz']!r}"
            assert o["s_ts_ntz"].tzinfo is None, f"[{path}] {did}.s_ts_ntz should be naive, got {o['s_ts_ntz']!r}"

    @pytest.mark.parametrize("path", PATHS)
    def test_subms_timestamp_floors_to_ms(self, path):
        # DOCUMENTED delta: microsecond precision is floored to the millisecond on write. The round-
        # trip must equal the source FLOORED to ms (not the original micros), proven through live ES.
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[path][did]
            src_ts = s["s_ts_subms"]
            floored = src_ts.replace(microsecond=(src_ts.microsecond // 1000) * 1000)
            assert o["s_ts_subms"] == floored, \
                f"[{path}] {did}.s_ts_subms: {o['s_ts_subms']!r} != floored {floored!r} (src {src_ts!r})"

    @pytest.mark.parametrize("path", PATHS)
    def test_high_precision_decimal_via_string_is_exact(self, path):
        # DOCUMENTED workaround: casting a high-precision decimal to STRING in Spark before writing
        # preserves exactness past double's ~15-17 sig figs. Read back as StringType, the 18-digit
        # value must be exact -- unlike s_decimal_hi in test_datatype_coverage, which loses low digits
        # because it goes through float. This proves the mitigation the README recommends.
        assert self.out_rows[path]["r1"]["s_decimal_exact_str"] == "123456789012345678", path
        assert self.out_rows[path]["r2"]["s_decimal_exact_str"] == "-98765432109876543", path

    @pytest.mark.parametrize("path", PATHS)
    def test_binary_roundtrip(self, path):
        for did in ("r1", "r2"):
            assert self.out_rows[path][did]["s_binary"] == self.src_rows[did]["s_binary"], (path, did)

    @pytest.mark.parametrize("path", PATHS)
    def test_array_and_struct_roundtrip(self, path):
        for did in ("r1", "r2"):
            s, o = self.src_rows[did], self.out_rows[path][did]
            assert o["s_array"] == s["s_array"], f"[{path}] {did}.s_array: {o['s_array']!r} != {s['s_array']!r}"
            assert o["s_struct"] == s["s_struct"], f"[{path}] {did}.s_struct: {o['s_struct']!r} != {s['s_struct']!r}"

    # --- decimal: within DecimalType(10,2) there is no loss, so this round-trips exactly (both paths) ---
    @pytest.mark.parametrize("path", PATHS)
    def test_decimal_roundtrip_within_scale(self, path):
        # 1.50 and 99.99 both fit in decimal(10,2); read back as Decimal, equal to source.
        assert self.out_rows[path]["r1"]["s_decimal"] == self.src_rows["r1"]["s_decimal"], path
        assert self.out_rows[path]["r2"]["s_decimal"] == self.src_rows["r2"]["s_decimal"], path

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
