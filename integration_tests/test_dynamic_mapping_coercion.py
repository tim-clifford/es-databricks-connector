# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: ES dynamic-mapping coercion vs the connector round-trip
# MAGIC Proves the claim in the README's "Dynamic-mapping gotcha" note: when no index mapping is
# MAGIC pre-created, ES infers each field's type from the FIRST document and coerces later docs to fit
# MAGIC it for **indexing**, but keeps the original value in `_source`. The consequence:
# MAGIC
# MAGIC 1. **The connector round-trip is faithful.** `read_index` reads `_source`, so it returns the
# MAGIC    original value regardless of the coercion (a `float` into an int-mapped field, a string past
# MAGIC    `ignore_above`). This is the reassuring half: the connector loses nothing.
# MAGIC 2. **A DIRECT ES query sees the coerced/truncated value.** An aggregation reads the indexed
# MAGIC    doc-values (the coerced `long`), and a `term` on the keyword sub-field can't find a string
# MAGIC    that exceeded `ignore_above` (default 256). This is the surprise half, and it's ES behavior,
# MAGIC    not a connector bug.
# MAGIC
# MAGIC If either half regressed (e.g. the connector started reading indexed values instead of
# MAGIC `_source`, or ES changed its coercion), a test here fails. Live ES + the `es_poc` scope
# MAGIC required. Throwaway indices, dropped per run.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
import pytest
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsWriteConfig, EsReadConfig, bulk_write, read_index
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

SCOPE = "es_poc"
NUM_INDEX = "connector-integration-dynmap-numeric"   # throwaway; recreated + dropped by the fixture
STR_INDEX = "connector-integration-dynmap-ignoreabove"
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

LONG_STR = "x" * 300                                  # exceeds keyword ignore_above default (256)

# Both halves of the note are asserted for BOTH write paths: the default per-row path and
# serialize_in_spark=True. The Spark path must render a double (10.7) and a long string the same way
# in _source (JSON number / verbatim string), so ES coerces/ignore_above them identically. Each path
# gets its own pair of indices.
PATHS = ["default", "spark"]


def _idx(base, path):
    return f"{base}-{path}"


class TestDynamicMappingCoercion(NotebookTestFixture):
    """No pre-created mapping: ES dynamic-maps from the first doc and coerces later docs for INDEXING,
    but _source stays verbatim. The connector (reads _source) round-trips faithfully; a direct ES
    query (reads the indexed value) sees the coercion. Proves both halves of the README note, on both
    the default and serialize_in_spark write paths."""

    def _recreate(self, index):
        # Delete then let the FIRST written doc trigger ES dynamic mapping (no explicit mapping body).
        requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
        requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"},
                     data=json.dumps({"settings": {"index": {"number_of_shards": 1,
                                                            "number_of_replicas": 0}}}))

    def _run_path(self, path):
        """Run the numeric-coercion and ignore_above batteries for one write path; return its results."""
        sis = path == "spark"
        num_index, str_index = _idx(NUM_INDEX, path), _idx(STR_INDEX, path)

        # --- numeric coercion: int first (maps field as long), then a float that doesn't fit ---
        self._recreate(num_index)
        num_cfg = EsWriteConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                index=num_index, id_field="doc_id", http_compress=True, serialize_in_spark=sis)
        # Batch 1: score is an INTEGER -> ES dynamic-maps `score` as `long`.
        b1 = spark.sql("SELECT 'a' AS doc_id, CAST(10 AS INT) AS score")
        num_write1 = bulk_write(b1, num_cfg)
        requests.post(f"{ES_HOSTS}/{num_index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        # Batch 2: score is 10.7 (double). _source keeps 10.7; the INDEXED value is coerced to 10.
        b2 = spark.sql("SELECT 'b' AS doc_id, CAST(10.7 AS DOUBLE) AS score")
        num_write2 = bulk_write(b2, num_cfg)
        requests.post(f"{ES_HOSTS}/{num_index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)

        # Connector read-back (reads _source): doc b's score must be the verbatim 10.7.
        num_read = EsReadConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                index=num_index, id_field="doc_id", num_slices=1)
        num_schema = StructType([StructField("doc_id", StringType()),
                                 StructField("score", DoubleType())])
        num_rows = {r["doc_id"]: r.asDict() for r in
                    read_index(spark, num_read, num_schema).collect()}

        # Direct ES sum aggregation (reads the INDEXED doc-values): 10 + coerced(10.7)=10 -> 20,
        # NOT 20.7. This is the value a Kibana dashboard / raw aggregation would report.
        agg = requests.get(f"{ES_HOSTS}/{num_index}/_search", auth=ES_AUTH, verify=False, timeout=30,
                           headers={"Content-Type": "application/json"},
                           data=json.dumps({"size": 0, "aggs": {"total": {"sum": {"field": "score"}}}}))
        num_agg_sum = agg.json()["aggregations"]["total"]["value"]
        # The raw _source of doc b, straight from ES, to show _source kept 10.7 verbatim.
        gb = requests.get(f"{ES_HOSTS}/{num_index}/_doc/b", auth=ES_AUTH, verify=False, timeout=30)
        num_source_b = gb.json().get("_source", {})

        # --- ignore_above: a >256-char string dynamic-maps as text+keyword(ignore_above:256) ---
        self._recreate(str_index)
        str_cfg = EsWriteConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                index=str_index, id_field="doc_id", http_compress=True, serialize_in_spark=sis)
        s = spark.sql(f"SELECT 'a' AS doc_id, '{LONG_STR}' AS tag")
        str_write = bulk_write(s, str_cfg)
        requests.post(f"{ES_HOSTS}/{str_index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)

        # Connector read-back (reads _source): the full 300-char string comes back intact.
        str_read = EsReadConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                index=str_index, id_field="doc_id", num_slices=1)
        str_schema = StructType([StructField("doc_id", StringType()),
                                 StructField("tag", StringType())])
        str_rows = {r["doc_id"]: r.asDict() for r in
                    read_index(spark, str_read, str_schema).collect()}

        # Direct term query on the keyword sub-field for the exact 300-char value: because the value
        # exceeded ignore_above (256), it was NOT indexed into `tag.keyword`, so this finds NOTHING.
        tq = requests.get(f"{ES_HOSTS}/{str_index}/_search", auth=ES_AUTH, verify=False, timeout=30,
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"query": {"term": {"tag.keyword": LONG_STR}}}))
        str_term_hits = tq.json()["hits"]["total"]["value"]

        return {"num_write1": num_write1, "num_write2": num_write2, "num_rows": num_rows,
                "num_agg_sum": num_agg_sum, "num_source_b": num_source_b, "str_write": str_write,
                "str_rows": str_rows, "str_term_hits": str_term_hits}

    def run_setup(self):
        self.by_path = {path: self._run_path(path) for path in PATHS}

    def run_cleanup(self):
        for path in PATHS:
            requests.delete(f"{ES_HOSTS}/{_idx(NUM_INDEX, path)}", auth=ES_AUTH, verify=False, timeout=30)
            requests.delete(f"{ES_HOSTS}/{_idx(STR_INDEX, path)}", auth=ES_AUTH, verify=False, timeout=30)

    # --- both writes landed cleanly ---
    @pytest.mark.parametrize("path", PATHS)
    def test_writes_clean(self, path):
        r = self.by_path[path]
        assert r["num_write1"]["errors"] == 0 and r["num_write1"]["written"] == 1, (path, r["num_write1"])
        assert r["num_write2"]["errors"] == 0 and r["num_write2"]["written"] == 1, (path, r["num_write2"])
        assert r["str_write"]["errors"] == 0 and r["str_write"]["written"] == 1, (path, r["str_write"])

    # === NUMERIC: connector faithful, direct aggregation coerced ===
    @pytest.mark.parametrize("path", PATHS)
    def test_connector_reads_float_verbatim_from_source(self, path):
        # The reassuring half: read_index returns the original 10.7 even though the field is long-mapped.
        rows = self.by_path[path]["num_rows"]
        assert rows["a"]["score"] == 10.0, (path, rows["a"])
        assert rows["b"]["score"] == 10.7, (path, rows["b"])

    @pytest.mark.parametrize("path", PATHS)
    def test_source_kept_the_float_verbatim(self, path):
        # _source is verbatim (this is WHY the connector round-trip is faithful).
        assert self.by_path[path]["num_source_b"].get("score") == 10.7, (path, self.by_path[path]["num_source_b"])

    @pytest.mark.parametrize("path", PATHS)
    def test_direct_aggregation_sees_coerced_integer(self, path):
        # The surprise half: the indexed value of doc b was coerced to 10, so sum = 10 + 10 = 20,
        # NOT 20.7. A raw ES aggregation / dashboard would under-report. (If ES ever stopped coercing
        # and rejected the doc instead, num_write2 would have errored and test_writes_clean would fail.)
        got = self.by_path[path]["num_agg_sum"]
        assert got == 20.0, f"[{path}] expected coerced sum 20.0 (10 + truncated 10), got {got}"

    # === IGNORE_ABOVE: connector faithful, direct term query blind ===
    @pytest.mark.parametrize("path", PATHS)
    def test_connector_reads_long_string_verbatim(self, path):
        # The reassuring half: the full 300-char string round-trips (it lives in _source).
        rows = self.by_path[path]["str_rows"]
        assert rows["a"]["tag"] == LONG_STR, (path, len(rows["a"]["tag"]))

    @pytest.mark.parametrize("path", PATHS)
    def test_direct_term_query_cannot_find_the_long_string(self, path):
        # The surprise half: >256 chars was not indexed into tag.keyword, so an exact term finds nothing.
        got = self.by_path[path]["str_term_hits"]
        assert got == 0, f"[{path}] expected 0 hits (value exceeded ignore_above, not indexed), got {got}"


# COMMAND ----------
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
