# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: the bulk_write result contract (live mapInPandas + ES)
# MAGIC Owns the **write-result** side of the round-trip: that `bulk_write` returns the 0.3.1 contract
# MAGIC (`written` / `deleted` / `errors` / `total_input` / `error_samples`) correctly against real
# MAGIC serverless Spark + ES, including a deliberately-rejected doc so `errors` and `error_samples`
# MAGIC are exercised (not just the clean path), and idempotent re-write via a deterministic `_id`.
# MAGIC
# MAGIC Datatype fidelity (every Spark type out == in) lives in `test_datatype_coverage.py`; this
# MAGIC fixture does not re-assert per-type transforms. Live ES + the `es_poc` scope required.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
import pytest
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-roundtrip"       # throwaway; recreated + dropped by the fixture
DUP_INDEX = INDEX + "-dup"                       # separate throwaway for the duplicate-id case
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# The write-result contract is asserted for BOTH write paths with IDENTICAL expectations: the default
# per-row Python path and serialize_in_spark=True (JVM to_json). The two paths reuse the same per-doc
# result classification (classify_bulk_result), so counts / reconciliation / error_samples must match.
# Each path gets its own index (and dup index) so the two runs never collide.
PATHS = ["default", "spark"]


def _idx(base, path):
    return f"{base}-{path}"


class TestBulkWriteResultContract(NotebookTestFixture):
    """bulk_write's return dict is correct end-to-end: clean write, idempotent re-write, and a
    write with a doc ES rejects so errors + error_samples are populated (not just the happy path).
    Every case runs on both the default and serialize_in_spark write paths."""

    def _run_path(self, path):
        """Run the full battery for one write path; return a dict of its results."""
        sis = path == "spark"
        index, dup_index = _idx(INDEX, path), _idx(DUP_INDEX, path)
        cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                       index=index, id_field="doc_id", http_compress=True, serialize_in_spark=sis)

        # A strict mapping so we can force a REJECTED doc: n is an integer field, so a row whose n is
        # a non-numeric string fails to index. This exercises the error path deterministically.
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
        requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        good = spark.sql("""
            SELECT 'd1' AS doc_id, 1 AS n UNION ALL
            SELECT 'd2', 2 UNION ALL
            SELECT 'd3', 3
        """)
        res_clean = bulk_write(good, cfg)
        # Re-write the SAME rows: deterministic _id => upsert, not duplicate.
        res_idem = bulk_write(good, cfg)
        requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        es_count_after_idem = requests.get(
            f"{ES_HOSTS}/{index}/_count", auth=ES_AUTH, verify=False, timeout=30).json()["count"]

        # A batch with one doc ES will reject. n is a STRING column (both rows cast to string, so the
        # UNION's common type is string, otherwise Spark coerces to BIGINT and 'not-an-int' fails to
        # cast inside Spark before the connector ever runs). The connector sends both as JSON strings;
        # under the integer mapping ES coerces "10" to the int 10 (indexes) but rejects "not-an-int"
        # (mapper_parsing_exception), the deterministic ES-side error this test wants.
        mixed = spark.sql("""
            SELECT 'ok1' AS doc_id, CAST('10' AS STRING) AS n UNION ALL
            SELECT 'bad1', CAST('not-an-int' AS STRING)
        """)
        res_mixed = bulk_write(mixed, cfg)

        # DUPLICATE _id within one input. Two rows share doc_id 'dup' => the deterministic _id
        # makes the second upsert OVER the first. Every op succeeds (written == total_input == 3)
        # and the reconciliation identity holds, yet ES ends up with FEWER docs than rows fed in.
        # A client exporting a table with non-unique id_field values sees a lower doc count with no
        # error signal: this pins that documented behavior. Fresh index so the count is unambiguous.
        # CREATE it explicitly rather than leaning on ES auto-creation: `require_existing_index`
        # (default True) rejects a write to a missing index, so an auto-created one would fail this
        # setup, and a setup failure reports as zero tests run rather than as a failure.
        requests.delete(f"{ES_HOSTS}/{dup_index}", auth=ES_AUTH, verify=False, timeout=30)
        requests.put(f"{ES_HOSTS}/{dup_index}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))
        dup_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                           index=dup_index, id_field="doc_id", http_compress=True, serialize_in_spark=sis)
        dup = spark.sql("""
            SELECT 'uniq' AS doc_id, 1 AS n UNION ALL
            SELECT 'dup', 2 UNION ALL
            SELECT 'dup', 3
        """)
        res_dup = bulk_write(dup, dup_cfg)
        requests.post(f"{ES_HOSTS}/{dup_index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        dup_es_count = requests.get(
            f"{ES_HOSTS}/{dup_index}/_count", auth=ES_AUTH, verify=False, timeout=30).json()["count"]

        return {"clean": res_clean, "idem": res_idem, "count_after_idem": es_count_after_idem,
                "mixed": res_mixed, "dup": res_dup, "dup_count": dup_es_count}

    def run_setup(self):
        self.by_path = {path: self._run_path(path) for path in PATHS}

    def run_cleanup(self):
        # Reference the module constants (not setup-time instance state) so cleanup runs correctly
        # even if run_setup raised before finishing, no masking AttributeError on self.dup_index.
        for path in PATHS:
            requests.delete(f"{ES_HOSTS}/{_idx(INDEX, path)}", auth=ES_AUTH, verify=False, timeout=30)
            requests.delete(f"{ES_HOSTS}/{_idx(DUP_INDEX, path)}", auth=ES_AUTH, verify=False, timeout=30)

    # --- clean write ---
    @pytest.mark.parametrize("path", PATHS)
    def test_clean_write_counts(self, path):
        r = self.by_path[path]["clean"]
        assert r["written"] == 3, (path, r)
        assert r["errors"] == 0, (path, r)
        assert r["deleted"] == 0, (path, r)

    @pytest.mark.parametrize("path", PATHS)
    def test_clean_write_reconciles(self, path):
        r = self.by_path[path]["clean"]
        assert r["total_input"] == 3, (path, r)
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], (path, r)

    @pytest.mark.parametrize("path", PATHS)
    def test_clean_write_no_error_samples(self, path):
        assert self.by_path[path]["clean"]["error_samples"] == [], path

    # --- idempotency ---
    @pytest.mark.parametrize("path", PATHS)
    def test_idempotent_rewrite_no_duplicates(self, path):
        # Second write of the same _ids upserts; ES still holds exactly 3 docs.
        assert self.by_path[path]["idem"]["written"] == 3, (path, self.by_path[path]["idem"])
        assert self.by_path[path]["count_after_idem"] == 3, (path, self.by_path[path]["count_after_idem"])

    # --- duplicate _id within one input: collapses to one doc, counts still "succeed" ---
    @pytest.mark.parametrize("path", PATHS)
    def test_duplicate_id_collapses_but_counts_succeed(self, path):
        r = self.by_path[path]["dup"]
        # 3 rows in, 2 distinct ids => all 3 ops report success (written == total_input == 3)...
        assert r["written"] == 3, (path, r)
        assert r["errors"] == 0, (path, r)
        assert r["total_input"] == 3, (path, r)
        # ...and the reconciliation identity HOLDS, so it gives no warning...
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], (path, r)
        # ...yet ES holds only 2 docs: the duplicate id upserted over itself. This is the client
        # gotcha: fewer docs than input rows, with no error to signal it.
        assert self.by_path[path]["dup_count"] == 2, (path, self.by_path[path]["dup_count"])

    # --- error path: a rejected doc must be counted AND sampled, good docs still written ---
    @pytest.mark.parametrize("path", PATHS)
    def test_rejected_doc_counted(self, path):
        r = self.by_path[path]["mixed"]
        assert r["written"] == 1, (path, r)   # the good row indexed
        assert r["errors"] == 1, (path, r)    # the bad row rejected

    @pytest.mark.parametrize("path", PATHS)
    def test_rejected_doc_reconciles(self, path):
        r = self.by_path[path]["mixed"]
        assert r["total_input"] == 2, (path, r)
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], (path, r)

    @pytest.mark.parametrize("path", PATHS)
    def test_error_sample_is_populated_and_diagnostic(self, path):
        samples = self.by_path[path]["mixed"]["error_samples"]
        assert len(samples) == 1, (path, samples)
        s = samples[0]
        assert s["_id"] == "bad1", (path, s)
        assert s["op_type"] in ("index", "create"), (path, s)
        assert s["status"] >= 400, (path, s)
        assert s["reason"], (path, s)   # non-empty ES reason, so the failure is diagnosable


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
