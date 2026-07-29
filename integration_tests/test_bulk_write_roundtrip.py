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
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-roundtrip"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))


class TestBulkWriteResultContract(NotebookTestFixture):
    """bulk_write's return dict is correct end-to-end: clean write, idempotent re-write, and a
    write with a doc ES rejects so errors + error_samples are populated (not just the happy path)."""

    def run_setup(self):
        self.cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id", http_compress=True)

        # A strict mapping so we can force a REJECTED doc: n is an integer field, so a row whose n is
        # a non-numeric string fails to index. This exercises the error path deterministically.
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        # 3 good rows.
        good = spark.sql("""
            SELECT 'd1' AS doc_id, 1 AS n UNION ALL
            SELECT 'd2', 2 UNION ALL
            SELECT 'd3', 3
        """)
        self.res_clean = bulk_write(good, self.cfg)

        # Re-write the SAME rows: deterministic _id => upsert, not duplicate.
        self.res_idem = bulk_write(good, self.cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        self.es_count_after_idem = requests.get(
            f"{ES_HOSTS}/{INDEX}/_count", auth=ES_AUTH, verify=False, timeout=30).json()["count"]

        # A batch with one doc ES will reject. n is a STRING column (both rows cast to string, so the
        # UNION's common type is string — otherwise Spark coerces to BIGINT and 'not-an-int' fails to
        # cast inside Spark before the connector ever runs). The connector sends both as JSON strings;
        # under the integer mapping ES coerces "10" to the int 10 (indexes) but rejects "not-an-int"
        # (mapper_parsing_exception) — the deterministic ES-side error this test wants.
        mixed = spark.sql("""
            SELECT 'ok1' AS doc_id, CAST('10' AS STRING) AS n UNION ALL
            SELECT 'bad1', CAST('not-an-int' AS STRING)
        """)
        self.res_mixed = bulk_write(mixed, self.cfg)

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- clean write ---
    def test_clean_write_counts(self):
        assert self.res_clean["written"] == 3, self.res_clean
        assert self.res_clean["errors"] == 0, self.res_clean
        assert self.res_clean["deleted"] == 0, self.res_clean

    def test_clean_write_reconciles(self):
        r = self.res_clean
        assert r["total_input"] == 3
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"]

    def test_clean_write_no_error_samples(self):
        assert self.res_clean["error_samples"] == []

    # --- idempotency ---
    def test_idempotent_rewrite_no_duplicates(self):
        # Second write of the same _ids upserts; ES still holds exactly 3 docs.
        assert self.res_idem["written"] == 3, self.res_idem
        assert self.es_count_after_idem == 3, self.es_count_after_idem

    # --- error path: a rejected doc must be counted AND sampled, good docs still written ---
    def test_rejected_doc_counted(self):
        assert self.res_mixed["written"] == 1, self.res_mixed   # the good row indexed
        assert self.res_mixed["errors"] == 1, self.res_mixed    # the bad row rejected

    def test_rejected_doc_reconciles(self):
        r = self.res_mixed
        assert r["total_input"] == 2
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"]

    def test_error_sample_is_populated_and_diagnostic(self):
        samples = self.res_mixed["error_samples"]
        assert len(samples) == 1, samples
        s = samples[0]
        assert s["_id"] == "bad1"
        assert s["op_type"] in ("index", "create")
        assert s["status"] >= 400
        assert s["reason"]   # non-empty ES reason, so the failure is diagnosable


# COMMAND ----------
# Explicit fixture class (no-arg auto-discovery finds nothing through the wrapper).
dbutils.notebook.exit(json.dumps(run_notebook_tests(TestBulkWriteResultContract)))
