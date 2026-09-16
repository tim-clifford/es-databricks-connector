# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: the filter_path="errors" fast path on AUTO-ID writes (live mapInPandas + ES)
# MAGIC Proves the fast path now applies to auto-id writes (no `id_field`), which previously always took
# MAGIC the full per-item-decode path. Two live, deterministic cases:
# MAGIC
# MAGIC 1. **Clean auto-id chunk** takes the fast path and lands every doc EXACTLY once: the probe returns
# MAGIC    `{"errors": false}`, so there is no full re-ship and ES holds exactly N docs (no duplication).
# MAGIC 2. **Auto-id chunk containing a rejected doc** re-ships the whole chunk: the probe indexes the good
# MAGIC    docs (fresh auto-ids) and flags `errors: true`, then the re-ship indexes the good docs AGAIN
# MAGIC    (new auto-ids) for per-item classification. ES therefore holds 2x the good docs while `written`
# MAGIC    counts them once. This is the documented, accepted at-least-once tradeoff for auto-id, and it is
# MAGIC    the behavior that DISTINGUISHES the new gate from the old one (the old full path classified
# MAGIC    per-item without re-shipping, so it would hold the good docs only once). The duplication is
# MAGIC    SCOPED to a chunk that contains both a success and a failure -- good docs in a clean chunk (a
# MAGIC    separate partition) are not duplicated -- so the case forces all rows into one chunk via
# MAGIC    `repartition(1)`.
# MAGIC
# MAGIC Live ES + the `es_poc` scope required.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
CLEAN_INDEX = "connector-integration-autoid-clean"    # throwaway; recreated + dropped by the fixture
ERR_INDEX = "connector-integration-autoid-err"        # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# Strict mapping: n is an integer, so a row whose n is a non-numeric string is REJECTED by ES
# (mapper_parsing_exception) -- the deterministic ES-side error the re-ship case needs.
_MAPPING = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {"n": {"type": "integer"}}}}


def _recreate(index):
    requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
    requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                 headers={"Content-Type": "application/json"}, data=json.dumps(_MAPPING))


def _count(index):
    requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
    return requests.get(f"{ES_HOSTS}/{index}/_count", auth=ES_AUTH, verify=False, timeout=30).json()["count"]


class TestAutoIdFastPath(NotebookTestFixture):
    """The fast path applies to auto-id (no id_field) writes: a clean chunk lands each doc once (no
    re-ship, no dup); a chunk with a rejected doc re-ships the whole chunk, duplicating the good docs."""

    def run_setup(self):
        # id_field OMITTED => ES auto-assigns _id. No deletes => the fast path is engaged.
        clean_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                             index=CLEAN_INDEX, http_compress=True)
        _recreate(CLEAN_INDEX)
        # Three good rows, one chunk (default chunk_size). Clean probe => errors:false => counted
        # written with NO re-ship => ES holds exactly 3.
        clean = spark.sql("SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3")
        self.res_clean = bulk_write(clean, clean_cfg)
        self.clean_count = _count(CLEAN_INDEX)

        err_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                           index=ERR_INDEX, http_compress=True)
        _recreate(ERR_INDEX)
        # Two good rows + one ES-rejected row, forced into ONE partition (repartition(1)) so all three
        # land in a SINGLE bulk chunk. This is what makes the good rows share the chunk with the bad row
        # -- the duplication is scoped to a chunk that contains both a success and a failure; good rows
        # in a clean chunk (a separate partition) are NOT duplicated (that is the clean case above). n is
        # a STRING column (both rows cast to string so the UNION's common type is string; otherwise Spark
        # coerces to BIGINT and the bad value fails to cast inside Spark before the connector runs).
        # Under the integer mapping ES coerces "10"/"20" to ints (indexed) but rejects "not-an-int". The
        # probe indexes the two good rows and flags errors:true; the re-ship indexes the two good rows
        # AGAIN (fresh auto-ids) => ES ends with 4 good docs, while written counts the good rows once
        # (from the re-ship).
        mixed = spark.sql("""
            SELECT CAST('10' AS STRING) AS n UNION ALL
            SELECT CAST('20' AS STRING) UNION ALL
            SELECT CAST('not-an-int' AS STRING)
        """).repartition(1)
        self.res_err = bulk_write(mixed, err_cfg)
        self.err_count = _count(ERR_INDEX)

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{CLEAN_INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        requests.delete(f"{ES_HOSTS}/{ERR_INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- clean auto-id chunk: fast path, no re-ship, each doc landed exactly once ---
    def test_clean_autoid_counts(self):
        r = self.res_clean
        assert r["written"] == 3, r
        assert r["errors"] == 0, r
        assert r["total_input"] == 3, r
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], r

    def test_clean_autoid_no_duplication(self):
        # Clean probe => no re-ship => ES holds exactly the 3 input docs (auto-ids, each written once).
        assert self.clean_count == 3, self.clean_count

    # --- auto-id chunk with a rejected doc: whole-chunk re-ship duplicates the good docs ---
    def test_error_autoid_counts_reconcile(self):
        r = self.res_err
        assert r["written"] == 2, r     # two good rows, counted once (from the re-ship)
        assert r["errors"] == 1, r      # the bad row rejected
        assert r["total_input"] == 3, r
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], r

    def test_error_autoid_good_docs_duplicated_by_reship(self):
        # The discriminating proof of the new gate: the probe wrote the 2 good docs, then the re-ship
        # wrote them AGAIN (fresh auto-ids), so ES holds 4 -- twice the good rows -- even though written
        # is 2. The old full path (auto-id) classified per item without re-shipping and would hold 2.
        assert self.err_count == 4, self.err_count

    def test_error_autoid_sample_populated(self):
        samples = self.res_err["error_samples"]
        assert len(samples) == 1, samples
        s = samples[0]
        assert s["op_type"] in ("index", "create"), s
        assert s["status"] >= 400, s
        assert s["reason"], s   # non-empty ES reason, so the failure is diagnosable


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
