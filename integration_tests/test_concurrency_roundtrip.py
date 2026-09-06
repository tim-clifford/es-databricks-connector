# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: per-partition write concurrency (live mapInPandas + threaded bulk + ES)
# MAGIC Proves `EsWriteConfig.write_concurrency > 1` is CORRECT under real serverless Spark, on BOTH
# MAGIC write paths: the threaded fan-out inside each partition must not lose, duplicate, or mis-count
# MAGIC a single document, and must keep per-document error accounting and deterministic-`_id`
# MAGIC idempotency intact.
# MAGIC   - default path: several concurrent `streaming_bulk` streams merged through a bounded queue
# MAGIC     (`bulk._iter_bulk_results`);
# MAGIC   - serialize_in_spark path: several workers each shipping their strided slice of pre-built
# MAGIC     NDJSON via `es.bulk(operations=...)` (`bulk._ship_ndjson_lines`).
# MAGIC The unit tier proves the merge/retry/fail-closed logic off-cluster
# MAGIC (`tests/test_bulk_concurrency.py`, `tests/test_spark_serialize.py`); only this tier proves it
# MAGIC over the real `mapInPandas` write to a live ES. Live ES + `es_poc` scope.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
import pytest
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
N = 5000                                         # enough rows that the fan-out spans many chunks
CONCURRENCY = 4
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# Both write paths run the identical concurrency scenario into their own throwaway index.
PATHS = ["default", "spark"]
INDEX = {"default": "connector-integration-concurrency",
         "spark": "connector-integration-concurrency-sis"}


class TestWriteConcurrencyRoundtrip(NotebookTestFixture):
    """write_concurrency > 1 writes every doc exactly once with correct counts, is idempotent on
    re-write, and still counts a rejected doc -- over the live threaded mapInPandas path, on both the
    default (streaming_bulk) and serialize_in_spark (NDJSON) write paths."""

    def _cfg(self, path):
        # chunk_size deliberately small so each of the CONCURRENCY workers sends several bulk requests
        # (fan-out spans multiple chunks per worker), not one chunk each.
        return EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                        index=INDEX[path], id_field="doc_id", http_compress=True,
                        write_concurrency=CONCURRENCY, chunk_size=100,
                        serialize_in_spark=(path == "spark"))

    def _count(self, index):
        return requests.get(f"{ES_HOSTS}/{index}/_count", auth=ES_AUTH,
                            verify=False, timeout=30).json()["count"]

    def _run_path(self, path):
        index = INDEX[path]
        cfg = self._cfg(path)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
        requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        r = {}
        # N unique rows across a few partitions, so mapInPandas runs several partitions AND each
        # partition fans across CONCURRENCY workers. Unique doc_id => ES _count == N iff nothing was
        # lost or duplicated by the concurrent merge.
        df = (spark.range(N)
                   .selectExpr("concat('d', id) AS doc_id", "CAST(id AS INT) AS n")
                   .repartition(4))
        r["res"] = bulk_write(df, cfg)

        # Re-write the SAME rows, still concurrent: deterministic _id => upsert, not duplicate.
        r["res_idem"] = bulk_write(df, cfg)
        requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        r["es_count"] = self._count(index)

        # A batch with one doc ES rejects, written concurrently: proves per-document error accounting
        # survives the thread merge (the good row indexed, the bad one counted + sampled). n is a
        # STRING column so ES coerces "10" but rejects "not-an-int" under the integer mapping.
        mixed = spark.sql("""
            SELECT 'ok1' AS doc_id, CAST('10' AS STRING) AS n UNION ALL
            SELECT 'bad1', CAST('not-an-int' AS STRING)
        """)
        r["res_mixed"] = bulk_write(mixed, cfg)
        return r

    def run_setup(self):
        self.by_path = {p: self._run_path(p) for p in PATHS}

    def run_cleanup(self):
        for index in INDEX.values():
            requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)

    # --- every doc written exactly once, counts correct (per path) ---
    @pytest.mark.parametrize("path", PATHS)
    def test_all_docs_written(self, path):
        r = self.by_path[path]["res"]
        assert r["written"] == N, (path, r)
        assert r["errors"] == 0, (path, r)
        assert r["deleted"] == 0, (path, r)

    @pytest.mark.parametrize("path", PATHS)
    def test_counts_reconcile_no_loss(self, path):
        r = self.by_path[path]["res"]
        assert r["total_input"] == N, (path, r)
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], (path, r)
        assert r["unaccounted"] == 0, (path, r)      # nothing lost below the per-doc level by the merge
        assert r["overcounted"] == 0, (path, r)      # no doc classified twice by the merge

    @pytest.mark.parametrize("path", PATHS)
    def test_es_holds_exactly_n_docs(self, path):
        # Ground truth: the live index has exactly N docs, so the concurrent workers neither dropped
        # nor duplicated any document.
        assert self.by_path[path]["es_count"] == N, (path, self.by_path[path]["es_count"])

    # --- idempotency holds under concurrency ---
    @pytest.mark.parametrize("path", PATHS)
    def test_idempotent_rewrite_under_concurrency(self, path):
        r = self.by_path[path]
        assert r["res_idem"]["written"] == N, (path, r["res_idem"])
        assert r["es_count"] == N, (path, r["es_count"])   # second concurrent write upserted, no dupes

    # --- error accounting survives the thread merge ---
    @pytest.mark.parametrize("path", PATHS)
    def test_rejected_doc_still_counted_under_concurrency(self, path):
        r = self.by_path[path]["res_mixed"]
        assert r["written"] == 1, (path, r)
        assert r["errors"] == 1, (path, r)
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], (path, r)
        samples = r["error_samples"]
        assert len(samples) == 1 and samples[0]["_id"] == "bad1", (path, samples)


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
