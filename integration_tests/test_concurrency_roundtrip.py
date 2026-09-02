# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: per-partition write concurrency (live mapInPandas + threaded bulk + ES)
# MAGIC Proves `EsWriteConfig.write_concurrency > 1` is CORRECT under real serverless Spark: the
# MAGIC threaded fan-out inside each partition (several concurrent `streaming_bulk` streams merged
# MAGIC through a bounded queue) must not lose, duplicate, or mis-count a single document, and must
# MAGIC keep per-document error accounting and deterministic-`_id` idempotency intact. The unit tier
# MAGIC (`tests/test_bulk_concurrency.py`) proves the merge/retry/fail-closed logic off-cluster; only
# MAGIC this tier proves it over the real `mapInPandas` write to a live ES. Live ES + `es_poc` scope.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-concurrency"     # throwaway; recreated + dropped by the fixture
N = 5000                                         # enough rows that the fan-out spans many chunks
CONCURRENCY = 4
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))


class TestWriteConcurrencyRoundtrip(NotebookTestFixture):
    """write_concurrency > 1 writes every doc exactly once with correct counts, is idempotent on
    re-write, and still counts a rejected doc, all over the live threaded mapInPandas path."""

    def run_setup(self):
        # chunk_size deliberately small so each of the CONCURRENCY threads sends several bulk
        # requests (fan-out spans multiple chunks per stream), not one chunk each.
        self.cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id", http_compress=True,
                            write_concurrency=CONCURRENCY, chunk_size=100)

        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        # N unique rows across a few partitions, so mapInPandas runs several partitions AND each
        # partition fans across CONCURRENCY threads. Unique doc_id => ES _count == N iff nothing was
        # lost or duplicated by the concurrent merge.
        df = (spark.range(N)
                   .selectExpr("concat('d', id) AS doc_id", "CAST(id AS INT) AS n")
                   .repartition(4))
        self.res = bulk_write(df, self.cfg)

        # Re-write the SAME rows, still concurrent: deterministic _id => upsert, not duplicate.
        self.res_idem = bulk_write(df, self.cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        self.es_count = requests.get(
            f"{ES_HOSTS}/{INDEX}/_count", auth=ES_AUTH, verify=False, timeout=30).json()["count"]

        # A batch with one doc ES rejects, written concurrently: proves per-document error accounting
        # survives the thread merge (the good row indexed, the bad one counted + sampled). n is a
        # STRING column so ES coerces "10" but rejects "not-an-int" under the integer mapping.
        mixed = spark.sql("""
            SELECT 'ok1' AS doc_id, CAST('10' AS STRING) AS n UNION ALL
            SELECT 'bad1', CAST('not-an-int' AS STRING)
        """)
        self.res_mixed = bulk_write(mixed, self.cfg)

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- every doc written exactly once, counts correct ---
    def test_all_docs_written(self):
        assert self.res["written"] == N, self.res
        assert self.res["errors"] == 0, self.res
        assert self.res["deleted"] == 0, self.res

    def test_counts_reconcile_no_loss(self):
        r = self.res
        assert r["total_input"] == N, r
        assert r["written"] + r["deleted"] + r["errors"] == r["total_input"], r
        assert r["unaccounted"] == 0, r      # nothing lost below the per-doc level by the merge
        assert r["overcounted"] == 0, r      # no doc classified twice by the merge

    def test_es_holds_exactly_n_docs(self):
        # Ground truth: the live index has exactly N docs, so the concurrent streams neither dropped
        # nor duplicated any document.
        assert self.es_count == N, self.es_count

    # --- idempotency holds under concurrency ---
    def test_idempotent_rewrite_under_concurrency(self):
        assert self.res_idem["written"] == N, self.res_idem
        assert self.es_count == N, self.es_count   # second concurrent write upserted, no duplicates

    # --- error accounting survives the thread merge ---
    def test_rejected_doc_still_counted_under_concurrency(self):
        assert self.res_mixed["written"] == 1, self.res_mixed
        assert self.res_mixed["errors"] == 1, self.res_mixed
        assert (self.res_mixed["written"] + self.res_mixed["deleted"] + self.res_mixed["errors"]
                == self.res_mixed["total_input"]), self.res_mixed
        samples = self.res_mixed["error_samples"]
        assert len(samples) == 1 and samples[0]["_id"] == "bad1", samples


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
