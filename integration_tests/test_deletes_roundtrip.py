# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: delete propagation (has_deletes) live through mapInPandas + ES
# MAGIC Owns the **delete-routing** contract end-to-end: with `has_deletes=True` and a
# MAGIC `delete_flag_column`, rows whose flag is truthy are sent to ES as delete-by-`_id` while every
# MAGIC other row indexes as usual. Proves (against real serverless Spark + ES, not a stub) that:
# MAGIC   - flagged `_id`s are removed from ES and unflagged rows are indexed;
# MAGIC   - `result["deleted"]` counts successful deletes exactly, and the flag column is not indexed;
# MAGIC   - a delete of an `_id` that isn't in ES is a **404 no-op** (counted as neither delete nor
# MAGIC     error), the connector's most subtle documented rule (`classify_bulk_result`'s scoped
# MAGIC     404 suppression), which unit tests cover in isolation but has never run live.
# MAGIC
# MAGIC The pure-Python suite covers `build_action` delete routing and `classify_bulk_result` with
# MAGIC hand-built inputs; this fixture is the live counterpart. Live ES + the `es_poc` scope required.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-deletes"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))


class TestDeletesRoundtrip(NotebookTestFixture):
    """has_deletes routing end-to-end: index the unflagged, delete the flagged, and treat a delete
    of an absent _id as a 404 no-op (not an error), all against a live ES index."""

    def run_setup(self):
        self.cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id",
                            has_deletes=True, delete_flag_column="_is_delete",
                            http_compress=True)

        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        # --- phase 1: seed 4 live docs (all unflagged => all index) ---
        seed = spark.sql("""
            SELECT 'k1' AS doc_id, 1 AS n, false AS _is_delete UNION ALL
            SELECT 'k2', 2, false UNION ALL
            SELECT 'k3', 3, false UNION ALL
            SELECT 'k4', 4, false
        """)
        self.res_seed = bulk_write(seed, self.cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        self.count_after_seed = self._count()

        # --- phase 2: a mixed batch of deletes + an index, plus a delete of an ABSENT id ---
        # k1,k2 flagged for delete (present => real deletes). k5 is a NEW live row (index).
        # k_absent is flagged for delete but was never indexed => ES returns 404 => no-op (the
        # connector suppresses delete-404 to errors=0, and it must NOT count as a delete either).
        mixed = spark.sql("""
            SELECT 'k1' AS doc_id, CAST(NULL AS INT) AS n, true  AS _is_delete UNION ALL
            SELECT 'k2', CAST(NULL AS INT), true  UNION ALL
            SELECT 'k5', 5,               false UNION ALL
            SELECT 'k_absent', CAST(NULL AS INT), true
        """)
        self.res_mixed = bulk_write(mixed, self.cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        self.count_after_mixed = self._count()
        self.ids_after_mixed = self._ids()

        # --- phase 3: idempotent re-delete, deleting k1 again is another 404 no-op ---
        redelete = spark.sql("""
            SELECT 'k1' AS doc_id, CAST(NULL AS INT) AS n, true AS _is_delete
        """)
        self.res_redelete = bulk_write(redelete, self.cfg)

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    def _count(self):
        return requests.get(f"{ES_HOSTS}/{INDEX}/_count", auth=ES_AUTH,
                            verify=False, timeout=30).json()["count"]

    def _ids(self):
        hits = requests.get(f"{ES_HOSTS}/{INDEX}/_search", auth=ES_AUTH, verify=False, timeout=30,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps({"size": 100, "query": {"match_all": {}}})).json()
        return {h["_id"] for h in hits.get("hits", {}).get("hits", [])}

    # --- seed: all rows index, nothing deleted ---
    def test_seed_indexes_all_unflagged(self):
        assert self.res_seed["written"] == 4, self.res_seed
        assert self.res_seed["deleted"] == 0, self.res_seed
        assert self.res_seed["errors"] == 0, self.res_seed
        assert self.count_after_seed == 4, self.count_after_seed

    # --- mixed batch: flagged rows delete, unflagged row indexes, absent-delete is a no-op ---
    def test_mixed_batch_counts(self):
        # k1,k2 deleted (present) => deleted=2; k5 indexed => written=1; k_absent delete is a 404
        # no-op => NOT counted as deleted and NOT an error.
        assert self.res_mixed["deleted"] == 2, self.res_mixed
        assert self.res_mixed["written"] == 1, self.res_mixed
        assert self.res_mixed["errors"] == 0, self.res_mixed

    def test_mixed_batch_es_state(self):
        # Started with k1..k4 (4). Deleted k1,k2; added k5. Expect k3,k4,k5 => 3 docs.
        assert self.count_after_mixed == 3, self.count_after_mixed
        assert self.ids_after_mixed == {"k3", "k4", "k5"}, self.ids_after_mixed

    def test_absent_delete_is_404_noop_not_error(self):
        # The scoped-suppression rule, proven live: k_absent was flagged for delete but never
        # existed. total_input=4, but only 3 ops "count" (2 deletes + 1 index); the 4th (absent
        # delete) is an ignored no-op. So written+deleted+errors = 3 < total_input = 4, with
        # errors=0: the reconciliation gap here is the EXPECTED delete-404, not lost data.
        r = self.res_mixed
        assert r["total_input"] == 4, r
        assert r["errors"] == 0, r
        assert r["written"] + r["deleted"] + r["errors"] == 3, r   # the 404 no-op is not among these

    def test_flag_column_not_indexed(self):
        # The delete-flag column must never land in _source (it's control data, not document data).
        doc = requests.get(f"{ES_HOSTS}/{INDEX}/_doc/k5", auth=ES_AUTH,
                           verify=False, timeout=30).json()["_source"]
        assert "_is_delete" not in doc, doc
        assert doc == {"doc_id": "k5", "n": 5}, doc

    # --- idempotent re-delete: deleting an already-gone id is again a clean no-op ---
    def test_redelete_is_clean_noop(self):
        assert self.res_redelete["deleted"] == 0, self.res_redelete
        assert self.res_redelete["errors"] == 0, self.res_redelete
        assert self.res_redelete["written"] == 0, self.res_redelete


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
