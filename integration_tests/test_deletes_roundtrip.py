# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: delete propagation (has_deletes) live through mapInPandas + ES
# MAGIC Owns the **delete-routing** contract end-to-end, on BOTH write paths: with `has_deletes=True`
# MAGIC and a `delete_flag_column`, rows whose flag is truthy are sent to ES as delete-by-`_id` while
# MAGIC every other row indexes as usual. Proves (against real serverless Spark + ES, not a stub) that:
# MAGIC   - flagged `_id`s are removed from ES and unflagged rows are indexed;
# MAGIC   - `result["deleted"]` counts successful deletes exactly, and the flag column is not indexed;
# MAGIC   - a delete of an `_id` that isn't in ES is a **404 no-op** (counted as neither delete nor
# MAGIC     error), the connector's most subtle documented rule (`classify_bulk_result`'s scoped
# MAGIC     404 suppression), which unit tests cover in isolation but has never run live.
# MAGIC
# MAGIC The whole scenario runs under `serialize_in_spark=False` (per-row Python `build_action`) AND
# MAGIC `serialize_in_spark=True` (Catalyst `build_ndjson`, which routes a `flag === true` row to a
# MAGIC delete-by-id line with no source). `test_both_paths_agree_on_es_state` is the direct lock that
# MAGIC the Spark path deletes identically to the per-row path. The delete flag is a real boolean
# MAGIC column, which the Spark path requires (bulk._preflight). Live ES + the `es_poc` scope required.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
import pytest
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# Both write paths run the identical delete scenario into their own throwaway index, so a single
# fixture proves the per-row path and the Catalyst path agree on delete routing. `serialize_in_spark`
# requires the flag column to be a real boolean (bulk._preflight); the SQL below uses true/false
# literals, which are boolean, so the same rows drive both paths.
PATHS = ["default", "spark"]
INDEX = {"default": "connector-integration-deletes",
         "spark": "connector-integration-deletes-sis"}


class TestDeletesRoundtrip(NotebookTestFixture):
    """has_deletes routing end-to-end on BOTH write paths: index the unflagged, delete the flagged,
    and treat a delete of an absent _id as a 404 no-op (not an error), all against a live ES index."""

    def _cfg(self, path):
        return EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                        index=INDEX[path], id_field="doc_id",
                        has_deletes=True, delete_flag_column="_is_delete",
                        serialize_in_spark=(path == "spark"), http_compress=True)

    def _count(self, index):
        return requests.get(f"{ES_HOSTS}/{index}/_count", auth=ES_AUTH,
                            verify=False, timeout=30).json()["count"]

    def _ids(self, index):
        hits = requests.get(f"{ES_HOSTS}/{index}/_search", auth=ES_AUTH, verify=False, timeout=30,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps({"size": 100, "query": {"match_all": {}}})).json()
        return {h["_id"] for h in hits.get("hits", {}).get("hits", [])}

    def _run_path(self, path):
        """Run the full seed -> mixed -> redelete scenario for one write path; return its results."""
        index = INDEX[path]
        cfg = self._cfg(path)
        requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        r = {}
        # --- phase 1: seed 4 live docs (all unflagged => all index) ---
        seed = spark.sql("""
            SELECT 'k1' AS doc_id, 1 AS n, false AS _is_delete UNION ALL
            SELECT 'k2', 2, false UNION ALL
            SELECT 'k3', 3, false UNION ALL
            SELECT 'k4', 4, false
        """)
        r["res_seed"] = bulk_write(seed, cfg)
        requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        r["count_after_seed"] = self._count(index)

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
        r["res_mixed"] = bulk_write(mixed, cfg)
        requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        r["count_after_mixed"] = self._count(index)
        r["ids_after_mixed"] = self._ids(index)

        # --- phase 3: idempotent re-delete, deleting k1 again is another 404 no-op ---
        redelete = spark.sql("""
            SELECT 'k1' AS doc_id, CAST(NULL AS INT) AS n, true AS _is_delete
        """)
        r["res_redelete"] = bulk_write(redelete, cfg)
        return r

    def run_setup(self):
        # self.by_path[path] = {"res_seed", "count_after_seed", "res_mixed", "count_after_mixed",
        #                       "ids_after_mixed", "res_redelete"}
        self.by_path = {p: self._run_path(p) for p in PATHS}

    def run_cleanup(self):
        for index in INDEX.values():
            requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)

    # --- seed: all rows index, nothing deleted (per path) ---
    @pytest.mark.parametrize("path", PATHS)
    def test_seed_indexes_all_unflagged(self, path):
        r = self.by_path[path]
        assert r["res_seed"]["written"] == 4, (path, r["res_seed"])
        assert r["res_seed"]["deleted"] == 0, (path, r["res_seed"])
        assert r["res_seed"]["errors"] == 0, (path, r["res_seed"])
        assert r["count_after_seed"] == 4, (path, r["count_after_seed"])

    # --- mixed batch: flagged rows delete, unflagged row indexes, absent-delete is a no-op ---
    @pytest.mark.parametrize("path", PATHS)
    def test_mixed_batch_counts(self, path):
        # k1,k2 deleted (present) => deleted=2; k5 indexed => written=1; k_absent delete is a 404
        # no-op => NOT counted as deleted and NOT an error.
        r = self.by_path[path]
        assert r["res_mixed"]["deleted"] == 2, (path, r["res_mixed"])
        assert r["res_mixed"]["written"] == 1, (path, r["res_mixed"])
        assert r["res_mixed"]["errors"] == 0, (path, r["res_mixed"])

    @pytest.mark.parametrize("path", PATHS)
    def test_mixed_batch_es_state(self, path):
        # Started with k1..k4 (4). Deleted k1,k2; added k5. Expect k3,k4,k5 => 3 docs.
        r = self.by_path[path]
        assert r["count_after_mixed"] == 3, (path, r["count_after_mixed"])
        assert r["ids_after_mixed"] == {"k3", "k4", "k5"}, (path, r["ids_after_mixed"])

    @pytest.mark.parametrize("path", PATHS)
    def test_absent_delete_is_404_noop_not_error(self, path):
        # The scoped-suppression rule, proven live: k_absent was flagged for delete but never
        # existed. total_input=4, but only 3 ops "count" (2 deletes + 1 index); the 4th (absent
        # delete) is an ignored no-op. So written+deleted+errors = 3 < total_input = 4, with
        # errors=0: the reconciliation gap here is the EXPECTED delete-404, not lost data.
        r = self.by_path[path]["res_mixed"]
        assert r["total_input"] == 4, (path, r)
        assert r["errors"] == 0, (path, r)
        assert r["written"] + r["deleted"] + r["errors"] == 3, (path, r)   # the 404 no-op is not among these

    @pytest.mark.parametrize("path", PATHS)
    def test_flag_column_not_indexed(self, path):
        # The delete-flag column must never land in _source (it's control data, not document data).
        doc = requests.get(f"{ES_HOSTS}/{INDEX[path]}/_doc/k5", auth=ES_AUTH,
                           verify=False, timeout=30).json()["_source"]
        assert "_is_delete" not in doc, (path, doc)
        assert doc == {"doc_id": "k5", "n": 5}, (path, doc)

    # --- idempotent re-delete: deleting an already-gone id is again a clean no-op ---
    @pytest.mark.parametrize("path", PATHS)
    def test_redelete_is_clean_noop(self, path):
        r = self.by_path[path]["res_redelete"]
        assert r["deleted"] == 0, (path, r)
        assert r["errors"] == 0, (path, r)
        assert r["written"] == 0, (path, r)

    # --- the two write paths agree on final ES state: the direct lock that serialize_in_spark
    #     (Catalyst build_ndjson) routes deletes identically to the per-row build_action path ---
    def test_both_paths_agree_on_es_state(self):
        d, s = self.by_path["default"], self.by_path["spark"]
        assert d["ids_after_mixed"] == s["ids_after_mixed"], (d["ids_after_mixed"], s["ids_after_mixed"])
        assert d["count_after_mixed"] == s["count_after_mixed"]
        for key in ("written", "deleted", "errors", "ignored"):
            assert d["res_mixed"][key] == s["res_mixed"][key], \
                (key, d["res_mixed"][key], s["res_mixed"][key])


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
