# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: the streaming sink (make_foreach_batch) live on Structured Streaming + ES
# MAGIC Owns the **streaming** contract end-to-end: `make_foreach_batch(cfg)` used as a real
# MAGIC `foreachBatch` sink on serverless (`readStream` → `trigger(availableNow=True)` → UC-Volume
# MAGIC checkpoint) actually writes each micro-batch to ES. The unit suite (`tests/test_stream.py`)
# MAGIC covers the glue with a stubbed `bulk_write` and a fake DataFrame; this is the only place the
# MAGIC sink runs against genuine Structured Streaming.
# MAGIC
# MAGIC Proves:
# MAGIC   - a stream over a Delta source lands exactly one ES doc per source row;
# MAGIC   - **restart idempotency** — re-running the SAME stream from the SAME checkpoint after
# MAGIC     appending new rows writes only the new rows (deterministic `_id` upserts, no duplicates),
# MAGIC     and a re-run with no new source rows is a clean no-op.
# MAGIC
# MAGIC The `on_batch` callback's per-batch contract is unit-tested in `tests/test_stream.py`; it is
# MAGIC not re-checked here because on serverless `foreachBatch` runs server-side, so a driver-local
# MAGIC capture from `on_batch` does not propagate back to the notebook (the ES doc count is the
# MAGIC authoritative signal instead).
# MAGIC
# MAGIC Needs live ES + the `es_poc` scope, a UC catalog/schema to hold a throwaway Delta table, and a
# MAGIC UC Volume for the checkpoint (dbfs:/tmp fails with INSUFFICIENT_PERMISSIONS on serverless).
# MAGIC All three are created/dropped by the fixture.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, make_foreach_batch

SCOPE = "es_poc"
INDEX = "connector-integration-streaming"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# A throwaway Delta source table + a checkpoint dir on a UC Volume. These mirror where the demos
# live so the fixture runs in the same workspace, but nothing here is shared with a demo (distinct
# names) — both are dropped in cleanup.
CATALOG = "tim_clifford_classic_dsl_lite_catalog"
SCHEMA = "es_poc"
SRC_TABLE = f"{CATALOG}.{SCHEMA}.connector_it_streaming_src"
CHECKPOINT = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts/checkpoints/connector_it_streaming"


class TestStreamingSink(NotebookTestFixture):
    """make_foreach_batch as a live foreachBatch sink: one doc per source row, restart-idempotent."""

    def run_setup(self):
        self.cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id", http_compress=True)

        # Fresh ES index, fresh checkpoint, fresh source table — a clean slate so counts are
        # unambiguous. (A stale checkpoint would make availableNow think it had already consumed
        # the source and write nothing.)
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))
        try:
            dbutils.fs.rm(CHECKPOINT, recurse=True)
        except Exception:
            pass
        spark.sql(f"DROP TABLE IF EXISTS {SRC_TABLE}")
        spark.sql(f"CREATE TABLE {SRC_TABLE} (doc_id STRING, n INT) USING DELTA")

        # --- run 1: seed 3 rows, stream them to ES ---
        spark.sql(f"INSERT INTO {SRC_TABLE} VALUES ('s1', 1), ('s2', 2), ('s3', 3)")
        self._run_stream()
        self.count_after_run1 = self._count()

        # --- run 2: append 2 NEW rows, re-run the SAME stream (same checkpoint) ---
        # availableNow should drain only the new commits; deterministic _id means even if a row
        # were re-read it would upsert, not duplicate. Expect ES to grow by exactly 2.
        spark.sql(f"INSERT INTO {SRC_TABLE} VALUES ('s4', 4), ('s5', 5)")
        self._run_stream()
        self.count_after_run2 = self._count()
        self.ids_after_run2 = self._ids()

        # --- run 3: no new rows, re-run again — must be a clean no-op (no new docs, no dupes) ---
        self._run_stream()
        self.count_after_run3 = self._count()

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        try:
            dbutils.fs.rm(CHECKPOINT, recurse=True)
        except Exception:
            pass
        spark.sql(f"DROP TABLE IF EXISTS {SRC_TABLE}")

    def _run_stream(self):
        """Run one availableNow pass of the streaming sink to completion.

        The authoritative signal is the ES doc count, NOT anything captured from on_batch: on
        serverless / Spark Connect, foreachBatch runs server-side, so a driver-local list appended
        inside on_batch does not propagate back to this notebook process (the streaming demo
        documents the same caveat). on_batch's per-batch contract is unit-tested in
        tests/test_stream.py; here we prove the sink actually writes to ES by counting docs.
        """
        q = (spark.readStream.table(SRC_TABLE)
             .writeStream
             .foreachBatch(make_foreach_batch(self.cfg))
             .option("checkpointLocation", CHECKPOINT)
             .trigger(availableNow=True)
             .start())
        q.awaitTermination()   # availableNow terminates on its own once the source is drained
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)

    def _count(self):
        return requests.get(f"{ES_HOSTS}/{INDEX}/_count", auth=ES_AUTH,
                            verify=False, timeout=30).json()["count"]

    def _ids(self):
        hits = requests.get(f"{ES_HOSTS}/{INDEX}/_search", auth=ES_AUTH, verify=False, timeout=30,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps({"size": 100, "query": {"match_all": {}}})).json()
        return {h["_id"] for h in hits.get("hits", {}).get("hits", [])}

    # --- run 1: the stream writes one doc per source row ---
    def test_run1_writes_all_seed_rows(self):
        # The sink fired against real Structured Streaming and landed all 3 seed rows.
        assert self.count_after_run1 == 3, self.count_after_run1

    # --- run 2: incremental pickup, no duplicates ---
    def test_run2_appends_only_new_rows(self):
        # Grew from 3 to 5: exactly the 2 appended rows, no re-write of the first 3. The checkpoint
        # tracked the offset (incremental drain) and the deterministic _id kept it duplicate-free.
        assert self.count_after_run2 == 5, self.count_after_run2
        assert self.ids_after_run2 == {"s1", "s2", "s3", "s4", "s5"}, self.ids_after_run2

    # --- run 3: empty re-run is a clean no-op ---
    def test_run3_no_new_rows_is_noop(self):
        # No source appends since run 2 => availableNow has nothing to drain => ES unchanged, no dupes.
        assert self.count_after_run3 == 5, self.count_after_run3


# COMMAND ----------
# Explicit fixture class (no-arg auto-discovery finds nothing through the wrapper).
dbutils.notebook.exit(json.dumps(run_notebook_tests(TestStreamingSink)))
