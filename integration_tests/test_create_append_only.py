# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: op_type="create" append-only dedup (live mapInPandas + ES)
# MAGIC Proves `op_type="create"` end to end on an `id_field` feed, the Spark-built `create` action header
# MAGIC included (only the tier can prove `build_ndjson`). Three live, deterministic cases against one index:
# MAGIC
# MAGIC 1. **First write** of N docs: every `create` succeeds, ES holds exactly N and `written == N`.
# MAGIC 2. **Resend of the same `_id`s with a CHANGED field value**: every doc returns `409` (already
# MAGIC    exists) and is treated as an append-only **dedup** -- counted `ignored`, surfaced as
# MAGIC    `docs_deduped` under `bulk_stats`, and specifically NOT an `error`. ES still holds exactly N
# MAGIC    (no duplication) and the stored value is **unchanged** (the create did not overwrite). This is
# MAGIC    what distinguishes `create` from `index`: `index` would overwrite with the changed value and
# MAGIC    report `written == N`.
# MAGIC 3. **Mixed resend** (one NEW `_id` + the existing ones) with `bypass_fast_path=True`, forced into
# MAGIC    ONE chunk (`repartition(1)`): the new doc is written **once** and the existing ones dedup, with
# MAGIC    EXACT counts (`written==1`, `docs_deduped==N`). This is the case that miscounts under the default
# MAGIC    fast path (the probe creates the new doc, then the whole-chunk re-ship self-409s it, so it lands
# MAGIC    as a dedup); `bypass_fast_path` sends the chunk once on the full classify path, so the new doc is
# MAGIC    counted `written`. ES ends with N+1 and no duplicates. (The default-fast-path miscount itself is
# MAGIC    partition-dependent, so it is pinned deterministically in the unit tier, not here.)
# MAGIC
# MAGIC Live ES + the `es_poc` scope required.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-create-append-only"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# id is the deterministic _id (id_field), v is a value we mutate on resend to prove create does NOT
# overwrite an existing doc (unlike index).
_MAPPING = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {"id": {"type": "integer"},
                                        "n": {"type": "integer"},
                                        "v": {"type": "keyword"}}}}
_N = 5


def _recreate(index):
    requests.delete(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30)
    requests.put(f"{ES_HOSTS}/{index}", auth=ES_AUTH, verify=False, timeout=30,
                 headers={"Content-Type": "application/json"}, data=json.dumps(_MAPPING))


def _count(index):
    requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
    return requests.get(f"{ES_HOSTS}/{index}/_count", auth=ES_AUTH, verify=False, timeout=30).json()["count"]


def _source(index, doc_id):
    requests.post(f"{ES_HOSTS}/{index}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
    return requests.get(f"{ES_HOSTS}/{index}/_doc/{doc_id}", auth=ES_AUTH, verify=False,
                        timeout=30).json()["_source"]


def _deduped(res):
    """Sum docs_deduped across the per-partition bulk_stats entries (bulk_stats must be on)."""
    return sum(int(p["docs_deduped"]) for p in res["bulk_stats"])


class TestCreateAppendOnly(NotebookTestFixture):
    """op_type='create': a resend of an existing _id is a 409 dedup (ignored + docs_deduped, not an
    error), never a duplicate and never an overwrite; a genuinely new _id still writes."""

    def run_setup(self):
        # op_type='create' requires id_field. bulk_stats on so docs_deduped is observable.
        cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False, index=INDEX,
                       id_field="id", op_type="create", http_compress=True, bulk_stats=True)
        _recreate(INDEX)

        # 1. First write: N fresh docs, v="orig". Every create succeeds.
        first = spark.range(1, _N + 1).selectExpr("CAST(id AS INT) AS id", "CAST(id AS INT) AS n",
                                                  "'orig' AS v")
        self.res_first = bulk_write(first, cfg)
        self.count_first = _count(INDEX)

        # 2. Resend the SAME ids with a CHANGED value. Every doc already exists => 409 => append-only
        #    dedup. Nothing new is written and (proven below) the stored v stays "orig".
        resend = spark.range(1, _N + 1).selectExpr("CAST(id AS INT) AS id", "CAST(id AS INT) AS n",
                                                   "'CHANGED' AS v")
        self.res_resend = bulk_write(resend, cfg)
        self.count_resend = _count(INDEX)
        self.v_after_resend = _source(INDEX, 1)["v"]

        # 3. Mixed resend: one NEW id (N+1) plus the existing ids, forced into ONE chunk (repartition(1))
        #    so the new doc SHARES a chunk with existing docs -- the exact case that miscounts under the
        #    default fast path. bypass_fast_path=True classifies the chunk on one full-path send, so the
        #    new doc counts `written` and the existing ones `docs_deduped`, exactly.
        cfg_exact = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False, index=INDEX,
                             id_field="id", op_type="create", http_compress=True, bulk_stats=True,
                             bypass_fast_path=True)
        mixed = spark.range(1, _N + 2).selectExpr("CAST(id AS INT) AS id", "CAST(id AS INT) AS n",
                                                  "'orig' AS v").repartition(1)
        self.res_mixed = bulk_write(mixed, cfg_exact)
        self.count_mixed = _count(INDEX)

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- 1. first write: every create lands exactly once ---
    def test_first_write_counts(self):
        r = self.res_first
        assert r["written"] == _N, r
        assert r["errors"] == 0 and r["ignored"] == 0, r
        assert r["total_input"] == _N, r
        assert r["written"] + r["deleted"] + r["errors"] + r["ignored"] == r["total_input"], r

    def test_first_write_landed_n_docs(self):
        assert self.count_first == _N, self.count_first

    # --- 2. resend of existing ids: 409 dedup, not error, not duplicate, not overwrite ---
    def test_resend_is_deduped_not_errored(self):
        r = self.res_resend
        assert r["ignored"] == _N, r          # every doc a create-409 dedup
        assert r["written"] == 0, r           # nothing newly written
        assert r["errors"] == 0, r            # a 409 under create is NOT an error
        assert r["total_input"] == _N, r
        assert r["written"] + r["deleted"] + r["errors"] + r["ignored"] == r["total_input"], r

    def test_resend_surfaces_docs_deduped(self):
        assert _deduped(self.res_resend) == _N, self.res_resend["bulk_stats"]

    def test_resend_did_not_duplicate(self):
        # create rejected the duplicate _ids, so ES still holds exactly N (no at-least-once duplication).
        assert self.count_resend == _N, self.count_resend

    def test_resend_did_not_overwrite(self):
        # The discriminating proof vs op_type="index": the resend carried v="CHANGED", but the create
        # 409'd instead of overwriting, so the stored value is still the original "orig".
        assert self.v_after_resend == "orig", self.v_after_resend

    # --- 3. mixed resend (bypass_fast_path, one chunk): the new id writes, the existing ones dedup ---
    def test_mixed_resend_writes_only_the_new_doc(self):
        r = self.res_mixed
        assert r["written"] == 1, r           # only id N+1 is new
        assert r["ignored"] == _N, r          # the N existing ids dedup
        assert r["errors"] == 0, r
        assert r["total_input"] == _N + 1, r
        assert r["written"] + r["deleted"] + r["errors"] + r["ignored"] == r["total_input"], r

    def test_mixed_resend_landed_n_plus_one(self):
        assert self.count_mixed == _N + 1, self.count_mixed
        assert _deduped(self.res_mixed) == _N, self.res_mixed["bulk_stats"]


# COMMAND ----------
# Auto-discovers the fixture class in this notebook's scope.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
