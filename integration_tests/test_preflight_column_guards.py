# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: preflight column guards (the negative case)
# MAGIC Owns the **fail-closed** contract for every config field that names a DataFrame column
# MAGIC (`bulk._COLUMN_NAMING_FIELDS`: `id_field`, `drop_fields`, `delete_flag_column`). Proves against
# MAGIC real serverless Spark + ES that a misspelled name raises BEFORE any document is written, and
# MAGIC that a correct config still writes normally.
# MAGIC
# MAGIC Why this fixture exists, and why it is the negative counterpart to
# MAGIC `test_deletes_roundtrip.py`: every other delete test passes a flag column that EXISTS, so they
# MAGIC all validated the happy path and none could see the failure. With a misspelled
# MAGIC `delete_flag_column`, `row.get(flag)` returned None for every row, so no row was routed to a
# MAGIC delete and every intended deletion was applied as an **upsert** instead, reporting
# MAGIC `deleted=0, errors=0, unaccounted=0` and passing `raise_on_error=True` clean while the
# MAGIC documents stayed in Elasticsearch. This fixture pins the guard AND asserts ES state, so a
# MAGIC regression cannot hide behind counts that look right.
# MAGIC
# MAGIC The pure-Python suite covers `_preflight` with a stub DataFrame; this is the live counterpart
# MAGIC (real `df.columns` after `sanitize_for_arrow`, real index). Live ES + the `es_poc` scope required.

# COMMAND ----------
import json, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-preflight"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))


class TestPreflightColumnGuards(NotebookTestFixture):
    """A config field naming a column that does not exist must fail the write, not proceed quietly."""

    def run_setup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {"properties": {"doc_id": {"type": "keyword"}, "n": {"type": "integer"}}}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        # Seed two live docs with a correct config, so the deletes below have something to remove.
        self.good_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                 index=INDEX, id_field="doc_id",
                                 has_deletes=True, delete_flag_column="_is_delete")
        seed = spark.sql("""
            SELECT 'k1' AS doc_id, 1 AS n, false AS _is_delete UNION ALL
            SELECT 'k2', 2, false
        """)
        self.res_seed = bulk_write(seed, self.good_cfg)
        self._refresh()
        self.count_after_seed = self._count()

        # The batch that intends to delete both docs.
        self.deletes = spark.sql("""
            SELECT 'k1' AS doc_id, CAST(NULL AS INT) AS n, true AS _is_delete UNION ALL
            SELECT 'k2', CAST(NULL AS INT), true
        """)

        # --- the misspelled delete_flag_column (the live silent failure) ---
        typo_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id",
                            has_deletes=True, delete_flag_column="_is_deleted")  # real col: _is_delete
        self.flag_error = None
        try:
            bulk_write(self.deletes, typo_cfg, raise_on_error=True)
        except Exception as exc:
            self.flag_error = exc
        self._refresh()
        self.count_after_typo = self._count()
        # Snapshot k1's _source HERE, not in the assertion: the last setup phase below deletes k1 and
        # k2 with the correct config, so a live read at assertion time would find nothing and fail
        # for a reason that has nothing to do with the guard under test.
        self.k1_source_after_typo = self._source_of("k1")

        # --- a misspelled id_field ---
        id_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                          index=INDEX, id_field="docid")                          # real col: doc_id
        self.id_error = None
        try:
            bulk_write(spark.sql("SELECT 'k9' AS doc_id, 9 AS n"), id_cfg)
        except Exception as exc:
            self.id_error = exc

        # --- a misspelled drop_fields entry ---
        drop_cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id", drop_fields=("nn",))  # real col: n
        self.drop_error = None
        try:
            bulk_write(spark.sql("SELECT 'k9' AS doc_id, 9 AS n"), drop_cfg)
        except Exception as exc:
            self.drop_error = exc

        # --- the correct config still deletes (the guard-rail against over-blocking) ---
        self.res_good = bulk_write(self.deletes, self.good_cfg, raise_on_error=True)
        self._refresh()
        self.count_after_good = self._count()

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    def _refresh(self):
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)

    def _count(self):
        return requests.get(f"{ES_HOSTS}/{INDEX}/_count", auth=ES_AUTH,
                            verify=False, timeout=30).json()["count"]

    def _source_of(self, doc_id):
        r = requests.get(f"{ES_HOSTS}/{INDEX}/_doc/{doc_id}", auth=ES_AUTH, verify=False, timeout=30)
        return r.json().get("_source", {}) if r.status_code == 200 else None

    # --- setup sanity ---
    def test_seed_wrote_two_docs(self):
        assert self.res_seed["written"] == 2, self.res_seed
        assert self.count_after_seed == 2, self.count_after_seed

    # --- the delete_flag_column guard: the finding this fixture exists for ---
    def test_misspelled_delete_flag_column_raises(self):
        assert self.flag_error is not None, (
            "a misspelled delete_flag_column did NOT raise: every intended delete would have been "
            "applied as an upsert with clean counts")
        assert isinstance(self.flag_error, ValueError), type(self.flag_error)

    def test_misspelled_delete_flag_error_is_actionable(self):
        msg = str(self.flag_error)
        assert "_is_deleted" in msg, msg          # the name they typed
        assert "_is_delete" in msg, msg           # the columns actually available
        assert "upsert" in msg, msg               # what would have gone wrong

    def test_misspelled_delete_flag_wrote_nothing(self):
        # The point of a PREFLIGHT: it fails before any document is touched, so ES is untouched
        # (not partially upserted). Both seeded docs are still present and unmodified. Asserted
        # against the snapshot taken right after that write attempt, since later setup phases
        # legitimately delete these docs.
        assert self.count_after_typo == 2, self.count_after_typo
        assert self.k1_source_after_typo == {"doc_id": "k1", "n": 1}, self.k1_source_after_typo

    # --- the sibling fields, same class ---
    def test_misspelled_id_field_raises(self):
        assert self.id_error is not None, "a misspelled id_field must fail before writing"
        assert "id_field" in str(self.id_error), str(self.id_error)

    def test_misspelled_drop_field_raises(self):
        assert self.drop_error is not None, "a misspelled drop_fields entry must fail closed"
        assert "drop_fields" in str(self.drop_error), str(self.drop_error)

    # --- guard-rail: the guards must not block a CORRECT config ---
    def test_correct_config_still_deletes(self):
        assert self.res_good["deleted"] == 2, self.res_good
        assert self.res_good["errors"] == 0, self.res_good
        assert self.count_after_good == 0, self.count_after_good


# COMMAND ----------
# Return the results through notebook.exit so the runner reports each test by NAME. Calling
# run_notebook_tests() bare makes the tier count the whole fixture as one opaque pass, which cannot
# be attributed to the assertions that actually ran.
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
