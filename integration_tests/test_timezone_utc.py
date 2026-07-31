# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: timestamp epoch is session-timezone-independent
# MAGIC Pins the timezone contract from both sides against live ES:
# MAGIC
# MAGIC 1. **UTC session (regression guard):** a `timestamp` -- top-level AND nested in
# MAGIC    struct/array/map -- must store its TRUE UTC instant and read back exactly. This is the
# MAGIC    behavior that was already correct under UTC; these tests lock it so the tz fix (or any
# MAGIC    future change) can't silently alter it.
# MAGIC 2. **Non-UTC session (the fix):** the SAME instants written under `America/New_York` must
# MAGIC    store the SAME epoch as under UTC. Before the fix (spark_prep.normalize_timestamps_for_utc)
# MAGIC    the non-UTC epoch was shifted by the session offset (-5h); after it, the two agree.
# MAGIC
# MAGIC Companion to test_datatype_coverage.py (which runs its full matrix under a non-UTC session).
# MAGIC Live ES + the `es_poc` scope required. Throwaway index, dropped in cleanup.

# COMMAND ----------
import json, datetime, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-timezone"          # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# Known instants (session-INDEPENDENT ground truth), computed the plain way from a UTC datetime.
def _epoch(y, mo, d, h=0, mi=0, s=0):
    return int(datetime.datetime(y, mo, d, h, mi, s, tzinfo=datetime.timezone.utc).timestamp() * 1000)

TS_INSTANT = _epoch(2021, 1, 1, 0, 0, 0)          # TIMESTAMP'2021-01-01 00:00:00Z' -> 1609459200000
TS_PREEPOCH = -1000                                # TIMESTAMP'1969-12-31 23:59:59Z'
NTZ_AS_UTC = _epoch(2021, 6, 1, 12, 0, 0)          # TIMESTAMP_NTZ wall-clock read as UTC
DATE_MIDNIGHT = _epoch(2021, 1, 1)                 # DATE'2021-01-01' -> midnight UTC

# One row exercising a timestamp at every nesting depth, plus ntz/date (which must NOT shift).
_ROW_SQL = """
  SELECT '{doc_id}' AS doc_id,
    TIMESTAMP'2021-01-01 00:00:00Z'                       AS s_ts,
    TIMESTAMP'1969-12-31 23:59:59Z'                       AS s_ts_preepoch,
    named_struct('t', TIMESTAMP'2021-01-01 00:00:00Z')    AS s_struct_ts,
    array(TIMESTAMP'2021-01-01 00:00:00Z',
          TIMESTAMP'2021-01-01 00:00:00Z')                AS s_array_ts,
    map('k', TIMESTAMP'2021-01-01 00:00:00Z')             AS s_map_ts,
    named_struct('a', array(TIMESTAMP'2021-01-01 00:00:00Z')) AS s_struct_array_ts,
    TIMESTAMP_NTZ'2021-06-01 12:00:00'                    AS s_ntz,
    DATE'2021-01-01'                                      AS s_date
"""


class TestTimezoneEpochStability(NotebookTestFixture):
    """A timestamp's stored epoch is the true UTC instant regardless of spark.sql.session.timeZone,
    at every nesting depth. Writes the same instants under UTC and under America/New_York and asserts
    both land on the ground-truth epoch (and that ntz/date are unaffected)."""

    def _write_under(self, tz, doc_id):
        spark.conf.set("spark.sql.session.timeZone", tz)
        df = spark.sql(_ROW_SQL.format(doc_id=doc_id))
        bulk_write(df, EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                index=INDEX, id_field="doc_id", http_compress=True))

    def run_setup(self):
        # Fresh index; all temporal columns mapped epoch_millis so ES stores the number verbatim.
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        ts_cols = ["s_ts", "s_ts_preepoch", "s_ntz", "s_date"]
        props = {"doc_id": {"type": "keyword"}}
        props.update({c: {"type": "date", "format": "epoch_millis"} for c in ts_cols})
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"},
                     data=json.dumps({"settings": {"index": {"number_of_shards": 1,
                                                            "number_of_replicas": 0}},
                                      "mappings": {"properties": props}}))
        try:
            self._write_under("UTC", "utc")
            self._write_under("America/New_York", "ny")
        finally:
            spark.conf.set("spark.sql.session.timeZone", "UTC")   # leave the session as we found it
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        hits = requests.get(f"{ES_HOSTS}/{INDEX}/_search", auth=ES_AUTH, verify=False, timeout=30,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps({"size": 10, "query": {"match_all": {}}})).json()
        docs = {h["_id"]: h["_source"] for h in hits.get("hits", {}).get("hits", [])}
        self.utc = docs.get("utc", {})
        self.ny = docs.get("ny", {})

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- both rows present ---
    def test_both_rows_written(self):
        assert self.utc and self.ny, (bool(self.utc), bool(self.ny))

    # --- REGRESSION GUARD: UTC-session behavior is the true instant, unchanged ---
    def test_utc_session_top_level_timestamp_is_true_instant(self):
        assert self.utc["s_ts"] == TS_INSTANT, (self.utc["s_ts"], TS_INSTANT)
        assert self.utc["s_ts_preepoch"] == TS_PREEPOCH, self.utc["s_ts_preepoch"]

    def test_utc_session_nested_timestamps_are_true_instant(self):
        assert self.utc["s_struct_ts"]["t"] == TS_INSTANT, self.utc["s_struct_ts"]
        assert self.utc["s_array_ts"] == [TS_INSTANT, TS_INSTANT], self.utc["s_array_ts"]
        assert self.utc["s_map_ts"]["k"] == TS_INSTANT, self.utc["s_map_ts"]
        assert self.utc["s_struct_array_ts"]["a"][0] == TS_INSTANT, self.utc["s_struct_array_ts"]

    def test_utc_session_ntz_and_date(self):
        assert self.utc["s_ntz"] == NTZ_AS_UTC, self.utc["s_ntz"]
        assert self.utc["s_date"] == DATE_MIDNIGHT, self.utc["s_date"]

    # --- THE FIX: a non-UTC session stores the SAME epoch as UTC (no session-offset shift) ---
    def test_non_utc_session_matches_utc_top_level(self):
        assert self.ny["s_ts"] == TS_INSTANT, (self.ny["s_ts"], TS_INSTANT)
        assert self.ny["s_ts_preepoch"] == TS_PREEPOCH, self.ny["s_ts_preepoch"]

    def test_non_utc_session_matches_utc_nested(self):
        assert self.ny["s_struct_ts"]["t"] == TS_INSTANT, self.ny["s_struct_ts"]
        assert self.ny["s_array_ts"] == [TS_INSTANT, TS_INSTANT], self.ny["s_array_ts"]
        assert self.ny["s_map_ts"]["k"] == TS_INSTANT, self.ny["s_map_ts"]
        assert self.ny["s_struct_array_ts"]["a"][0] == TS_INSTANT, self.ny["s_struct_array_ts"]

    def test_non_utc_ntz_and_date_also_stable(self):
        # ntz is zoneless and date has no time-of-day: both are already session-independent and must
        # stay equal to the UTC-session values (the fix must not touch them).
        assert self.ny["s_ntz"] == self.utc["s_ntz"] == NTZ_AS_UTC, (self.ny["s_ntz"], self.utc["s_ntz"])
        assert self.ny["s_date"] == self.utc["s_date"] == DATE_MIDNIGHT, (self.ny["s_date"], self.utc["s_date"])

    # --- the two sessions agree field-by-field on every temporal column ---
    def test_utc_and_non_utc_sessions_agree(self):
        cols = ["s_ts", "s_ts_preepoch", "s_struct_ts", "s_array_ts", "s_map_ts",
                "s_struct_array_ts", "s_ntz", "s_date"]
        diffs = {c: (self.utc.get(c), self.ny.get(c)) for c in cols if self.utc.get(c) != self.ny.get(c)}
        assert not diffs, f"session-dependent epochs (fix regressed): {diffs}"


# COMMAND ----------
dbutils.notebook.exit(json.dumps(run_notebook_tests()))
