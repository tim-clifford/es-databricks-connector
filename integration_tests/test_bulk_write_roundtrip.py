# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: bulk_write round-trip through real Spark + Arrow + Elasticsearch
# MAGIC The pure-Python suite tests `coerce_value` / the partition writer / the driver merge in
# MAGIC isolation (monkeypatched ES, hand-built dicts). It never runs the actual
# MAGIC `df.mapInPandas(...).collect()` path: real Arrow conversion of every dtype, partition fan-out,
# MAGIC and the result-schema (`total_input` / `error_samples`) surviving the Spark→driver trip.
# MAGIC This fixture does, end-to-end against a live ES index, and locks in the 0.3.1 fidelity fixes
# MAGIC (non-string map keys, float32 widening, pre-epoch timestamp floor) at the Spark layer.

# COMMAND ----------
import json, base64, datetime, requests, urllib3
urllib3.disable_warnings()
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-roundtrip"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))


class TestBulkWriteRoundtrip(NotebookTestFixture):
    """Write a wide, edge-case-laden DataFrame to ES via the connector, read it back, assert
    the documented transforms held and the 0.3.1 result contract is present."""

    def run_setup(self):
        # A single row exercising the transforms most likely to regress, including the exact
        # 0.3.1 edge cases. Built via SQL so the Spark column types are precise.
        self.df = spark.sql("""
            SELECT
              'doc1'                                        AS doc_id,
              CAST(0.1 AS FLOAT)                            AS f32,          -- float32 widening
              CAST(0.1 AS DOUBLE)                           AS f64,
              CAST(1.50 AS DECIMAL(10,2))                   AS dec,          -- -> float
              DATE'2021-01-01'                              AS d,            -- -> epoch millis
              TIMESTAMP'2021-01-01 00:00:00Z'               AS ts,
              TIMESTAMP'1969-12-31 23:59:59Z'               AS ts_preepoch,  -- pre-epoch floor
              CAST(X'0102' AS BINARY)                       AS bin,          -- -> base64
              map(1, 'a', 2, 'b')                           AS m_int_keys,   -- non-string map keys
              map(DATE'2021-01-01', 'x')                    AS m_date_keys,  -- crash case pre-0.3.1
              named_struct('ip','10.0.0.1','port',443)      AS strct,
              array(1, 2, 3)                                AS arr
        """)
        self.cfg = EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                            index=INDEX, id_field="doc_id", http_compress=True)

        # Clean slate, write via the connector (real mapInPandas + Arrow + ES bulk), read back.
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        self.result = bulk_write(self.df, self.cfg)
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        hits = requests.get(f"{ES_HOSTS}/{INDEX}/_search", auth=ES_AUTH, verify=False, timeout=30,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps({"query": {"match_all": {}}})).json()
        docs = hits.get("hits", {}).get("hits", [])
        self.src = docs[0]["_source"] if docs else {}

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    # --- the write result contract (0.3.1) ---
    def test_write_succeeded_no_errors(self):
        assert self.result["written"] == 1, self.result
        assert self.result["errors"] == 0, self.result

    def test_result_has_total_input_reconciling(self):
        # 0.3.1: total_input present, and written+deleted+errors reconciles against it.
        assert self.result["total_input"] == 1
        assert self.result["written"] + self.result["deleted"] + self.result["errors"] == 1

    def test_result_has_error_samples_field(self):
        # 0.3.1: error_samples always present; empty on a clean write.
        assert self.result["error_samples"] == []

    # --- documented value transforms, verified through a real round-trip ---
    def test_doc_present(self):
        assert self.src.get("doc_id") == "doc1"

    def test_float32_widens_to_exact_32bit_value(self):
        # 0.3.1 documented: a FLOAT stores its exact widened value, not the literal 0.1.
        assert self.src["f32"] == 0.10000000149011612
        assert self.src["f64"] == 0.1

    def test_decimal_to_float(self):
        assert self.src["dec"] == 1.5

    def test_date_and_timestamp_epoch_millis(self):
        assert self.src["d"] == 1609459200000
        assert self.src["ts"] == 1609459200000

    def test_preepoch_timestamp_floors(self):
        # 0.3.1: floor (not truncate-toward-zero). One whole second before epoch = -1000 ms.
        assert self.src["ts_preepoch"] == -1000

    def test_binary_base64(self):
        assert self.src["bin"] == base64.b64encode(b"\x01\x02").decode("ascii")

    def test_int_map_keys_become_strings(self):
        # 0.3.1: non-string map keys are rendered to strings (pre-0.3.1 silently mutated).
        assert self.src["m_int_keys"] == {"1": "a", "2": "b"}

    def test_date_map_keys_do_not_crash_and_serialize(self):
        # 0.3.1: a date-keyed map used to crash json.dumps on the executor; now the key is coerced
        # (date -> epoch-millis) then stringified, same as a date value.
        assert self.src["m_date_keys"] == {"1609459200000": "x"}

    def test_struct_and_array_preserved(self):
        assert self.src["strct"] == {"ip": "10.0.0.1", "port": 443}
        assert self.src["arr"] == [1, 2, 3]


# COMMAND ----------
# Pass the fixture class explicitly (see note in test_sanitize_for_arrow): no-arg auto-discovery
# finds zero fixtures through this wrapper, so we name the class.
dbutils.notebook.exit(json.dumps(run_notebook_tests(TestBulkWriteRoundtrip)))
