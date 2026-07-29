# Databricks notebook source
# MAGIC %md
# MAGIC # Integration: every Spark datatype + edge cases, round-tripped through the connector
# MAGIC The exhaustive datatype-fidelity test. Builds one wide row exercising **every** Spark type and
# MAGIC the edge cases a symmetric round-trip can hide, plus an all-NULL row of the same schema, writes
# MAGIC both through the connector to a live ES index, reads them back, and asserts each value equals
# MAGIC the connector's **documented** transform (see the connector README "Datatype coverage" table).
# MAGIC
# MAGIC Formalizes the `datatype_coverage` demo as a permanent connector test. Companion:
# MAGIC `test_bulk_write_roundtrip.py` owns the write-result contract (total_input / error_samples);
# MAGIC this fixture owns "every datatype round-trips unchanged except the documented transforms".
# MAGIC
# MAGIC Live ES + the `es_poc` secret scope required. Throwaway index, dropped in cleanup.

# COMMAND ----------
import json, math, base64, datetime, requests, urllib3
urllib3.disable_warnings()
import numpy as np
from dbx_test import NotebookTestFixture, run_notebook_tests
from databricks_es_connector import EsConfig, bulk_write

SCOPE = "es_poc"
INDEX = "connector-integration-datatypes"   # throwaway; recreated + dropped by the fixture
ES_HOSTS = dbutils.secrets.get(SCOPE, "hosts")
ES_AUTH = (dbutils.secrets.get(SCOPE, "username"), dbutils.secrets.get(SCOPE, "password"))

# Columns the connector serializes to a JSON/interval STRING (Arrow-hostile). Mapped as keyword so
# ES accepts the string (an object mapping would reject it), and compared parsed-equal below.
VARIANT_COLS = {"s_variant", "s_struct_variant", "s_array_variant"}
INTERVAL_COLS = {"s_interval_dt", "s_interval_ym"}


def _epoch_millis(dt_):
    if isinstance(dt_, datetime.datetime):
        v = dt_ if dt_.tzinfo else dt_.replace(tzinfo=datetime.timezone.utc)
    else:
        v = datetime.datetime(dt_.year, dt_.month, dt_.day, tzinfo=datetime.timezone.utc)
    return math.floor(v.timestamp() * 1000)


def _norm(v):
    """Normalize for comparison: numeric 5.0==5, recurse containers, leave None as-is."""
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


class TestDatatypeCoverage(NotebookTestFixture):
    """Every Spark datatype + edge cases, Spark -> connector -> ES -> read back, asserted against
    the documented transforms. One full row and one all-NULL row of the same schema."""

    def run_setup(self):
        # FULL row: one column per Spark type / edge case. Explicit CASTs so the column TYPE is what
        # we intend (createDataFrame would infer from Python values).
        full = spark.sql("""
            SELECT
              'row-full'                                     AS doc_id,
              CAST('hello ☃ 世界' AS STRING)                AS s_string,        -- unicode
              CAST(true AS BOOLEAN)                          AS s_bool,
              CAST(7 AS BYTE)                                AS s_byte,
              CAST(300 AS SHORT)                             AS s_short,
              CAST(70000 AS INT)                             AS s_int,
              CAST(9223372036854775807 AS BIGINT)            AS s_long,          -- max long
              CAST(1.5 AS FLOAT)                             AS s_float,
              CAST(1.5 AS DOUBLE)                            AS s_double,
              CAST(0.1 AS FLOAT)                             AS s_float32_prec,  -- 32-bit widening
              CAST(1.50 AS DECIMAL(10,2))                    AS s_decimal,
              DATE'2021-01-01'                               AS s_date,
              TIMESTAMP'2021-01-01 00:00:00Z'                AS s_timestamp,
              TIMESTAMP'1969-12-31 23:59:59Z'                AS s_ts_preepoch,   -- pre-epoch floor
              TIMESTAMP_NTZ'2021-06-01 12:00:00'             AS s_timestamp_ntz, -- wall-clock, no tz
              CAST(X'0102' AS BINARY)                        AS s_binary,
              named_struct('ip','10.0.0.1','port',443)       AS s_struct,
              map('k1','v1','k2','v2')                        AS s_map,           -- map<string,string>
              map('a', 1, 'b', 2)                             AS s_map_int,       -- map<string,int>
              map(1, 'a', 2, 'b')                             AS s_map_intkey,    -- non-string keys
              CAST(map() AS MAP<STRING,INT>)                  AS s_empty_map,     -- empty map container
              map('a', 1, 'b', CAST(NULL AS INT))             AS s_map_null_val,  -- null map VALUE
              array(1,2,3)                                    AS s_array,
              array(named_struct('k','a','v',1),
                    named_struct('k','b','v',2))              AS s_array_struct,
              array()                                         AS s_empty_array,   -- empty container
              array(1, CAST(NULL AS INT), 3)                  AS s_array_null_el, -- null array ELEMENT
              named_struct('inner', named_struct('x',1))      AS s_nested_struct,
              named_struct('a', 1, 'b', CAST(NULL AS INT))    AS s_struct_null_fld, -- partial-null struct
              CAST(123456789012345678 AS DECIMAL(38,0))       AS s_decimal_hi,    -- 18 sig figs: lossy
              double('Infinity')                              AS s_pos_inf,       -- non-finite
              double('-Infinity')                             AS s_neg_inf,
              double('NaN')                                   AS s_nan,
              parse_json('{"k":1,"nested":[2,3]}')            AS s_variant,       -- VARIANT top-level
              named_struct('v', parse_json('{"a":true}'),
                           'label','hi')                      AS s_struct_variant,-- VARIANT in struct
              array(parse_json('{"n":1}'),
                    parse_json('{"n":2}'))                     AS s_array_variant, -- VARIANT in array
              INTERVAL '1 02:03:04' DAY TO SECOND             AS s_interval_dt,   -- INTERVAL day-time
              INTERVAL '2-3' YEAR TO MONTH                    AS s_interval_ym    -- INTERVAL year-month
        """)
        # All-NULL row (same schema) to prove the null-of-every-type contract. Explicit CASTs because
        # touching df.schema on a VARIANT column throws on Spark Connect.
        null = spark.sql("""
            SELECT
              'row-null'                                     AS doc_id,
              CAST(NULL AS STRING)                           AS s_string,
              CAST(NULL AS BOOLEAN)                          AS s_bool,
              CAST(NULL AS BYTE)                             AS s_byte,
              CAST(NULL AS SHORT)                            AS s_short,
              CAST(NULL AS INT)                              AS s_int,
              CAST(NULL AS BIGINT)                           AS s_long,
              CAST(NULL AS FLOAT)                            AS s_float,
              CAST(NULL AS DOUBLE)                           AS s_double,
              CAST(NULL AS FLOAT)                            AS s_float32_prec,
              CAST(NULL AS DECIMAL(10,2))                    AS s_decimal,
              CAST(NULL AS DATE)                             AS s_date,
              CAST(NULL AS TIMESTAMP)                        AS s_timestamp,
              CAST(NULL AS TIMESTAMP)                        AS s_ts_preepoch,
              CAST(NULL AS TIMESTAMP_NTZ)                    AS s_timestamp_ntz,
              CAST(NULL AS BINARY)                           AS s_binary,
              CAST(NULL AS STRUCT<ip:STRING,port:INT>)       AS s_struct,
              CAST(NULL AS MAP<STRING,STRING>)               AS s_map,
              CAST(NULL AS MAP<STRING,INT>)                  AS s_map_int,
              CAST(NULL AS MAP<INT,STRING>)                  AS s_map_intkey,
              CAST(NULL AS MAP<STRING,INT>)                  AS s_empty_map,
              CAST(NULL AS MAP<STRING,INT>)                  AS s_map_null_val,
              CAST(NULL AS ARRAY<INT>)                       AS s_array,
              CAST(NULL AS ARRAY<STRUCT<k:STRING,v:INT>>)    AS s_array_struct,
              CAST(NULL AS ARRAY<INT>)                       AS s_empty_array,
              CAST(NULL AS ARRAY<INT>)                       AS s_array_null_el,
              CAST(NULL AS STRUCT<inner:STRUCT<x:INT>>)      AS s_nested_struct,
              CAST(NULL AS STRUCT<a:INT,b:INT>)              AS s_struct_null_fld,
              CAST(NULL AS DECIMAL(38,0))                    AS s_decimal_hi,
              CAST(NULL AS DOUBLE)                           AS s_pos_inf,
              CAST(NULL AS DOUBLE)                           AS s_neg_inf,
              CAST(NULL AS DOUBLE)                           AS s_nan,
              CAST(NULL AS VARIANT)                          AS s_variant,
              CAST(NULL AS STRUCT<v:VARIANT,label:STRING>)   AS s_struct_variant,
              CAST(NULL AS ARRAY<VARIANT>)                   AS s_array_variant,
              CAST(NULL AS INTERVAL DAY TO SECOND)           AS s_interval_dt,
              CAST(NULL AS INTERVAL YEAR TO MONTH)           AS s_interval_ym
        """)
        df = full.unionByName(null)

        # Create the index: variant/interval-bearing cols + doc_id as keyword; everything else
        # dynamic-mapped by ES. A variant/interval column arrives as a STRING, which a keyword field
        # accepts but an object field would reject (document_parsing_exception).
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)
        props = {c: {"type": "keyword"} for c in (VARIANT_COLS | INTERVAL_COLS | {"doc_id"})}
        body = {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0,
                                       "mapping": {"total_fields": {"limit": 2000}}}},
                "mappings": {"properties": props}}
        requests.put(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30,
                     headers={"Content-Type": "application/json"}, data=json.dumps(body))

        # Write via the connector (real mapInPandas + Arrow + sanitize_for_arrow for VARIANT/INTERVAL).
        self.result = bulk_write(df, EsConfig(hosts=ES_HOSTS, basic_auth=ES_AUTH, verify_certs=False,
                                              index=INDEX, id_field="doc_id", http_compress=True))
        requests.post(f"{ES_HOSTS}/{INDEX}/_refresh", auth=ES_AUTH, verify=False, timeout=30)
        hits = requests.get(f"{ES_HOSTS}/{INDEX}/_search", auth=ES_AUTH, verify=False, timeout=30,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps({"size": 10, "query": {"match_all": {}}})).json()
        docs = {h["_id"]: h["_source"] for h in hits.get("hits", {}).get("hits", [])}
        self.full = docs.get("row-full", {})
        self.null = docs.get("row-null", {})

    def run_cleanup(self):
        requests.delete(f"{ES_HOSTS}/{INDEX}", auth=ES_AUTH, verify=False, timeout=30)

    def _got(self, col):
        return self.full.get(col, "<<ABSENT>>")

    def _assert(self, col, expected):
        assert _norm(self._got(col)) == _norm(expected), f"{col}: got {self._got(col)!r}, want {expected!r}"

    def _assert_variant(self, col, expected_obj):
        # ES holds the JSON string the connector emitted; compare parsed-equal (tolerates key order).
        got = self._got(col)
        assert isinstance(got, str), f"{col}: expected JSON string, got {type(got).__name__}"
        assert json.loads(got) == expected_obj, f"{col}: parsed {json.loads(got)!r} != {expected_obj!r}"

    # --- write succeeded, both rows present ---
    def test_write_clean(self):
        assert self.result["errors"] == 0, self.result
        assert self.result["written"] == 2, self.result
        assert set([bool(self.full), bool(self.null)]) == {True}, "both rows must be present"

    # --- scalars ---
    def test_string_unicode(self):
        self._assert("s_string", "hello ☃ 世界")

    def test_bool_type_strict(self):
        # In Python True == 1, so a bool that regressed to a number would slip past _norm; require type.
        got = self._got("s_bool")
        assert isinstance(got, bool) and got is True, f"s_bool: got {got!r} ({type(got).__name__})"

    def test_integer_widths(self):
        self._assert("s_byte", 7)
        self._assert("s_short", 300)
        self._assert("s_int", 70000)
        self._assert("s_long", 9223372036854775807)   # max long, must not lose precision

    def test_float_and_double(self):
        self._assert("s_float", 1.5)
        self._assert("s_double", 1.5)

    def test_float32_widening(self):
        # A Spark FLOAT stores its exact 32-bit value widened to double, not the literal 0.1.
        assert self._got("s_float32_prec") == float(np.float32(0.1))

    def test_decimal_to_float(self):
        self._assert("s_decimal", 1.5)

    def test_date_and_timestamp_epoch_millis(self):
        self._assert("s_date", _epoch_millis(datetime.date(2021, 1, 1)))
        self._assert("s_timestamp", _epoch_millis(datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)))

    def test_preepoch_timestamp_floors(self):
        # One whole second before epoch floors to -1000 ms (floor, not truncate-toward-zero).
        self._assert("s_ts_preepoch", -1000)

    def test_timestamp_ntz_interpreted_as_utc(self):
        # TIMESTAMP_NTZ has no zone. It crosses Arrow as a NAIVE datetime, and the connector treats
        # naive datetimes as UTC (deterministic across executor timezones). So a wall-clock
        # 2021-06-01 12:00:00 is stored as the epoch-ms of that instant *in UTC*, NOT the executor's
        # local zone. A client storing wall-clock NTZ values must expect UTC interpretation.
        self._assert("s_timestamp_ntz",
                     _epoch_millis(datetime.datetime(2021, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)))

    def test_binary_base64(self):
        self._assert("s_binary", base64.b64encode(b"\x01\x02").decode("ascii"))

    # --- non-finite floats ---
    def test_non_finite_floats_become_null(self):
        for col in ("s_pos_inf", "s_neg_inf", "s_nan"):
            assert self._got(col) is None, f"{col}: expected null, got {self._got(col)!r}"

    # --- containers ---
    def test_struct(self):
        self._assert("s_struct", {"ip": "10.0.0.1", "port": 443})

    def test_nested_struct(self):
        self._assert("s_nested_struct", {"inner": {"x": 1}})

    def test_map_string_keys(self):
        self._assert("s_map", {"k1": "v1", "k2": "v2"})
        self._assert("s_map_int", {"a": 1, "b": 2})

    def test_map_non_string_keys(self):
        # 0.3.1: int keys rendered to strings (JSON object keys must be strings).
        self._assert("s_map_intkey", {"1": "a", "2": "b"})

    def test_empty_map(self):
        # Empty map() -> empty JSON object, the sibling of the empty-array edge.
        self._assert("s_empty_map", {})

    def test_array(self):
        self._assert("s_array", [1, 2, 3])

    def test_array_of_structs(self):
        self._assert("s_array_struct", [{"k": "a", "v": 1}, {"k": "b", "v": 2}])

    def test_empty_array(self):
        self._assert("s_empty_array", [])

    # --- nulls INSIDE containers (not just a fully-null row) ---
    def test_null_map_value(self):
        # A null map VALUE lands as JSON null, keeping the key (recursion hits _is_null).
        self._assert("s_map_null_val", {"a": 1, "b": None})

    def test_null_array_element(self):
        # A null element inside an array is preserved positionally as JSON null.
        self._assert("s_array_null_el", [1, None, 3])

    def test_partial_null_struct(self):
        # A struct with one null field: present field kept, null field -> JSON null.
        self._assert("s_struct_null_fld", {"a": 1, "b": None})

    # --- decimal precision loss at the documented boundary ---
    def test_decimal_precision_loss(self):
        # An 18-sig-fig decimal widened to a double loses its low digits — the connector's
        # documented decimal->float behavior. Proves the README caveat end-to-end, not just in a
        # unit test: what a client sends (…678) is NOT what ES holds (…680).
        got = self._got("s_decimal_hi")
        assert int(got) == 123456789012345680, f"s_decimal_hi: got {got!r}"
        assert int(got) != 123456789012345678, "expected the low digits to be lost to float64"

    # --- Arrow-hostile: VARIANT (any depth) + INTERVAL, serialized to strings by the connector ---
    def test_variant_top_level(self):
        self._assert_variant("s_variant", {"k": 1, "nested": [2, 3]})

    def test_variant_in_struct(self):
        self._assert_variant("s_struct_variant", {"v": {"a": True}, "label": "hi"})

    def test_variant_in_array(self):
        self._assert_variant("s_array_variant", [{"n": 1}, {"n": 2}])

    def test_intervals_as_string(self):
        # Assert the KNOWN canonical literal (not a value derived from Spark's own cast, which would
        # be tautological). Matches the connector README's documented interval example.
        self._assert("s_interval_dt", "INTERVAL '1 02:03:04' DAY TO SECOND")
        self._assert("s_interval_ym", "INTERVAL '2-3' YEAR TO MONTH")

    # --- null contract: every field present as JSON null (NOT absent, NOT non-null) ---
    def test_null_row_fields_present_as_null(self):
        absent, non_null = [], []
        for col in self.full:
            if col == "doc_id":
                continue
            if col not in self.null:
                absent.append(col)
            elif self.null[col] is not None:
                non_null.append(col)
        assert not absent, f"null-row fields ABSENT from _source (should be present as null): {absent}"
        assert not non_null, f"null-row fields present but NOT null: {non_null}"

    # --- reverse check: the connector must not add or leak any field ---
    def test_no_unexpected_fields(self):
        expected_cols = {
            "doc_id", "s_string", "s_bool", "s_byte", "s_short", "s_int", "s_long", "s_float",
            "s_double", "s_float32_prec", "s_decimal", "s_decimal_hi", "s_date", "s_timestamp",
            "s_ts_preepoch", "s_timestamp_ntz", "s_binary", "s_struct", "s_map", "s_map_int", "s_map_intkey",
            "s_empty_map", "s_map_null_val", "s_array", "s_array_struct", "s_empty_array",
            "s_array_null_el", "s_struct_null_fld", "s_nested_struct", "s_pos_inf", "s_neg_inf",
            "s_nan", "s_variant", "s_struct_variant", "s_array_variant", "s_interval_dt",
            "s_interval_ym",
        }
        # ES omits explicit nulls and an empty array from _source, so the full row may legitimately
        # have FEWER keys than expected — but never MORE. Only extras are a leak.
        extra = set(self.full) - expected_cols
        assert not extra, f"connector emitted unexpected field(s): {sorted(extra)}"


# COMMAND ----------
# Explicit fixture class (no-arg auto-discovery finds nothing through the wrapper).
dbutils.notebook.exit(json.dumps(run_notebook_tests(TestDatatypeCoverage)))
