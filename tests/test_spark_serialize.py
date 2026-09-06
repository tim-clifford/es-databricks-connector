"""Unit tests for the serialize_in_spark write path (bulk.py shipper + config validation).

The Spark-side builder (spark_serialize.build_ndjson) needs a live Spark session and is proven in
the integration tier; here we cover everything that does NOT need Spark:
  - iter_bulk_response_outcomes: es.bulk response item -> WRITTEN/DELETED/IGNORED/ERROR (reuses the
    same classify_bulk_result rules as streaming_bulk, so the two paths count identically).
  - _ship_ndjson_chunk: tallying + the per-document 429 retry the connector treats as load-bearing.
  - make_ndjson_partition_writer: chunking by chunk_size, the yielded summary schema, total_input.
  - the config guard that fails closed on serialize_in_spark + has_deletes.
  - _payload_columns: which columns land in _source.
"""
import pytest

from databricks_es_connector.config import EsConfig
from databricks_es_connector.bulk import (
    iter_bulk_response_outcomes, _ship_ndjson_chunk, make_ndjson_partition_writer, _preflight,
    WRITTEN, DELETED, IGNORED, ERROR, ERROR_SAMPLE_CAP,
)


def _cfg(**kw):
    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id",
                serialize_in_spark=True, require_existing_index=False)
    base.update(kw)
    return EsConfig(**base)


class _FakeES:
    """Returns queued canned responses, one per bulk() call. Records the lines it was asked to send."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []          # list of the `operations` lists per call

    def bulk(self, operations=None, **kw):
        self.calls.append(list(operations))
        return self._responses.pop(0)


# --- classification -------------------------------------------------------------------------

def test_iter_bulk_response_outcomes_classifies():
    items = [
        {"index":  {"status": 201}},
        {"index":  {"status": 200}},
        {"delete": {"status": 200}},
        {"delete": {"status": 404}},                 # expected no-op -> IGNORED
        {"index":  {"status": 409}},                 # real error
        {"delete": {"status": 409}},                 # non-404 delete -> real error
    ]
    outcomes = [o for (_op, _b, _ok, o) in iter_bulk_response_outcomes(items)]
    assert outcomes == [WRITTEN, WRITTEN, DELETED, IGNORED, ERROR, ERROR]


def test_iter_bulk_response_outcomes_missing_status_is_error():
    # A malformed item with no status must fail closed (ERROR), never be read as success.
    (op, body, ok, outcome), = list(iter_bulk_response_outcomes([{"index": {}}]))
    assert ok is False and outcome == ERROR


def test_iter_bulk_response_outcomes_empty_item_is_error_not_crash():
    # An empty item {} must fail that doc closed (ERROR), NOT raise: next(iter({}.items())) would
    # StopIteration -> RuntimeError (PEP-479) and abort the whole mapInPandas partition.
    (op, body, ok, outcome), = list(iter_bulk_response_outcomes([{}]))
    assert outcome == ERROR and ok is False and op == "unknown"


def test_ship_chunk_empty_item_counts_error_not_crash():
    es = _FakeES([{"items": [{"index": {"status": 201}}, {}]}])   # one good, one empty
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, [])
    assert counts["written"] == 1 and counts["errors"] == 1        # no crash, empty -> error


# --- _ship_ndjson_chunk: tally + retry ------------------------------------------------------

def test_ship_chunk_tallies_mixed_outcomes():
    resp = {"items": [
        {"index":  {"status": 201, "_id": "a"}},
        {"delete": {"status": 404, "_id": "b"}},     # IGNORED
        {"index":  {"status": 400, "_id": "c",
                    "error": {"type": "mapper_parsing_exception", "reason": "boom"}}},
    ]}
    es = _FakeES([resp])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["l1", "l2", "l3"], _cfg(), counts, samples)
    assert counts == {"written": 1, "deleted": 0, "ignored": 1, "errors": 1}
    assert len(samples) == 1 and samples[0]["_id"] == "c" and "boom" in samples[0]["reason"]
    assert len(es.calls) == 1


def test_ship_chunk_retries_only_the_429_line_then_succeeds():
    # First call: line 2 is 429 (retryable). Retry must resend ONLY line 2, which then succeeds.
    es = _FakeES([
        {"items": [{"index": {"status": 201}}, {"index": {"status": 429}}, {"index": {"status": 201}}]},
        {"items": [{"index": {"status": 201}}]},
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(max_retries_per_doc=3), counts, [])
    assert counts["written"] == 3 and counts["errors"] == 0
    assert es.calls[0] == ["a", "b", "c"]
    assert es.calls[1] == ["b"]          # only the retryable line was resent


def test_ship_chunk_429_becomes_error_after_max_retries():
    # A 429 that never clears is counted as an error once retries are exhausted (loud, not lost).
    always_429 = {"items": [{"index": {"status": 429, "_id": "x"}}]}
    es = _FakeES([always_429, always_429, always_429])   # initial + 2 retries
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["x"], _cfg(max_retries_per_doc=2), counts, [])
    assert counts["errors"] == 1 and counts["written"] == 0
    assert len(es.calls) == 3            # 1 initial + 2 retries, then give up


# --- make_ndjson_partition_writer -----------------------------------------------------------

def test_ndjson_writer_schema_and_counts(monkeypatch):
    pd = pytest.importorskip("pandas")
    import json
    import elasticsearch

    resp = {"items": [{"index": {"status": 201}}, {"index": {"status": 201}},
                      {"index": {"status": 400, "_id": "z", "error": {"reason": "no"}}}]}
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: _FakeES([resp]))

    writer = make_ndjson_partition_writer(_cfg(chunk_size=500))
    out = list(writer(iter([pd.DataFrame({"_ndjson": ["l1", "l2", "l3"]})])))
    assert len(out) == 1
    row = out[0].iloc[0]
    assert list(out[0].columns) == ["written", "deleted", "errors", "ignored", "coerced_nonfinite",
                                    "total_input", "error_samples"]
    assert (int(row["written"]), int(row["errors"]), int(row["total_input"])) == (2, 1, 3)
    assert int(row["coerced_nonfinite"]) == 0            # documented: not tracked in this mode
    assert json.loads(row["error_samples"])[0]["_id"] == "z"


def test_ndjson_writer_chunks_by_chunk_size(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch

    ok1 = lambda: {"items": [{"index": {"status": 201}}]}
    # 5 rows, chunk_size=2 -> chunks of [2,2,1] -> 3 bulk calls.
    es = _FakeES([{"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]},
                  {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]},
                  {"items": [{"index": {"status": 201}}]}])
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)

    writer = make_ndjson_partition_writer(_cfg(chunk_size=2))
    out = list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b", "c", "d", "e"]})])))
    row = out[0].iloc[0]
    assert int(row["written"]) == 5 and int(row["total_input"]) == 5
    assert [len(c) for c in es.calls] == [2, 2, 1]


def test_ndjson_writer_null_line_raises(monkeypatch):
    # A null action line means build_ndjson hit a null/non-finite id. The writer must RAISE
    # (failing the write unconditionally, like _require_id), not silently drop it to `unaccounted`
    # (which only surfaces under raise_on_error=True). Both None and float-NaN nulls must trip it.
    pd = pytest.importorskip("pandas")
    import elasticsearch

    es = _FakeES([{"items": [{"index": {"status": 201}}]}])
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)
    writer = make_ndjson_partition_writer(_cfg(chunk_size=500))
    with pytest.raises(ValueError, match="null action line"):
        list(writer(iter([pd.DataFrame({"_ndjson": ["good", None]})])))
    with pytest.raises(ValueError, match="null action line"):
        list(writer(iter([pd.DataFrame({"_ndjson": ["good", float("nan")]})])))


def test_ship_chunk_transport_error_counts_errors_not_crash():
    # A whole-request transport failure (es.bulk raises) must be recorded as chunk errors and NOT
    # propagate out (which would abort the mapInPandas partition). Mirrors streaming_bulk's
    # raise_on_exception=False.
    class _RaisingES:
        def bulk(self, operations=None, **kw):
            raise ConnectionError("es unreachable")
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(_RaisingES(), ["a", "b", "c"], _cfg(), counts, samples)   # must not raise
    assert counts["errors"] == 3 and counts["written"] == 0
    assert len(samples) == 1 and "ConnectionError" in samples[0]["reason"]


# --- config guard ---------------------------------------------------------------------------

def test_config_accepts_serialize_in_spark_with_deletes():
    # 0.9.0: the serialize_in_spark path builds delete actions too, so the old fail-closed guard
    # (which rejected the combination) is gone. The remaining requirement -- the flag column must be
    # boolean -- needs the DataFrame schema and is enforced in bulk._preflight, not here.
    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id",
                   serialize_in_spark=True, has_deletes=True, delete_flag_column="d")
    assert cfg.has_deletes is True and cfg.delete_flag_column == "d"


def test_config_serialize_in_spark_defaults_off():
    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i")
    assert cfg.serialize_in_spark is False


def test_config_warns_write_concurrency_with_serialize_in_spark():
    import warnings
    # write_concurrency>1 has no effect on the serialize_in_spark path; must WARN (not silently ignore).
    with pytest.warns(UserWarning, match="effect with serialize_in_spark"):
        EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i",
                 serialize_in_spark=True, write_concurrency=4)
    # No warning when the two are not combined.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", serialize_in_spark=True)
        EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", write_concurrency=4)


# --- _payload_columns (pure) ----------------------------------------------------------------

def test_payload_columns_keeps_id_drops_only_drop_fields():
    from databricks_es_connector.spark_serialize import _payload_columns
    cols = ["id", "a", "b", "secret", "c"]
    # id_field is KEPT (document stays self-describing); only drop_fields are removed; order preserved.
    assert _payload_columns(cols, ("secret",)) == ["id", "a", "b", "c"]
    assert _payload_columns(cols, ()) == cols
    assert _payload_columns(cols, None) == cols


# --- _type_has_float (pure): drives the recursive non-finite guard --------------------------

def test_type_has_float_detects_nested():
    pytest.importorskip("pyspark")
    from pyspark.sql.types import (ArrayType, DoubleType, FloatType, IntegerType, MapType,
                                   StringType, StructField, StructType)
    from databricks_es_connector.spark_serialize import _type_has_float

    assert _type_has_float(DoubleType()) is True
    assert _type_has_float(FloatType()) is True
    assert _type_has_float(IntegerType()) is False
    assert _type_has_float(StringType()) is False
    # nested: struct/array/map that CONTAIN a float must be detected (else the guard skips them and a
    # nested NaN reaches to_json).
    assert _type_has_float(StructType([StructField("a", IntegerType()),
                                       StructField("b", DoubleType())])) is True
    assert _type_has_float(ArrayType(FloatType())) is True
    assert _type_has_float(MapType(StringType(), DoubleType())) is True
    assert _type_has_float(ArrayType(StructType([StructField("x", DoubleType())]))) is True
    # no float anywhere -> not walked
    assert _type_has_float(StructType([StructField("a", IntegerType()),
                                       StructField("s", StringType())])) is False
    assert _type_has_float(ArrayType(IntegerType())) is False


# --- _type_has_date_or_ntz / _epoch_type (pure): drive the date/ntz -> epoch-millis rewrite --------

def test_type_has_date_or_ntz_detects_nested():
    pytest.importorskip("pyspark")
    from pyspark.sql.types import (ArrayType, DateType, IntegerType, MapType, StringType,
                                   StructField, StructType, TimestampNTZType, TimestampType)
    from databricks_es_connector.spark_serialize import _type_has_date_or_ntz

    assert _type_has_date_or_ntz(DateType()) is True
    assert _type_has_date_or_ntz(TimestampNTZType()) is True
    # A plain TimestampType is NOT matched here: it is already an epoch-millis Long by the time
    # build_ndjson runs (normalize_timestamps_for_utc converted it upstream), so this rewrite must
    # leave it alone. Guards against double-converting the (now integer) timestamp column.
    assert _type_has_date_or_ntz(TimestampType()) is False
    assert _type_has_date_or_ntz(IntegerType()) is False
    # nested: a date/ntz inside struct/array/map must be detected (else the rewrite skips it and a
    # nested date/ntz reaches to_json as an ISO string, breaking the round-trip).
    assert _type_has_date_or_ntz(StructType([StructField("a", IntegerType()),
                                             StructField("d", DateType())])) is True
    assert _type_has_date_or_ntz(ArrayType(TimestampNTZType())) is True
    assert _type_has_date_or_ntz(MapType(StringType(), DateType())) is True
    assert _type_has_date_or_ntz(ArrayType(StructType([StructField("n", TimestampNTZType())]))) is True
    # no date/ntz anywhere -> not walked
    assert _type_has_date_or_ntz(StructType([StructField("a", IntegerType()),
                                             StructField("t", TimestampType())])) is False
    assert _type_has_date_or_ntz(ArrayType(StringType())) is False


def test_epoch_type_maps_date_ntz_to_long_recursively():
    pytest.importorskip("pyspark")
    from pyspark.sql.types import (ArrayType, DateType, IntegerType, LongType, MapType, StringType,
                                   StructField, StructType, TimestampNTZType)
    from databricks_es_connector.spark_serialize import _epoch_type

    # A null struct literal is typed with this so `when(null)` keeps the rewritten schema: every
    # DateType/TimestampNTZType leaf becomes LongType, everything else is unchanged.
    assert isinstance(_epoch_type(DateType()), LongType)
    assert isinstance(_epoch_type(TimestampNTZType()), LongType)
    assert isinstance(_epoch_type(StringType()), StringType)
    nested = StructType([StructField("a", IntegerType()), StructField("d", DateType()),
                         StructField("n", TimestampNTZType()), StructField("s", StringType())])
    out = _epoch_type(nested)
    got = {f.name: type(f.dataType).__name__ for f in out.fields}
    assert got == {"a": "IntegerType", "d": "LongType", "n": "LongType", "s": "StringType"}
    # array/map element/value types recurse too
    assert isinstance(_epoch_type(ArrayType(DateType())).elementType, LongType)
    assert isinstance(_epoch_type(MapType(StringType(), TimestampNTZType())).valueType, LongType)
    # a date/ntz map KEY is left UNCHANGED (map keys are not temporally converted on this path), so the
    # null-branch literal type matches the rebuilt map whose keys are untouched; only the VALUE side of
    # a map is mapped to Long.
    mk = _epoch_type(MapType(DateType(), TimestampNTZType()))
    assert isinstance(mk.keyType, DateType) and isinstance(mk.valueType, LongType)


# --- _preflight: serialize_in_spark deletes require a BooleanType flag column ------------------
# The Catalyst delete routing (`flag === true` in build_ndjson) has no way to parse a string/int flag,
# so a non-boolean flag must fail closed on the driver rather than silently upsert every intended
# delete. build_ndjson itself needs live Spark (the end-to-end path is proven in the integration
# tier); this covers the driver-side TYPE check that gates it. _preflight reads only df.columns and
# df.schema.fields[*].{name, dataType.typeName()}, so a stand-in exercises the check without pyspark
# (unavailable on this Python) -- the real schema is proven live in test_deletes_roundtrip.

class _FakeDataType:
    """Mirrors the pyspark DataType methods _preflight reads: typeName() / simpleString()."""
    def __init__(self, type_name): self._t = type_name
    def typeName(self): return self._t
    def simpleString(self): return self._t


class _FakeField:
    def __init__(self, name, type_name): self.name = name; self.dataType = _FakeDataType(type_name)


def _fake_df(fields):
    """Stand-in DataFrame exposing what _preflight reads: .columns and .schema.fields[*].{name,
    dataType}. `fields`: [(name, type_name), ...], e.g. ("d", "boolean")."""
    df = type("_DF", (), {})()
    df.columns = [n for n, _ in fields]
    df.schema = type("_Schema", (), {"fields": [_FakeField(n, t) for n, t in fields]})()
    return df


def test_preflight_rejects_non_boolean_delete_flag_with_serialize_in_spark():
    # flag column "d" is a STRING, not boolean: `flag === true` would be null for every row => no row
    # routed to a delete => every intended deletion silently upserted. Must fail closed on the driver.
    cfg = _cfg(has_deletes=True, delete_flag_column="d")   # _cfg sets serialize_in_spark=True, id_field="id"
    with pytest.raises(ValueError, match="must be a boolean column"):
        _preflight(_fake_df([("id", "string"), ("d", "string")]), cfg)


def test_preflight_accepts_boolean_delete_flag_with_serialize_in_spark():
    cfg = _cfg(has_deletes=True, delete_flag_column="d")
    _preflight(_fake_df([("id", "string"), ("d", "boolean")]), cfg)   # must not raise (require_existing_index=False)
