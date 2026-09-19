"""Unit tests for the write path (bulk.py shipper + config validation + the fan-out).

The Spark-side builder (spark_serialize.build_ndjson) needs a live Spark session and is proven in
the integration tier; here we cover everything that does NOT need Spark:
  - iter_bulk_response_outcomes: es.bulk response item -> WRITTEN/DELETED/IGNORED/ERROR.
  - _ship_ndjson_chunk: tallying + the per-document 429 retry the connector treats as load-bearing.
  - _PipelinedShipper: the write_concurrency cross-batch pipeline (bounded in-flight, no loss/dupe/
    miscount, cross-batch continuity, backpressure bound, straggler tolerance, fail-closed on a
    worker exception).
  - make_ndjson_partition_writer: chunking by chunk_size, the yielded summary schema, total_input.
  - _preflight: deletes require a BooleanType flag column.
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
                require_existing_index=False)
    base.update(kw)
    return EsConfig(**base)


# The full-path (round-2 reship + per-doc retry) filter_path the connector now uses to trim the ES
# response to just what it reads (mirrors bulk._ship_ndjson_chunk's _full_filter_path, no-stats form).
_FULL_FP = "items.*.status,items.*.error,items.*._id"


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
    # id_field=None takes the fast path; a probe error re-ships full, and the empty item must fail that
    # doc closed (ERROR) on the re-ship, not crash.
    es = _FakeES([{"errors": True},
                  {"items": [{"index": {"status": 201}}, {}]}])   # one good, one empty
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, [])
    assert counts["written"] == 1 and counts["errors"] == 1        # no crash, empty -> error


# --- _ship_ndjson_chunk: tally + retry ------------------------------------------------------

def test_ship_chunk_tallies_mixed_outcomes():
    resp = {"items": [
        {"index":  {"status": 201, "_id": "a"}},
        {"delete": {"status": 404, "_id": "b"}},     # IGNORED
        {"index":  {"status": 400, "_id": "c",
                    "error": {"type": "mapper_parsing_exception", "reason": "boom"}}},
    ]}
    # id_field=None fast path: probe error -> full re-ship -> per-item classify (mixed outcomes).
    es = _FakeES([{"errors": True}, resp])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["l1", "l2", "l3"], _cfg(id_field=None), counts, samples)
    assert counts == {"written": 1, "deleted": 0, "ignored": 1, "errors": 1}
    assert len(samples) == 1 and samples[0]["_id"] == "c" and "boom" in samples[0]["reason"]
    assert len(es.calls) == 2          # probe + full re-ship


def test_ship_chunk_retries_only_the_429_line_then_succeeds():
    # id_field=None fast path: probe error -> full re-ship, on which line 2 is 429 (retryable). The
    # retry must resend ONLY line 2, which then succeeds.
    es = _FakeES([
        {"errors": True},
        {"items": [{"index": {"status": 201}}, {"index": {"status": 429}}, {"index": {"status": 201}}]},
        {"items": [{"index": {"status": 201}}]},
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(id_field=None, max_retries_per_doc=3), counts, [])
    assert counts["written"] == 3 and counts["errors"] == 0
    assert es.calls[1] == ["a", "b", "c"]    # full re-ship (es.calls[0] is the probe)
    assert es.calls[2] == ["b"]              # only the retryable line was resent


def test_ship_chunk_429_becomes_error_after_max_retries():
    # A 429 that never clears is counted as an error once retries are exhausted (loud, not lost).
    # id_field=None fast path: probe error -> full re-ship, then the per-doc 429 retry loop.
    always_429 = {"items": [{"index": {"status": 429, "_id": "x"}}]}
    es = _FakeES([{"errors": True}, always_429, always_429, always_429])   # probe + initial + 2 retries
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["x"], _cfg(id_field=None, max_retries_per_doc=2), counts, [])
    assert counts["errors"] == 1 and counts["written"] == 0
    assert len(es.calls) == 4            # probe + 1 initial re-ship + 2 retries, then give up


# --- make_ndjson_partition_writer -----------------------------------------------------------

def test_ndjson_writer_schema_and_counts(monkeypatch):
    pd = pytest.importorskip("pandas")
    import json
    import elasticsearch

    resp = {"items": [{"index": {"status": 201}}, {"index": {"status": 201}},
                      {"index": {"status": 400, "_id": "z", "error": {"reason": "no"}}}]}
    # id_field=None fast path: the probe flags errors (one 400) and the chunk re-ships full for detail.
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: _FakeES([{"errors": True}, resp]))

    writer = make_ndjson_partition_writer(_cfg(id_field=None, chunk_size=500))
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

    # 5 rows, chunk_size=2 -> chunks of [2,2,1] -> 3 clean fast-path probes (no re-ship).
    es = _FakeES([{"errors": False}, {"errors": False}, {"errors": False}])
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)

    writer = make_ndjson_partition_writer(_cfg(id_field=None, chunk_size=2))
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
    writer = make_ndjson_partition_writer(_cfg(id_field=None, chunk_size=500))
    with pytest.raises(ValueError, match="null action line"):
        list(writer(iter([pd.DataFrame({"_ndjson": ["good", None]})])))
    with pytest.raises(ValueError, match="null action line"):
        list(writer(iter([pd.DataFrame({"_ndjson": ["good", float("nan")]})])))


def test_ndjson_writer_reuses_one_pool_across_batches(monkeypatch):
    # A partition arrives as a STREAM of Arrow batches. With write_concurrency>1 the thread pool is
    # created ONCE per partition (in _write) and reused by the _PipelinedShipper for every batch, not
    # spun up per batch. Feed the writer three batches and assert exactly one pool was constructed,
    # while every line still ships exactly once with the correct tally. (RED-BEFORE-GREEN: creating a
    # pool per batch would construct three -> this asserts == 1.)
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import concurrent.futures as cf

    es = _ThreadSafeFakeES()
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)

    pools_created = []
    real_pool = cf.ThreadPoolExecutor
    monkeypatch.setattr(cf, "ThreadPoolExecutor",
                        lambda *a, **k: pools_created.append(real_pool(*a, **k)) or pools_created[-1])

    writer = make_ndjson_partition_writer(_cfg(id_field=None, write_concurrency=3, chunk_size=2))
    batches = [pd.DataFrame({"_ndjson": [f"b0_{i}" for i in range(5)]}),
               pd.DataFrame({"_ndjson": [f"b1_{i}" for i in range(4)]}),
               pd.DataFrame({"_ndjson": [f"b2_{i}" for i in range(3)]})]
    out = list(writer(iter(batches)))
    row = out[0].iloc[0]
    assert int(row["written"]) == 12 and int(row["total_input"]) == 12
    assert len(es.all_ops) == 12                 # every line shipped exactly once across batches
    assert len(pools_created) == 1               # ONE pool for the whole partition, not per batch


def test_ndjson_writer_reconciles_under_concurrency_with_errors_across_batches(monkeypatch):
    # End-to-end through make_ndjson_partition_writer + _merge_partition_results on the PIPELINED path:
    # multiple Arrow batches, write_concurrency>1, some docs rejected. Every input row must produce
    # exactly one outcome (written or error), so written+errors == total_input, unaccounted==0,
    # overcounted==0, and the rejected doc is sampled. This is the no-loss + correct-reporting-under-
    # failure guarantee on the concurrent, cross-batch write path.
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import threading
    from databricks_es_connector.bulk import _merge_partition_results

    class _SelectiveThreadSafeES:
        """400s any op line containing 'bad', 201s the rest. Thread-safe; closeable. Fast-path aware:
        a clean chunk's probe returns {"errors": False} (counted written, no re-ship); a chunk with a
        'bad' op flags errors on the probe and is re-shipped full for per-item classification."""
        def __init__(self):
            self._lock = threading.Lock()
            self.all_ops = []
            self.closed = False

        def bulk(self, operations=None, filter_path=None, **kw):
            ops = list(operations)
            has_bad = any("bad" in op for op in ops)
            if filter_path and "errors" in filter_path:
                if not has_bad:                      # clean chunk: shipped once via the probe
                    with self._lock:
                        self.all_ops.extend(ops)
                    return {"errors": False}
                return {"errors": True}              # bad present: fall through to the full re-ship
            with self._lock:                         # full re-ship: classify per item
                self.all_ops.extend(ops)
            return {"items": [{"index": {"status": 400, "_id": op, "error": {"reason": "boom"}}}
                              if "bad" in op else {"index": {"status": 201}} for op in ops]}

        def close(self):
            self.closed = True

    es = _SelectiveThreadSafeES()
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)
    # Auto-id (id_field=None) takes the fast path: clean chunks counted via the probe, chunks with a
    # 'bad' row re-shipped full and classified (the 'bad' row is a 400 error). 3 batches, each mixing
    # good and bad rows, chunk_size=2 so sends straddle batch boundaries under write_concurrency=3.
    writer = make_ndjson_partition_writer(_cfg(id_field=None, write_concurrency=3, chunk_size=2))
    batches = [pd.DataFrame({"_ndjson": ["ok0", "ok1", "bad0", "ok2", "ok3"]}),
               pd.DataFrame({"_ndjson": ["ok4", "bad1", "ok5"]}),
               pd.DataFrame({"_ndjson": ["ok6", "ok7", "ok8", "bad2"]})]
    out = list(writer(iter(batches)))
    row = out[0].iloc[0]
    assert int(row["total_input"]) == 12
    assert int(row["written"]) == 9 and int(row["errors"]) == 3
    # No row lost or double-counted below the per-doc level, across the concurrent cross-batch merge.
    result = _merge_partition_results([row])
    assert result["unaccounted"] == 0 and result["overcounted"] == 0
    assert result["written"] + result["errors"] == result["total_input"] == 12
    assert len(result["error_samples"]) == 3 and all("boom" in s["reason"] for s in result["error_samples"])
    assert es.closed is True


def test_ndjson_writer_null_line_in_later_batch_still_raises(monkeypatch):
    # Cross-batch null-id guard: batch 0 ships fine (its sends may still be in flight), then batch 1
    # carries a null action line (build_ndjson's null/non-finite-id signal). The per-batch guard runs
    # BEFORE each batch is fed, so the writer still RAISES and fails the partition -- pipelining must
    # not let a later null id slip through as `unaccounted`.
    pd = pytest.importorskip("pandas")
    import elasticsearch
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: _ThreadSafeFakeES())
    writer = make_ndjson_partition_writer(_cfg(id_field=None, write_concurrency=3, chunk_size=2))
    batches = [pd.DataFrame({"_ndjson": [f"b0_{i}" for i in range(6)]}),
               pd.DataFrame({"_ndjson": ["b1_ok", None]})]
    with pytest.raises(ValueError, match="null action line"):
        list(writer(iter(batches)))


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

def test_config_accepts_deletes():
    # The write path builds delete actions; the config accepts has_deletes + a flag column. The
    # remaining requirement -- the flag column must be boolean -- needs the DataFrame schema and is
    # enforced in bulk._preflight, not here.
    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id",
                   has_deletes=True, delete_flag_column="d")
    assert cfg.has_deletes is True and cfg.delete_flag_column == "d"


def test_config_write_concurrency_sizes_connection_pool():
    import warnings
    # write_concurrency drives the cross-batch pipeline (bulk._PipelinedShipper) and must not warn on
    # its own; the per-node connection pool is sized to it so the in-flight sends are not capped below
    # the configured value.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", write_concurrency=4)
    assert cfg.write_concurrency == 4
    assert cfg.client_kwargs().get("connections_per_node") == 4


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


# --- _preflight: deletes require a BooleanType flag column ------------------------------------
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


def test_preflight_rejects_non_boolean_delete_flag():
    # flag column "d" is a STRING, not boolean: `flag === true` would be null for every row => no row
    # routed to a delete => every intended deletion silently upserted. Must fail closed on the driver.
    cfg = _cfg(has_deletes=True, delete_flag_column="d")   # _cfg sets serialize_in_spark=True, id_field="id"
    with pytest.raises(ValueError, match="must be a boolean column"):
        _preflight(_fake_df([("id", "string"), ("d", "string")]), cfg)


def test_preflight_accepts_boolean_delete_flag():
    cfg = _cfg(has_deletes=True, delete_flag_column="d")
    _preflight(_fake_df([("id", "string"), ("d", "boolean")]), cfg)   # must not raise (require_existing_index=False)


# --- _PipelinedShipper: the write_concurrency cross-batch pipeline over pre-built lines ------------
# No Spark: _PipelinedShipper takes an ES client + a caller-owned pool, and `feed(lines)` per batch
# then `close()` exercises the pipeline exactly as make_ndjson_partition_writer drives it. The live
# end-to-end proof is integration test_concurrency_roundtrip.

class _ThreadSafeFakeES:
    """Models a clean ES on the fast path: a filter_path="errors" probe returns {"errors": False}
    (so the chunk is counted written with no re-ship) and a full call returns per-item 201s. Records
    what it shipped, once per line (a clean probe IS the real send; only the response is trimmed).
    Thread-safe for fan-out tests."""
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.chunks = []       # each es.bulk() call's operations list
        self.all_ops = []      # every op line shipped, flattened

    def bulk(self, operations=None, filter_path=None, **kw):
        ops = list(operations)
        with self._lock:
            self.chunks.append(ops)
            self.all_ops.extend(ops)
        if filter_path and "errors" in filter_path:      # clean probe: no per-item array, no re-ship
            return {"errors": False}
        return {"items": [{"index": {"status": 201}} for _ in ops]}


def _zero_counts():
    return {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}


def _run_shipper(es, batches, cfg, stats=None, max_inflight=None):
    """Feed `batches` (a list of line-lists, one per Arrow batch) through a _PipelinedShipper on a real
    pool, close, and return (counts, samples). Mirrors how make_ndjson_partition_writer drives it:
    feed each batch WITHOUT draining to zero, so in-flight sends carry across batch boundaries."""
    from concurrent.futures import ThreadPoolExecutor
    from databricks_es_connector.bulk import _PipelinedShipper
    counts, samples = _zero_counts(), []
    with ThreadPoolExecutor(max_workers=cfg.write_concurrency) as pool:
        shipper = _PipelinedShipper(es, cfg, pool, counts, samples, stats=stats, max_inflight=max_inflight)
        for b in batches:
            shipper.feed(b)
        shipper.close()
    return counts, samples


def test_pipelined_shipper_ships_all_lines_exactly_once():
    es = _ThreadSafeFakeES()
    lines = [f"L{i}" for i in range(10)]
    counts, _ = _run_shipper(es, [lines], _cfg(id_field=None, write_concurrency=3, chunk_size=2))
    # every line shipped exactly once across the in-flight sends -- no drops, no duplicates
    assert sorted(es.all_ops) == sorted(lines) and len(es.all_ops) == 10
    assert counts["written"] == 10 and counts["errors"] == 0


def test_pipelined_shipper_concurrency_matches_serial_tally():
    # Identical accounting and set of shipped ops regardless of write_concurrency (1 == 4).
    lines = [f"L{i}" for i in range(7)]

    def run(wc):
        es = _ThreadSafeFakeES()
        counts, _ = _run_shipper(es, [lines], _cfg(id_field=None, write_concurrency=wc, chunk_size=2))
        return counts, sorted(es.all_ops)

    serial_counts, serial_ops = run(1)
    conc_counts, conc_ops = run(4)
    assert serial_counts == conc_counts == {"written": 7, "deleted": 0, "ignored": 0, "errors": 0}
    assert serial_ops == conc_ops == sorted(lines)


def test_pipelined_shipper_carries_across_batches_no_loss():
    # The core new guarantee: sends carry across Arrow-batch boundaries (no per-batch join), and EVERY
    # line of EVERY batch is shipped exactly once with correct counts -- no loss, no dupe across the
    # boundary. Three batches of differing, chunk-straddling sizes.
    es = _ThreadSafeFakeES()
    batches = [[f"b0_{i}" for i in range(5)],
               [f"b1_{i}" for i in range(4)],
               [f"b2_{i}" for i in range(3)]]
    all_lines = [x for b in batches for x in b]
    counts, _ = _run_shipper(es, batches, _cfg(id_field=None, write_concurrency=3, chunk_size=2))
    assert sorted(es.all_ops) == sorted(all_lines) and len(es.all_ops) == 12
    assert counts == {"written": 12, "deleted": 0, "ignored": 0, "errors": 0}


def test_pipelined_shipper_merges_errors_and_samples_across_sends():
    # Per-send error tallies and (bounded) sample lists must merge correctly as sends complete.
    class _SelectiveFakeES:
        """400s any op line containing 'bad', 201s the rest."""
        def bulk(self, operations=None, **kw):
            items = []
            for op in operations:
                if "bad" in op:
                    items.append({"index": {"status": 400, "_id": op, "error": {"reason": "boom"}}})
                else:
                    items.append({"index": {"status": 201}})
            return {"items": items}

    lines = [f"ok{i}" for i in range(8)] + [f"bad{i}" for i in range(3)]
    counts, samples = _run_shipper(_SelectiveFakeES(), [lines], _cfg(write_concurrency=3, chunk_size=2))
    assert counts["written"] == 8 and counts["errors"] == 3
    assert len(samples) == 3 and all("boom" in s["reason"] for s in samples)


def test_pipelined_shipper_worker_exception_fails_closed(monkeypatch):
    # RED-BEFORE-GREEN guard: a worker exception must propagate (via f.result() in _drain/close), so a
    # partial write FAILS the partition rather than reporting the docs a dead worker never sent as a
    # clean count. Deleting the `f.result()` re-raise (using e.g. `f.done()` without reading the
    # result) makes this pass silently. _ship_ndjson_chunk normally catches transport errors and
    # counts them, so force a raw raise to exercise the re-raise guard itself.
    from databricks_es_connector import bulk as bulk_mod

    def _boom(es, chunk, cfg, counts, samples, stats=None, diag=None):
        raise RuntimeError("worker died mid-ship")

    monkeypatch.setattr(bulk_mod, "_ship_ndjson_chunk", _boom)
    with pytest.raises(RuntimeError, match="worker died"):
        _run_shipper(_ThreadSafeFakeES(), [[f"L{i}" for i in range(10)]],
                     _cfg(write_concurrency=3, chunk_size=2))


def test_pipelined_shipper_bounds_inflight_and_pipelines():
    # Two properties in one deterministic check, with sends held on a gate so they pile up:
    #   (1) BACKPRESSURE: at most write_concurrency sends are ever OUTSTANDING (submitted) at once, so
    #       peak memory is bounded to write_concurrency chunks, not the whole partition. A counting
    #       pool proves submissions stall at the cap (the pool's own max_workers cap would hide this,
    #       so we count submit() calls, not running threads).
    #   (2) PIPELINING: it actually REACHES write_concurrency in flight (not accidental serialization).
    import threading, time
    from concurrent.futures import ThreadPoolExecutor
    from databricks_es_connector.bulk import _PipelinedShipper

    WC = 3
    gate = threading.Event()
    lock = threading.Lock()
    st = {"in_flight": 0, "max_in_flight": 0, "submitted": 0}

    class _BlockingES:
        def bulk(self, operations=None, **kw):
            with lock:
                st["in_flight"] += 1
                st["max_in_flight"] = max(st["max_in_flight"], st["in_flight"])
            gate.wait(5)
            with lock:
                st["in_flight"] -= 1
            return {"errors": False}

    class _CountingPool:
        def __init__(self, real): self._real = real
        def submit(self, fn, *a, **k):
            with lock: st["submitted"] += 1
            return self._real.submit(fn, *a, **k)

    lines = [f"L{i}" for i in range(30)]           # 30 chunks at chunk_size=1, far more than WC
    cfg = _cfg(write_concurrency=WC, chunk_size=1)
    counts, samples = _zero_counts(), []
    real_pool = ThreadPoolExecutor(max_workers=WC)
    try:
        shipper = _PipelinedShipper(_BlockingES(), cfg, _CountingPool(real_pool), counts, samples)
        done = threading.Event()

        def _producer():
            for i in range(0, len(lines), 5):
                shipper.feed(lines[i:i + 5])
            shipper.close()
            done.set()

        t = threading.Thread(target=_producer)
        t.start()
        # Wait until WC sends are in flight (all blocked on the gate), then confirm it stalls there.
        deadline = time.time() + 5
        while st["in_flight"] < WC and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)                            # give any (buggy) extra submissions time to appear
        with lock:
            assert st["max_in_flight"] == WC, st   # (2) pipelines to write_concurrency
            assert st["in_flight"] == WC, st       # still exactly WC in flight, no more
            assert st["submitted"] == WC, st       # (1) backpressure: only WC submitted, rest blocked
        assert not done.is_set()                   # producer is blocked at the cap, not finished
        gate.set()                                 # release every send
        t.join(10)
        assert done.is_set()
    finally:
        gate.set()
        real_pool.shutdown(wait=True)
    assert counts["written"] == 30 and st["submitted"] == 30   # after release, all 30 ship


def test_pipelined_shipper_straggler_does_not_freeze_partition():
    # The barrier this change removes: one slow send must NOT stall the partition. A single straggler
    # blocks on a gate while many fast sends keep completing; assert the fast sends all finish WHILE
    # the straggler is still in flight (the old per-batch join would have frozen everything on it).
    import threading, time
    from concurrent.futures import ThreadPoolExecutor
    from databricks_es_connector.bulk import _PipelinedShipper

    WC = 3
    gate = threading.Event()
    straggler_running = threading.Event()
    lock = threading.Lock()
    shipped = []

    class _StragglerES:
        def bulk(self, operations=None, **kw):
            (op,) = operations                     # chunk_size=1 -> one op per send
            if op == "SLOW":
                straggler_running.set()
                gate.wait(5)                       # hold the slot until released
            with lock:
                shipped.append(op)
            return {"errors": False}

    lines = ["SLOW"] + [f"F{i}" for i in range(20)]
    cfg = _cfg(write_concurrency=WC, chunk_size=1)
    counts, samples = _zero_counts(), []
    real_pool = ThreadPoolExecutor(max_workers=WC)
    try:
        shipper = _PipelinedShipper(_StragglerES(), cfg, real_pool, counts, samples)
        done = threading.Event()

        def _producer():
            shipper.feed(lines)
            shipper.close()
            done.set()

        t = threading.Thread(target=_producer)
        t.start()
        assert straggler_running.wait(5)           # straggler is in flight, holding one slot
        # The 20 fast sends must all complete while SLOW is still blocked -> progress despite a straggler.
        deadline = time.time() + 5
        while len(shipped) < 20 and time.time() < deadline:
            time.sleep(0.01)
        with lock:
            assert len(shipped) == 20, shipped     # every fast send finished...
            assert "SLOW" not in shipped           # ...while the straggler is still in flight
        assert not done.is_set()                   # close() is still waiting on the straggler
        gate.set()                                 # release the straggler
        t.join(10)
        assert done.is_set()
    finally:
        gate.set()
        real_pool.shutdown(wait=True)
    assert counts["written"] == 21 and "SLOW" in shipped


def test_config_write_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="write_concurrency must be >= 1"):
        EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", write_concurrency=0)


# --- _ship_ndjson_chunk: the filter_path="errors" fast path (GIL avoidance) --------------------
# On any delete-free write, a clean bulk needs only the top-level `errors` flag, so the chunk is
# shipped with filter_path="errors" and the per-item response is never decoded or classified in Python
# (the GIL-held cost that serialized write_concurrency threads). On ANY failure the chunk is re-shipped
# with a FULL response and the existing classify + 429-retry runs. The re-ship is an idempotent upsert
# when id_field is set; with auto-generated ids it re-creates (DUPLICATES) the probe's writes, which is
# accepted since an auto-id write is already at-least-once. The gate is exactly no-deletes (a delete-404
# is IGNORED, so `errors: false` cannot be read as "all written"; delete writes take the full path).

class _RecordingES:
    """Records each call's (operations, filter_path) and returns queued responses in order."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []          # list of (operations_list, filter_path)

    def bulk(self, operations=None, filter_path=None, **kw):
        self.calls.append((list(operations), filter_path))
        return self._responses.pop(0)


def test_fast_path_clean_bulk_requests_only_errors_and_counts_all_written():
    # Happy path: ONE minimal request (filter_path="errors"), response carries no items, every line
    # counted written without decoding/classifying a per-item array. No re-ship.
    es = _RecordingES([{"errors": False}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(), counts, [])
    assert counts == {"written": 3, "deleted": 0, "ignored": 0, "errors": 0}
    assert es.calls == [(["a", "b", "c"], "errors")]      # minimal request, no full re-ship


def test_fast_path_reissues_full_on_error_and_classifies_per_item():
    # errors=True on the minimal probe -> re-ship FULL (no filter_path) -> existing per-item classify.
    full = {"items": [{"index": {"status": 201, "_id": "a"}},
                      {"index": {"status": 400, "_id": "b", "error": {"reason": "boom"}}}]}
    es = _RecordingES([{"errors": True}, full])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, samples)
    assert counts == {"written": 1, "deleted": 0, "ignored": 0, "errors": 1}
    assert samples and samples[0]["_id"] == "b" and "boom" in samples[0]["reason"]
    assert es.calls[0] == (["a", "b"], "errors")          # probe first
    assert es.calls[1] == (["a", "b"], _FULL_FP)              # then full re-ship for detail


def test_fast_path_reissue_still_drives_429_retry():
    # After the probe flags an error, the full re-ship must drive the normal per-doc 429 retry.
    es = _RecordingES([
        {"errors": True},
        {"items": [{"index": {"status": 201}}, {"index": {"status": 429}}, {"index": {"status": 201}}]},
        {"items": [{"index": {"status": 201}}]},
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(max_retries_per_doc=3), counts, [])
    assert counts["written"] == 3 and counts["errors"] == 0
    assert es.calls[1] == (["a", "b", "c"], _FULL_FP)         # full re-ship
    assert es.calls[2] == (["b"], _FULL_FP)                   # only the 429 line retried


def test_fast_path_missing_errors_key_fails_closed():
    # A probe response without an `errors` key must NOT read as clean: treat as error and re-ship full.
    es = _RecordingES([{}, {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, [])
    assert counts["written"] == 2 and counts["errors"] == 0
    assert len(es.calls) == 2 and es.calls[1] == (["a", "b"], _FULL_FP)


def test_fast_path_applies_without_id_field():
    # Auto-id (no id_field) is delete-free, so it takes the fast path too: a clean chunk is ONE minimal
    # probe (filter_path="errors"), every line counted written, no per-item decode and no re-ship. (The
    # error branch may re-ship and duplicate the probe's writes; that is tested separately and accepted
    # -- an auto-id write is already at-least-once.)
    es = _RecordingES([{"errors": False}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, [])
    assert counts == {"written": 2, "deleted": 0, "ignored": 0, "errors": 0}
    assert es.calls == [(["a", "b"], "errors")]           # minimal probe, no full re-ship


def test_fast_path_without_id_field_reships_full_on_error():
    # Auto-id error branch: errors=True on the probe -> re-ship FULL (no filter_path) for per-item
    # detail, exactly like the id_field path. The re-ship DUPLICATES the docs the probe already wrote
    # (ES assigns fresh ids), which is accepted for auto-id (already at-least-once); the written count
    # reflects the re-ship so reconciliation stays consistent.
    full = {"items": [{"index": {"status": 201, "_id": "auto1"}},
                      {"index": {"status": 400, "_id": "auto2", "error": {"reason": "boom"}}}]}
    es = _RecordingES([{"errors": True}, full])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, samples)
    assert counts == {"written": 1, "deleted": 0, "ignored": 0, "errors": 1}
    assert samples and "boom" in samples[0]["reason"]
    assert es.calls[0] == (["a", "b"], "errors")          # probe first
    assert es.calls[1] == (["a", "b"], _FULL_FP)              # then full re-ship for detail


def test_fast_path_without_id_field_transient_429_reships_whole_chunk_then_retries_line():
    # Documented regression (accepted): a single retryable 429 flips the probe's `errors` flag, so the
    # WHOLE auto-id chunk re-ships -- re-sending (and thus duplicating in ES) the good docs too, not just
    # the 429'd line -- and only then does the per-doc loop retry the 429'd line. The old auto-id path
    # took the full path directly and retried only that line with no whole-chunk re-ship. The assertion
    # that es.calls[1] carries ALL three lines (not just "b") is the duplication: "a" and "c" succeeded
    # on the probe and are re-sent on the re-ship.
    es = _RecordingES([
        {"errors": True},                                                              # probe: a 429 is present
        {"items": [{"index": {"status": 201}}, {"index": {"status": 429}}, {"index": {"status": 201}}]},
        {"items": [{"index": {"status": 201}}]},                                       # retry of "b" succeeds
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(id_field=None, max_retries_per_doc=3), counts, [])
    assert counts["written"] == 3 and counts["errors"] == 0
    assert es.calls[0] == (["a", "b", "c"], "errors")     # probe
    assert es.calls[1] == (["a", "b", "c"], _FULL_FP)         # whole chunk re-shipped (a + c duplicated)
    assert es.calls[2] == (["b"], _FULL_FP)                   # only the 429'd line retried after that


def test_fast_path_disabled_with_deletes():
    # With deletes, errors=False does NOT mean "all written" (some are deleted / delete-404 ignored),
    # so the fast path must not apply: ship full and classify so deleted/ignored split out correctly.
    from databricks_es_connector.config import EsWriteConfig
    cfg = EsWriteConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id",
                        require_existing_index=False, has_deletes=True, delete_flag_column="d")
    es = _RecordingES([{"items": [{"index": {"status": 201}}, {"delete": {"status": 404}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], cfg, counts, [])
    assert counts == {"written": 1, "deleted": 0, "ignored": 1, "errors": 0}
    assert es.calls == [(["a", "b"], _FULL_FP)]


class _NoGetResponse:
    """Mimics elasticsearch-py 8.x's ObjectApiResponse: supports resp["k"] and "k" in resp, but has
    NO .get method (a plain-dict assumption would AttributeError on a live cluster)."""
    def __init__(self, body): self._body = dict(body)
    def __getitem__(self, k): return self._body[k]
    def __contains__(self, k): return k in self._body


class _ObjResponseFakeES:
    """Fake whose bulk() returns an ObjectApiResponse-like object (no .get), one per queued body."""
    def __init__(self, bodies):
        self._bodies = list(bodies)
        self.calls = []
    def bulk(self, operations=None, filter_path=None, **kw):
        self.calls.append((list(operations), filter_path))
        return _NoGetResponse(self._bodies.pop(0))


def test_fast_path_reads_errors_from_objectapiresponse_without_get():
    # Real ES 8.x returns an ObjectApiResponse (indexing + `in`, no .get). The clean-probe flag read
    # must not assume a plain dict, or it AttributeErrors on the exact write this path targets.
    es = _ObjResponseFakeES([{"errors": False}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, [])
    assert counts["written"] == 2 and es.calls == [(["a", "b"], "errors")]


def test_fast_path_objectapiresponse_error_reships_and_classifies():
    # Same non-dict response type on the error path: errors=True -> full re-ship -> per-item classify.
    es = _ObjResponseFakeES([
        {"errors": True},
        {"items": [{"index": {"status": 201, "_id": "a"}},
                   {"index": {"status": 400, "_id": "b", "error": {"reason": "boom"}}}]},
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, samples)
    assert counts == {"written": 1, "deleted": 0, "ignored": 0, "errors": 1}
    assert samples and samples[0]["_id"] == "b"
    assert es.calls[1] == (["a", "b"], _FULL_FP)


def test_fast_path_probe_transport_error_counts_errors_not_crash():
    # The minimal probe raising (transport failure) must count every line as an error, not crash.
    class _RaisingES:
        def __init__(self): self.calls = []
        def bulk(self, operations=None, filter_path=None, **kw):
            self.calls.append(filter_path)
            raise RuntimeError("connection reset")
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(_RaisingES(), ["a", "b", "c"], _cfg(), counts, samples)
    assert counts["errors"] == 3 and samples


# --- bulk_stats: per-partition send aggregates (docs/send, rtt vs ES took, socket-vs-GIL wait) ---
# Off by default and zero-overhead. When on, each es.bulk send is timed and (docs, bytes, rtt_ms,
# took_ms, cpu_ms) recorded, and a background _GilWaitProbe samples GIL-acquisition latency; the writer
# aggregates per partition and _merge_partition_results surfaces one entry per partition under
# result["bulk_stats"]. `took` is requested via filter_path so it is available without the per-item
# array (the GIL win is preserved). cpu_ms (worker CPU inside es.bulk) and the gil_wait_ms columns
# together split a slow rtt into a real socket/ES wait versus the worker being starved of the GIL.

def test_config_bulk_stats_default_off():
    assert _cfg().bulk_stats is False


def test_percentile_linear_interpolation():
    from databricks_es_connector.bulk import _percentile
    assert _percentile([], 50) is None
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert _percentile([10.0, 20.0, 30.0], 95) == 29.0     # 0.95*2=1.9 -> 20*.1+30*.9
    assert _percentile([10.0, 20.0, 30.0], 0) == 10.0
    assert _percentile([10.0, 20.0, 30.0], 100) == 30.0


def test_aggregate_bulk_stats_computes_send_and_latency_summary():
    from databricks_es_connector.bulk import _aggregate_bulk_stats
    # (docs, bytes, rtt_ms, took_ms, cpu_ms, http_ms, outcome). All three succeeded ("ok"); one has a
    # None took (ES omitted it) -> excluded from the took aggregates only. Nested: rtt >= http >= took.
    agg = _aggregate_bulk_stats([(100, 1000, 10.0, 5.0, 1.0, 8.0, "ok"),
                                 (100, 2000, 20.0, 15.0, 2.0, 18.0, "ok"),
                                 (50, 500, 30.0, None, 3.0, 28.0, "ok")])
    assert agg["n_sends"] == [3]
    assert agg["docs_sent"] == [250]           # retries would count again; here 3 distinct sends
    assert agg["bytes_sent"] == [3500]         # 1000 + 2000 + 500 (uncompressed NDJSON)
    assert agg["send_busy_ms"] == [60.0]       # 10 + 20 + 30 (summed round trip; vs wall => concurrency)
    assert agg["send_cpu_ms"] == [6.0]         # 1 + 2 + 3 (summed worker CPU inside es.bulk)
    assert agg["rtt_ms_mean"] == [20.0] and agg["rtt_ms_p50"] == [20.0]
    assert agg["rtt_ms_p95"] == [29.0] and agg["rtt_ms_max"] == [30.0]
    assert agg["http_ms_mean"] == [18.0] and agg["http_ms_max"] == [28.0]   # node HTTP wall (rtt>=http>=took)
    assert agg["took_ms_mean"] == [10.0]       # (5+15)/2, None excluded
    assert agg["took_ms_p50"] == [10.0] and agg["took_ms_max"] == [15.0]
    assert agg["timeout_sends"] == [0] and agg["timeout_wait_ms"] == [0.0]  # no failed sends here
    assert agg["error_sends"] == [0] and agg["error_wait_ms"] == [0.0]


def test_aggregate_bulk_stats_splits_ok_timeout_and_error_sends():
    # Failed sends (took/http None) are summarized SEPARATELY so their wall cost is visible; the ok
    # sends alone drive docs/rtt/http/took. total es.bulk time = send_busy + timeout_wait + error_wait.
    from databricks_es_connector.bulk import _aggregate_bulk_stats
    agg = _aggregate_bulk_stats([
        (100, 1000, 12.0, 5.0, 1.0, 10.0, "ok"),
        (100, 1000, 300000.0, None, 0.0, None, "timeout"),   # two ~300s timed-out attempts
        (100, 1000, 300000.0, None, 0.0, None, "timeout"),
        (50, 500, 40.0, None, 0.0, None, "error"),           # a non-timeout transport failure
    ])
    assert agg["n_sends"] == [1]                 # only the successful send
    assert agg["docs_sent"] == [100] and agg["bytes_sent"] == [1000]
    assert agg["send_busy_ms"] == [12.0]         # ok rtt only
    assert agg["rtt_ms_max"] == [12.0]           # timed-out attempts are NOT in the rtt distribution
    assert agg["timeout_sends"] == [2] and agg["timeout_wait_ms"] == [600000.0]
    assert agg["error_sends"] == [1] and agg["error_wait_ms"] == [40.0]


def test_aggregate_bulk_stats_empty_partition_is_nulls_not_crash():
    from databricks_es_connector.bulk import _aggregate_bulk_stats
    agg = _aggregate_bulk_stats([])
    assert agg["n_sends"] == [0] and agg["docs_sent"] == [0]
    assert agg["bytes_sent"] == [0] and agg["send_busy_ms"] == [0.0]
    assert agg["send_cpu_ms"] == [0.0]
    assert agg["rtt_ms_mean"] == [None] and agg["rtt_ms_max"] == [None]
    assert agg["http_ms_mean"] == [None] and agg["http_ms_max"] == [None]
    assert agg["took_ms_p95"] == [None]
    assert agg["timeout_sends"] == [0] and agg["timeout_wait_ms"] == [0.0]
    assert agg["error_sends"] == [0] and agg["error_wait_ms"] == [0.0]


def test_aggregate_gil_wait_summary_and_empty():
    from databricks_es_connector.bulk import _aggregate_gil_wait
    agg = _aggregate_gil_wait([10.0, 20.0, 30.0])
    assert agg["gil_wait_ms_total"] == [60.0]
    assert agg["gil_wait_ms_p50"] == [20.0] and agg["gil_wait_ms_p95"] == [29.0]
    assert agg["gil_wait_ms_max"] == [30.0] and agg["gil_wait_samples"] == [3]
    # No contention observed (e.g. write_concurrency == 1): total/count zero, percentiles null, no crash.
    empty = _aggregate_gil_wait([])
    assert empty["gil_wait_ms_total"] == [0.0] and empty["gil_wait_samples"] == [0]
    assert empty["gil_wait_ms_p95"] == [None] and empty["gil_wait_ms_max"] == [None]


def test_gil_wait_probe_fires_under_contention():
    # A monitor you have not watched fire is not a monitor: prove the probe actually RECORDS a stall
    # when the GIL is held. Raise the interpreter's thread-switch interval so a pure-Python busy loop on
    # this thread keeps the GIL for the whole burn without ever yielding; the probe thread then cannot
    # re-acquire the GIL to finish a wakeup, and the excess it records must reflect that long stall.
    import sys
    import time
    from databricks_es_connector.bulk import _GilWaitProbe
    old_interval = sys.getswitchinterval()
    probe = _GilWaitProbe().start()
    time.sleep(0.02)   # let the probe reach its wait() loop before we stop yielding the GIL
    sys.setswitchinterval(1.0)   # >> the burn, so the busy loop never voluntarily drops the GIL
    try:
        deadline = time.perf_counter() + 0.3
        x = 0
        while time.perf_counter() < deadline:   # pure-Python, GIL-holding; perf_counter does not yield it
            x += 1
    finally:
        sys.setswitchinterval(old_interval)
    time.sleep(0.05)   # release the GIL so the probe can record the stall it accumulated during the burn
    lags = probe.stop()
    assert lags, "probe recorded no GIL-wait samples at all"
    assert max(lags) > 80.0, f"expected a large stall from the ~0.3s GIL burn, got max {max(lags)}ms"


def test_gil_wait_probe_stop_returns_independent_snapshot():
    # stop() must return a STABLE snapshot, not the live list: if the daemon outlived a timed-out join
    # (the exception-path hazard), the aggregate would otherwise sort/sum a list still being appended to.
    # A post-stop mutation of the probe's internal list must NOT change what stop() already returned.
    from databricks_es_connector.bulk import _GilWaitProbe
    probe = _GilWaitProbe().start()
    snap = probe.stop()
    assert isinstance(snap, list)
    probe._lags_ms.append(999.0)          # simulate a straggler daemon appending after the join
    assert 999.0 not in snap              # the returned snapshot is decoupled from the live list


def test_gil_wait_probe_quiet_when_idle():
    # With no thread hogging the GIL (the main thread is blocked in sleep, which RELEASES the GIL), the
    # probe wakes on time and records at most minor scheduler jitter -- never a large stall. This is the
    # write_concurrency==1 / uncontended shape, where gil_wait must read ~0.
    import time
    from databricks_es_connector.bulk import _GilWaitProbe
    probe = _GilWaitProbe().start()
    time.sleep(0.2)
    lags = probe.stop()
    assert not lags or max(lags) < 40.0, f"unexpected large GIL stall while idle: max {max(lags)}ms"


def test_ship_chunk_records_a_send_on_the_fast_path():
    # stats list given -> the clean fast-path send is timed and recorded, and `took` is requested.
    es = _RecordingES([{"errors": False, "took": 7}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, stats=stats, error_samples=[])
    assert counts["written"] == 2
    assert es.calls == [(["a", "b"], "errors,took")]    # took requested, items still omitted
    assert len(stats) == 1
    docs, byts, rtt_ms, took_ms, cpu_ms, http_ms, outcome = stats[0]
    assert docs == 2 and took_ms == 7 and rtt_ms >= 0.0
    assert byts == 2          # sum of len("a") + len("b") = uncompressed NDJSON size
    assert cpu_ms >= 0.0      # worker-thread CPU inside es.bulk (thread_time delta), never negative
    assert outcome == "ok"    # a returned response; a raised send would be "timeout"/"error"


def test_ship_chunk_no_stats_and_no_took_when_disabled():
    # stats=None (default) -> no recording, and the fast path requests only "errors" (no took).
    es = _RecordingES([{"errors": False}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, [])
    assert counts["written"] == 2 and es.calls == [(["a", "b"], "errors")]


def test_ship_chunk_records_took_on_full_path():
    # The full (re-ship) path records its send too, reading took from the full response. Reached here
    # via a probe error (id_field=None): both the probe and the re-ship are recorded, and the re-ship
    # (the full response) carries took=3.
    es = _RecordingES([
        {"errors": True},                          # probe: no took -> None recorded
        {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}], "took": 3},
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, stats=stats, error_samples=[])
    assert counts["written"] == 2 and len(stats) == 2
    # re-ship send is (docs, bytes, rtt, took, cpu, http, outcome): took at idx 3, cpu at idx 4.
    assert stats[-1][0] == 2 and stats[-1][3] == 3 and stats[-1][4] >= 0.0 and stats[-1][6] == "ok"


def test_pipelined_shipper_merges_stats_across_sends():
    es = _FastFakeES()   # returns {"errors": False} on the probe; no took, so took_ms is None
    lines = [f"L{i}" for i in range(10)]
    stats = []
    _run_shipper(es, [lines], _cfg(write_concurrency=3, chunk_size=2), stats=stats)
    # Per-send size is now chunk_size alone (decoupled from write_concurrency): 10 lines / chunk_size 2
    # = 5 sends; every line shipped exactly once (docs sum to 10), and every send is recorded.
    assert len(stats) == 5 and sum(s[0] for s in stats) == 10
    assert all(s[2] >= 0.0 for s in stats)          # rtt is index 2 in (docs, bytes, rtt, took)
    assert sum(s[1] for s in stats) == sum(len(x) for x in lines)   # bytes cover every line once


def test_writer_emits_per_partition_bulk_stats_columns(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    es = _FastFakeES()
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)
    writer = make_ndjson_partition_writer(_cfg(bulk_stats=True, chunk_size=2))
    out = list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b", "c", "d", "e"]})])))
    row = out[0].iloc[0]
    assert "n_sends" in out[0].columns and "rtt_ms_p95" in out[0].columns
    assert "bytes_sent" in out[0].columns and "send_busy_ms" in out[0].columns
    assert "partition_wall_ms" in out[0].columns
    # The socket-vs-GIL diagnostic columns must be emitted too.
    assert "send_cpu_ms" in out[0].columns
    assert {"gil_wait_ms_total", "gil_wait_ms_p95", "gil_wait_ms_max",
            "gil_wait_samples"} <= set(out[0].columns)
    assert int(row["n_sends"]) == 3 and int(row["docs_sent"]) == 5   # chunks [2,2,1]
    assert int(row["bytes_sent"]) == 5              # len("a".."e") = 5 single-char lines
    assert float(row["rtt_ms_max"]) >= 0.0
    assert float(row["send_busy_ms"]) >= 0.0 and float(row["partition_wall_ms"]) >= 0.0
    assert float(row["send_cpu_ms"]) >= 0.0
    assert float(row["gil_wait_ms_total"]) >= 0.0 and int(row["gil_wait_samples"]) >= 0


def test_writer_closes_es_client(monkeypatch):
    # The per-partition client must be closed at task end so its keep-alive connections are released
    # promptly rather than left for GC (the write path previously leaked one client per partition).
    pd = pytest.importorskip("pandas")
    import elasticsearch
    es = _FastFakeES()
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)
    writer = make_ndjson_partition_writer(_cfg(chunk_size=2))
    list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b"]})])))
    assert es.closed is True


def test_writer_omits_stats_columns_when_off(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: _FastFakeES())
    writer = make_ndjson_partition_writer(_cfg(chunk_size=2))   # bulk_stats default off
    out = list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b", "c"]})])))
    assert "n_sends" not in out[0].columns


def test_merge_builds_per_partition_bulk_stats_list():
    from databricks_es_connector.bulk import _merge_partition_results
    rows = [
        {"written": 4, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
         "total_input": 4, "error_samples": "[]", "n_sends": 2, "docs_sent": 4,
         "bytes_sent": 4000, "send_busy_ms": 24.0, "partition_wall_ms": 30.0,
         "rtt_ms_mean": 12.0, "rtt_ms_p50": 12.0, "rtt_ms_p95": 18.0, "rtt_ms_max": 20.0,
         "took_ms_mean": 4.0, "took_ms_p50": 4.0, "took_ms_p95": 6.0, "took_ms_max": 7.0},
        {"written": 6, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
         "total_input": 6, "error_samples": "[]", "n_sends": 3, "docs_sent": 6,
         "bytes_sent": 6000, "send_busy_ms": 27.0, "partition_wall_ms": 30.0,
         "rtt_ms_mean": 9.0, "rtt_ms_p50": 9.0, "rtt_ms_p95": 11.0, "rtt_ms_max": 12.0,
         "took_ms_mean": 3.0, "took_ms_p50": 3.0, "took_ms_p95": 4.0, "took_ms_max": 5.0},
    ]
    result = _merge_partition_results(rows)
    assert result["written"] == 10
    assert "bulk_stats" in result and len(result["bulk_stats"]) == 2
    assert result["bulk_stats"][0]["n_sends"] == 2 and result["bulk_stats"][1]["rtt_ms_max"] == 12.0
    assert result["bulk_stats"][0]["bytes_sent"] == 4000
    assert result["bulk_stats"][1]["send_busy_ms"] == 27.0 and result["bulk_stats"][0]["partition_wall_ms"] == 30.0


def test_merge_omits_bulk_stats_when_rows_lack_it():
    # Backward compat: rows without stat columns must NOT add a bulk_stats key (core key set unchanged).
    from databricks_es_connector.bulk import _merge_partition_results
    rows = [{"written": 3, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
             "total_input": 3, "error_samples": "[]"}]
    result = _merge_partition_results(rows)
    assert "bulk_stats" not in result


class _FastFakeES:
    """Thread-safe fake for the fast path: filter_path='errors' returns {'errors': False} (probe),
    a full call returns per-item 201s. Records the (ops, filter_path) of every call."""
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.calls = []          # (operations, filter_path)
        self.all_ops = []
        self.closed = False

    def bulk(self, operations=None, filter_path=None, **kw):
        ops = list(operations)
        with self._lock:
            self.calls.append((ops, filter_path))
            self.all_ops.extend(ops)
        if filter_path and "errors" in filter_path:      # "errors" or "errors,took" (stats mode)
            return {"errors": False, "took": 1}
        return {"items": [{"index": {"status": 201}} for _ in ops], "took": 1}

    def close(self):
        self.closed = True


def test_fast_path_under_pipeline_ships_each_line_once_via_probe():
    # The pipeline (write_concurrency) composes with the fast path: every send probes with
    # filter_path="errors", every line ships exactly once, all counted written, no full re-ship.
    es = _FastFakeES()
    lines = [f"L{i}" for i in range(10)]
    counts, _ = _run_shipper(es, [lines], _cfg(write_concurrency=3, chunk_size=2))
    assert sorted(es.all_ops) == sorted(lines) and len(es.all_ops) == 10
    assert counts == {"written": 10, "deleted": 0, "ignored": 0, "errors": 0}
    assert all(fp == "errors" for _ops, fp in es.calls)   # every request was a minimal probe


def test_fast_path_through_the_writer_counts_all_written(monkeypatch):
    # End-to-end through make_ndjson_partition_writer: a clean idempotent batch is all written and
    # the yielded summary schema is unchanged.
    pd = pytest.importorskip("pandas")
    import elasticsearch
    es = _FastFakeES()
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)
    writer = make_ndjson_partition_writer(_cfg(chunk_size=2))
    out = list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b", "c", "d", "e"]})])))
    row = out[0].iloc[0]
    assert int(row["written"]) == 5 and int(row["total_input"]) == 5
    assert int(row["errors"]) == 0
    assert all(fp == "errors" for _ops, fp in es.calls)


# =====================================================================================
# retry_transport_timeout: connector-owned whole-request timeout retry
#
# With cfg.retry_transport_timeout on, the write client is built with retry_on_timeout=False (so the
# transport stops re-sending timed-out bulks silently) and _ship_ndjson_chunk re-sends a timed-out
# chunk -- ConnectionTimeout ONLY -- up to transport_max_retries times with the same bounded
# exponential backoff the per-doc loop uses, then falls closed into errors exactly as before. Off by
# default: a timeout is counted as errors on its first surfaced attempt (today's behavior; the
# transport owns its own retries). A timed-out re-send re-applies the chunk (idempotent with id_field;
# a duplicate with auto-ids, accepted at-least-once).
# =====================================================================================

from elasticsearch import ConnectionTimeout, ConnectionError as _EsConnectionError


class _TimeoutThenOkES:
    """Raises ConnectionTimeout on the first `fail_times` bulk() calls, then returns `ok_response`."""
    def __init__(self, fail_times, ok_response):
        self.fail_times = fail_times
        self.ok_response = ok_response
        self.calls = 0

    def bulk(self, operations=None, filter_path=None, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionTimeout("read timed out")
        return self.ok_response


class _AlwaysTimeoutES:
    def __init__(self):
        self.calls = 0

    def bulk(self, operations=None, filter_path=None, **kw):
        self.calls += 1
        raise ConnectionTimeout("read timed out")


@pytest.fixture
def _no_backoff(monkeypatch):
    # The backoff sleeps are negligible next to a real request_timeout, but must not slow the suite.
    import time
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def test_config_retry_transport_timeout_off_by_default():
    assert _cfg().retry_transport_timeout is False
    # Default keeps the transport's own timeout retry, so existing callers are unaffected.
    assert _cfg().client_kwargs()["retry_on_timeout"] is True


def test_retry_transport_timeout_disables_the_transport_timeout_retry():
    # On => the connector owns it, so the client must NOT also re-send timeouts (no stacking layers).
    assert _cfg(retry_transport_timeout=True).client_kwargs()["retry_on_timeout"] is False
    # It overrides an explicit retry_on_timeout=True: the higher-level switch wins, never silent stacking.
    both = _cfg(retry_transport_timeout=True, retry_on_timeout=True)
    assert both.client_kwargs()["retry_on_timeout"] is False


def test_timeout_not_retried_when_knob_off():
    # Guard: knob off (default) => a ConnectionTimeout is counted as errors on its first surfaced
    # attempt, NOT re-sent by the connector. Pins that the feature is genuinely opt-in.
    es = _AlwaysTimeoutES()
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, samples)
    assert es.calls == 1
    assert counts["errors"] == 2 and counts["written"] == 0
    assert "ConnectionTimeout" in samples[0]["reason"]


def test_timeout_retried_then_succeeds_when_knob_on(_no_backoff):
    # A ConnectionTimeout re-sends the whole chunk; a later clean response writes every line and the
    # chunk is NOT an error.
    es = _TimeoutThenOkES(fail_times=2, ok_response={"errors": False})
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b", "c"],
                       _cfg(id_field=None, retry_transport_timeout=True, transport_max_retries=3),
                       counts, [])
    assert es.calls == 3                        # 1 initial + 2 retries, then success
    assert counts == {"written": 3, "deleted": 0, "ignored": 0, "errors": 0}


def test_timeout_exhausts_budget_then_fails_closed(_no_backoff):
    # After transport_max_retries connector retries all time out, fall closed into errors (surfaced via
    # reconcile), rather than retrying forever or aborting the partition.
    es = _AlwaysTimeoutES()
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["a", "b"],
                       _cfg(id_field=None, retry_transport_timeout=True, transport_max_retries=2),
                       counts, samples)
    assert es.calls == 3                        # 1 initial + 2 retries, all timed out
    assert counts["errors"] == 2 and counts["written"] == 0
    assert "ConnectionTimeout" in samples[0]["reason"]


def test_timeout_retry_backoff_is_bounded_exponential(monkeypatch):
    # The backoff between connector retries is the same bounded exponential the per-doc loop uses:
    # min(2**attempt, 30) for attempts 1, 2, 3 => 2, 4, 8. No unbounded or absent backoff.
    slept = []
    import time
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    es = _AlwaysTimeoutES()
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a"],
                       _cfg(id_field=None, retry_transport_timeout=True, transport_max_retries=3),
                       counts, [])
    assert slept == [2, 4, 8]
    assert es.calls == 4                         # 1 initial + 3 retries before the budget is spent


def test_timeout_retry_covers_the_full_path_too(_no_backoff):
    # The retry wraps BOTH the fast-path probe send and the full-path send. A delete-bearing write
    # skips the fast path and goes straight to the full path, so a timeout there is retried the same.
    es = _TimeoutThenOkES(fail_times=1, ok_response={"items": [{"delete": {"status": 200}}]})
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    cfg = _cfg(retry_transport_timeout=True, transport_max_retries=2,
               has_deletes=True, delete_flag_column="d")
    _ship_ndjson_chunk(es, ["header-only-delete-line"], cfg, counts, [])
    assert es.calls == 2                          # 1 timeout + 1 success on the full path
    assert counts["deleted"] == 1 and counts["errors"] == 0


def test_non_timeout_transport_error_is_not_retried_even_when_knob_on(_no_backoff):
    # Allow-list, not deny-list: only ConnectionTimeout is retried. A sibling ConnectionError (which
    # the transport already retries and then surfaces) still fails closed on its first surfaced attempt,
    # so a persistent non-timeout failure can't spin in the connector.
    class _RaisingES:
        def __init__(self):
            self.calls = 0

        def bulk(self, operations=None, filter_path=None, **kw):
            self.calls += 1
            raise _EsConnectionError("connection reset")

    es = _RaisingES()
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["a", "b"],
                       _cfg(id_field=None, retry_transport_timeout=True, transport_max_retries=3),
                       counts, samples)
    assert es.calls == 1                          # NOT retried by the connector
    assert counts["errors"] == 2 and counts["written"] == 0
    assert "ConnectionError" in samples[0]["reason"]


def test_ship_chunk_records_timeout_attempts_in_stats(_no_backoff):
    # End-to-end: with bulk_stats collecting and retry_transport_timeout on, each timed-out attempt is
    # recorded (outcome "timeout") and the final success as "ok", so the retry's wall cost is visible
    # in bulk_stats rather than vanishing inside a hidden transport retry.
    from databricks_es_connector.bulk import _aggregate_bulk_stats
    es = _TimeoutThenOkES(fail_times=2, ok_response={"errors": False})
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_chunk(es, ["a", "b"],
                       _cfg(id_field=None, retry_transport_timeout=True, transport_max_retries=3,
                            bulk_stats=True),
                       counts, [], stats=stats)
    assert [s[6] for s in stats] == ["timeout", "timeout", "ok"]
    assert counts["written"] == 2
    agg = _aggregate_bulk_stats(stats)
    assert agg["timeout_sends"] == [2] and agg["n_sends"] == [1]
    assert agg["timeout_wait_ms"][0] > 0


def test_ship_chunk_records_http_ms_from_response_meta():
    # http_ms is read from resp.meta.duration (elastic_transport node HTTP wall) on a successful send,
    # sitting between rtt and took. A response with no .meta records http_ms None (no crash).
    from elastic_transport import ObjectApiResponse, ApiResponseMeta, HttpHeaders
    meta = ApiResponseMeta(status=200, http_version="1.1", headers=HttpHeaders({}),
                           duration=0.25, node=None)   # 250 ms node HTTP wall

    class _MetaES:
        def bulk(self, operations=None, filter_path=None, **kw):
            return ObjectApiResponse(body={"errors": False, "took": 40}, meta=meta)

    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_chunk(_MetaES(), ["a"], _cfg(id_field=None, bulk_stats=True), counts, [], stats=stats)
    assert len(stats) == 1
    docs, _bytes, rtt, took, _cpu, http, outcome = stats[0]
    assert outcome == "ok" and took == 40 and http == 250.0     # took_ms=40 (ES), http_ms=250 (node)
    assert rtt >= 0.0                                           # real timer around the (instant) mock


# =====================================================================================
# Full-path diagnostics: docs_retried (429 retry volume) + fixed reject buckets, and the trimmed
# full-path filter_path. All bulk_stats-only and full-path-only (zero on a clean chunk).
# =====================================================================================

def test_diag_counts_docs_retried_and_reject_buckets(_no_backoff):
    from databricks_es_connector.bulk import _new_diag
    # id_field=None fast path: probe errors=True -> full reship carrying 201 / 429(retry->201) / 400 / 409.
    es = _RecordingES([
        {"errors": True},
        {"items": [{"index": {"status": 201}},
                   {"index": {"status": 429}},
                   {"index": {"status": 400, "_id": "d3", "error": {"reason": "map"}}},
                   {"index": {"status": 409, "_id": "d4", "error": {"reason": "conflict"}}}]},
        {"items": [{"index": {"status": 201}}]},          # retry of the 429'd line succeeds
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["a", "b", "c", "d"],
                       _cfg(id_field=None, max_retries_per_doc=3), counts, [], diag=diag)
    assert diag["docs_retried"] == 1          # one 429'd line re-sent
    assert diag["rejected_429"] == 1          # the 429, counted before the retry
    assert diag["rejected_4xx_other"] == 1    # the 400
    assert diag["rejected_409"] == 1          # the 409
    assert diag["rejected_5xx"] == 0
    assert counts == {"written": 2, "deleted": 0, "ignored": 0, "errors": 2}


def test_diag_zero_on_a_clean_fast_path():
    from databricks_es_connector.bulk import _new_diag
    es = _RecordingES([{"errors": False}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, [], diag=diag)
    assert counts["written"] == 2
    assert diag == _new_diag()                 # all zero: a clean chunk never touches the full path
    assert es.calls == [(["a", "b"], "errors")]   # only the probe, no full reship


def test_diag_5xx_bucket_and_delete_404_not_counted(_no_backoff):
    # 503 -> rejected_5xx; a delete-404 is an expected no-op (IGNORED) and must NOT count as a rejection.
    from databricks_es_connector.bulk import _new_diag
    from databricks_es_connector.config import EsWriteConfig
    cfg = EsWriteConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id",
                        require_existing_index=False, has_deletes=True, delete_flag_column="d")
    es = _RecordingES([{"items": [{"delete": {"status": 404}},
                                  {"index": {"status": 503, "_id": "x", "error": {"reason": "unavail"}}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["del", "idx"], cfg, counts, [], diag=diag)
    assert diag["rejected_5xx"] == 1
    assert diag["rejected_429"] == 0 and diag["rejected_4xx_other"] == 0 and diag["rejected_409"] == 0
    assert diag["docs_retried"] == 0           # 503 is not retryable by default (retry_on_doc_status=(429,))
    assert counts["ignored"] == 1 and counts["errors"] == 1   # delete-404 ignored; 503 an error


def test_full_path_uses_trimmed_filter_path_with_and_without_stats():
    # The full-path reship trims the ES response; the probe stays "errors"-only. Without stats the
    # trim omits took; with stats it prepends took.
    es = _RecordingES([{"errors": True},
                       {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]}])
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), {"written": 0, "deleted": 0, "ignored": 0,
                                                             "errors": 0}, [])
    assert es.calls[0][1] == "errors"                                        # probe unchanged
    assert es.calls[1][1] == "items.*.status,items.*.error,items.*._id"      # trimmed, no took

    es2 = _RecordingES([{"errors": True, "took": 1},
                        {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}], "took": 2}])
    _ship_ndjson_chunk(es2, ["a", "b"], _cfg(id_field=None), {"written": 0, "deleted": 0, "ignored": 0,
                                                              "errors": 0}, [], stats=[])
    assert es2.calls[0][1] == "errors,took"                                          # probe (stats)
    assert es2.calls[1][1] == "took,items.*.status,items.*.error,items.*._id"        # trimmed + took


def test_ndjson_writer_bulk_stats_surfaces_diag_columns(monkeypatch, _no_backoff):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    responses = [{"errors": True},
                 {"items": [{"index": {"status": 429}},
                            {"index": {"status": 400, "_id": "z", "error": {"reason": "bad"}}}], "took": 2},
                 {"items": [{"index": {"status": 201}}], "took": 1}]      # retry of the 429'd line
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: _RecordingES(responses))
    writer = make_ndjson_partition_writer(
        _cfg(id_field=None, chunk_size=500, bulk_stats=True, max_retries_per_doc=3))
    row = list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b"]})])))[0].iloc[0]
    assert int(row["docs_retried"]) == 1
    assert int(row["rejected_429"]) == 1
    assert int(row["rejected_4xx_other"]) == 1
    assert int(row["rejected_409"]) == 0 and int(row["rejected_5xx"]) == 0
    assert int(row["docs_deduped"]) == 0    # no create-409s on this index-mode chunk


# =====================================================================================
# op_type='create' + the create-409 append-only dedup (0.10.0). A create whose _id already
# exists returns 409, which classifies IGNORED (a benign dedup, not a rejection) and is tallied
# distinctly as docs_deduped. An index-mode 409 (external version conflict) stays a hard error.
# =====================================================================================

def test_classify_create_409_is_dedup_ignored_index_409_is_error():
    from databricks_es_connector.bulk import classify_bulk_result, IGNORED, ERROR, WRITTEN
    assert classify_bulk_result(False, "create", 409) == IGNORED   # append-only dedup: benign no-op
    assert classify_bulk_result(True, "create", 201) == WRITTEN    # a genuine create still counts written
    assert classify_bulk_result(False, "index", 409) == ERROR      # index-mode version conflict: an error
    assert classify_bulk_result(False, "delete", 409) == ERROR     # 409 on a delete: an error
    assert classify_bulk_result(False, "create", 400) == ERROR     # a real create rejection: an error


def test_diag_counts_create_409_as_deduped_not_rejected(_no_backoff):
    # create-mode resend: probe errors=True -> full reship carrying one new create (201) and two
    # already-existing docs (409). The two 409s are append-only dedups: IGNORED, tallied as
    # docs_deduped, and specifically NOT counted as rejected_409 or as errors.
    from databricks_es_connector.bulk import _new_diag
    es = _RecordingES([
        {"errors": True},
        {"items": [{"create": {"status": 201}},
                   {"create": {"status": 409, "_id": "d2", "error": {"reason": "version_conflict"}}},
                   {"create": {"status": 409, "_id": "d3", "error": {"reason": "version_conflict"}}}]},
    ])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(op_type="create"), counts, [], diag=diag)
    assert diag["docs_deduped"] == 2       # two already-existed: benign dedups, visible in bulk_stats
    assert diag["rejected_409"] == 0       # NOT a rejection under create mode
    assert counts == {"written": 1, "deleted": 0, "ignored": 2, "errors": 0}
    # Reconcile identity holds: written + deleted + errors + ignored == total_input (1+0+0+2 == 3).


def test_index_mode_409_is_error_not_deduped(_no_backoff):
    # The default op_type='index': a 409 (external version conflict) is a hard error, counted in
    # rejected_409, and never touches docs_deduped.
    from databricks_es_connector.bulk import _new_diag
    es = _RecordingES([{"errors": True},
                       {"items": [{"index": {"status": 409, "_id": "x", "error": {"reason": "conflict"}}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["a"], _cfg(), counts, [], diag=diag)   # default op_type=index
    assert counts["errors"] == 1 and counts["ignored"] == 0
    assert diag["rejected_409"] == 1 and diag["docs_deduped"] == 0


def test_bypass_fast_path_skips_probe_full_path_only(_no_backoff):
    # bypass_fast_path=True => no filter_path="errors" probe; the chunk is classified on the full path
    # directly (a single send with the trimmed item detail), so a clean chunk still counts written.
    es = _RecordingES([{"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(bypass_fast_path=True), counts, [])
    assert [c[1] for c in es.calls] == [_FULL_FP]      # only the full path, NO "errors" probe
    assert counts["written"] == 2


def test_bypass_fast_path_gives_exact_create_counts_on_mixed_chunk(_no_backoff):
    # The scenario that MIScounts under the default fast path (see the test below): a chunk with 1 new
    # + 2 existing create ops. Under bypass, one full-path send classifies exactly: new 201 -> written,
    # existing 409 -> deduped. No probe, no whole-chunk re-ship, so nothing self-409s.
    from databricks_es_connector.bulk import _new_diag
    es = _RecordingES([{"items": [{"create": {"status": 201}},
                                  {"create": {"status": 409}}, {"create": {"status": 409}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["new", "ex1", "ex2"],
                       _cfg(op_type="create", bypass_fast_path=True), counts, [], diag=diag)
    assert [c[1] for c in es.calls] == [_FULL_FP]      # no probe
    assert counts["written"] == 1 and counts["ignored"] == 2, counts
    assert diag["docs_deduped"] == 2


def test_default_fast_path_miscounts_mixed_create_chunk(_no_backoff):
    # Documents the count caveat on op_type="create" under the DEFAULT fast path (bypass_fast_path=False):
    # the probe sees errors:true (the two existing docs 409), re-ships the WHOLE chunk, and the new doc
    # the probe just created now self-409s -> written=0, docs_deduped=3 (not 1 / 2). The doc is still
    # correct in ES (present once); only the attribution is off. Use bypass_fast_path=True (test above)
    # for exact counts. This asserts the documented behavior so it cannot change silently.
    from databricks_es_connector.bulk import _new_diag
    es = _RecordingES([{"errors": True},
                       {"items": [{"create": {"status": 409}}, {"create": {"status": 409}},
                                  {"create": {"status": 409}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    diag = _new_diag()
    _ship_ndjson_chunk(es, ["new", "ex1", "ex2"], _cfg(op_type="create"), counts, [], diag=diag)
    assert [c[1] for c in es.calls] == ["errors", _FULL_FP]   # probe, then whole-chunk re-ship
    assert counts["written"] == 0 and counts["ignored"] == 3, counts
    assert diag["docs_deduped"] == 3


def test_bulk_stats_diag_keys_agree_across_new_diag_and_stat_keys():
    # Lock the invariant the comments on _DIAG_KEYS / _STAT_KEYS state: every full-path diag key must
    # be produced by _new_diag() and carried in _STAT_KEYS (which now splices *_DIAG_KEYS in, so this
    # holds by construction and this test catches a future hand-edit that breaks it). The remaining
    # leg -- _STAT_KEYS vs the bulk_write summary_schema string -- is enforced at runtime by mapInPandas
    # in the integration tier (a missing/extra column fails the write), which cannot be reached offline.
    from databricks_es_connector.bulk import _new_diag, _DIAG_KEYS, _STAT_KEYS
    assert set(_new_diag()) == set(_DIAG_KEYS)
    assert set(_DIAG_KEYS).issubset(_STAT_KEYS)
    assert "docs_deduped" in _DIAG_KEYS and "docs_deduped" in _STAT_KEYS
