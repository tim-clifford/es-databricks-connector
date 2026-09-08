"""Unit tests for the write path (bulk.py shipper + config validation + the fan-out).

The Spark-side builder (spark_serialize.build_ndjson) needs a live Spark session and is proven in
the integration tier; here we cover everything that does NOT need Spark:
  - iter_bulk_response_outcomes: es.bulk response item -> WRITTEN/DELETED/IGNORED/ERROR.
  - _ship_ndjson_chunk: tallying + the per-document 429 retry the connector treats as load-bearing.
  - _ship_ndjson_lines: the write_concurrency fan-out (all lines shipped once, tally == serial).
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
    es = _FakeES([resp])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    samples = []
    _ship_ndjson_chunk(es, ["l1", "l2", "l3"], _cfg(id_field=None), counts, samples)
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
    _ship_ndjson_chunk(es, ["a", "b", "c"], _cfg(id_field=None, max_retries_per_doc=3), counts, [])
    assert counts["written"] == 3 and counts["errors"] == 0
    assert es.calls[0] == ["a", "b", "c"]
    assert es.calls[1] == ["b"]          # only the retryable line was resent


def test_ship_chunk_429_becomes_error_after_max_retries():
    # A 429 that never clears is counted as an error once retries are exhausted (loud, not lost).
    always_429 = {"items": [{"index": {"status": 429, "_id": "x"}}]}
    es = _FakeES([always_429, always_429, always_429])   # initial + 2 retries
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["x"], _cfg(id_field=None, max_retries_per_doc=2), counts, [])
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

    ok1 = lambda: {"items": [{"index": {"status": 201}}]}
    # 5 rows, chunk_size=2 -> chunks of [2,2,1] -> 3 bulk calls.
    es = _FakeES([{"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]},
                  {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]},
                  {"items": [{"index": {"status": 201}}]}])
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
    # created ONCE per partition (in _write) and reused for every batch, not spun up per batch. Feed
    # the writer three batches and assert exactly one pool was constructed, while every line still
    # ships exactly once with the correct tally. (RED-BEFORE-GREEN: creating the pool per
    # _ship_ndjson_lines call, as before, would construct one per batch -> this asserts == 1.)
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
    # write_concurrency fans the chunk shipping across worker threads (bulk._ship_ndjson_lines) and
    # must not warn on its own; the per-node connection pool is sized to it so the workers are not
    # capped below the configured value.
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


# --- _ship_ndjson_lines: the write_concurrency fan-out over pre-built lines --------------------
# No Spark: _ship_ndjson_lines takes an ES client and a list of lines, so a thread-safe fake client
# exercises the fan-out directly. The live end-to-end proof is integration test_concurrency_roundtrip.

class _ThreadSafeFakeES:
    """Returns 201 for every operation and records what it shipped. Thread-safe for fan-out tests."""
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.chunks = []       # each es.bulk() call's operations list
        self.all_ops = []      # every op line shipped, flattened

    def bulk(self, operations=None, **kw):
        ops = list(operations)
        with self._lock:
            self.chunks.append(ops)
            self.all_ops.extend(ops)
        return {"items": [{"index": {"status": 201}} for _ in ops]}


def _zero_counts():
    return {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}


def test_ship_ndjson_lines_fans_all_lines_exactly_once():
    from databricks_es_connector.bulk import _ship_ndjson_lines
    es = _ThreadSafeFakeES()
    lines = [f"L{i}" for i in range(10)]
    counts, samples = _zero_counts(), []
    _ship_ndjson_lines(es, lines, _cfg(id_field=None, write_concurrency=3, chunk_size=2), counts, samples)
    # every line shipped exactly once across the workers -- no drops, no duplicates
    assert sorted(es.all_ops) == sorted(lines)
    assert len(es.all_ops) == 10
    assert counts["written"] == 10 and counts["errors"] == 0


def test_ship_ndjson_lines_concurrency_matches_serial_tally():
    # The whole point of the fan-out: identical accounting regardless of write_concurrency.
    from databricks_es_connector.bulk import _ship_ndjson_lines
    lines = [f"L{i}" for i in range(7)]

    def run(wc):
        es = _ThreadSafeFakeES()
        counts, samples = _zero_counts(), []
        _ship_ndjson_lines(es, lines, _cfg(id_field=None, write_concurrency=wc, chunk_size=2), counts, samples)
        return counts, sorted(es.all_ops)

    serial_counts, serial_ops = run(1)
    conc_counts, conc_ops = run(4)
    assert serial_counts == conc_counts == {"written": 7, "deleted": 0, "ignored": 0, "errors": 0}
    assert serial_ops == conc_ops == sorted(lines)


def test_ship_ndjson_lines_merges_errors_and_samples_across_workers():
    # Per-worker error tallies and (bounded) sample lists must merge correctly on join.
    from databricks_es_connector.bulk import _ship_ndjson_lines

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
    counts, samples = _zero_counts(), []
    _ship_ndjson_lines(_SelectiveFakeES(), lines, _cfg(write_concurrency=3, chunk_size=2), counts, samples)
    assert counts["written"] == 8 and counts["errors"] == 3
    assert len(samples) == 3 and all("boom" in s["reason"] for s in samples)


def test_ship_ndjson_lines_worker_exception_fails_closed(monkeypatch):
    # RED-BEFORE-GREEN guard: a worker exception must propagate (via f.result()), so a partial write
    # FAILS the partition rather than reporting the docs a dead worker never sent as a clean count.
    # Deleting the `f.result()` loop makes this pass silently. _ship_ndjson_chunk normally catches
    # transport errors and counts them, so force a raw raise to exercise the re-raise guard itself.
    from databricks_es_connector import bulk as bulk_mod

    def _boom(es, chunk, cfg, counts, samples, stats=None):
        raise RuntimeError("worker died mid-ship")

    monkeypatch.setattr(bulk_mod, "_ship_ndjson_chunk", _boom)
    with pytest.raises(RuntimeError, match="worker died"):
        bulk_mod._ship_ndjson_lines(_ThreadSafeFakeES(), [f"L{i}" for i in range(10)],
                                    _cfg(write_concurrency=3, chunk_size=2), _zero_counts(), [])


def test_config_write_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="write_concurrency must be >= 1"):
        EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", write_concurrency=0)


# --- _ship_ndjson_chunk: the filter_path="errors" fast path (GIL avoidance) --------------------
# On an idempotent (id_field set), delete-free write, a clean bulk needs only the top-level `errors`
# flag, so the chunk is shipped with filter_path="errors" and the per-item response is never decoded
# or classified in Python (the GIL-held cost that serialized write_concurrency threads). On ANY
# failure the chunk is re-shipped with a FULL response and the existing classify + 429-retry runs;
# re-shipping is safe only because every op is an idempotent upsert (id_field set, no deletes), so a
# doc that already succeeded is simply upserted again. The gate is exactly id_field-set + no-deletes.

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
    assert es.calls[1] == (["a", "b"], None)              # then full re-ship for detail


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
    assert es.calls[1] == (["a", "b", "c"], None)         # full re-ship
    assert es.calls[2] == (["b"], None)                   # only the 429 line retried


def test_fast_path_missing_errors_key_fails_closed():
    # A probe response without an `errors` key must NOT read as clean: treat as error and re-ship full.
    es = _RecordingES([{}, {"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, [])
    assert counts["written"] == 2 and counts["errors"] == 0
    assert len(es.calls) == 2 and es.calls[1] == (["a", "b"], None)


def test_fast_path_disabled_without_id_field():
    # No id_field -> ES auto-ids -> re-ship would DUPLICATE, so the fast path is off: one FULL request
    # (no filter_path), classified per item, exactly as before.
    es = _RecordingES([{"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, [])
    assert counts["written"] == 2
    assert es.calls == [(["a", "b"], None)]               # full response, never filter_path="errors"


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
    assert es.calls == [(["a", "b"], None)]


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
    assert es.calls[1] == (["a", "b"], None)


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


# --- bulk_stats: per-partition send aggregates (docs/send, rtt vs ES took) ---------------------
# Off by default and zero-overhead. When on, each es.bulk send is timed and (docs, rtt_ms, took_ms)
# recorded; the writer aggregates per partition and _merge_partition_results surfaces one entry per
# partition under result["bulk_stats"]. `took` is requested via filter_path so it is available
# without the per-item array (the GIL win is preserved).

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
    # (docs, rtt_ms, took_ms); one send has a None took (ES omitted it) -> excluded from took only.
    agg = _aggregate_bulk_stats([(100, 10.0, 5.0), (100, 20.0, 15.0), (50, 30.0, None)])
    assert agg["n_sends"] == [3]
    assert agg["docs_sent"] == [250]           # retries would count again; here 3 distinct sends
    assert agg["rtt_ms_mean"] == [20.0] and agg["rtt_ms_p50"] == [20.0]
    assert agg["rtt_ms_p95"] == [29.0] and agg["rtt_ms_max"] == [30.0]
    assert agg["took_ms_mean"] == [10.0]       # (5+15)/2, None excluded
    assert agg["took_ms_p50"] == [10.0] and agg["took_ms_max"] == [15.0]


def test_aggregate_bulk_stats_empty_partition_is_nulls_not_crash():
    from databricks_es_connector.bulk import _aggregate_bulk_stats
    agg = _aggregate_bulk_stats([])
    assert agg["n_sends"] == [0] and agg["docs_sent"] == [0]
    assert agg["rtt_ms_mean"] == [None] and agg["rtt_ms_max"] == [None]
    assert agg["took_ms_p95"] == [None]


def test_ship_chunk_records_a_send_on_the_fast_path():
    # stats list given -> the clean fast-path send is timed and recorded, and `took` is requested.
    es = _RecordingES([{"errors": False, "took": 7}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, stats=stats, error_samples=[])
    assert counts["written"] == 2
    assert es.calls == [(["a", "b"], "errors,took")]    # took requested, items still omitted
    assert len(stats) == 1
    docs, rtt_ms, took_ms = stats[0]
    assert docs == 2 and took_ms == 7 and rtt_ms >= 0.0


def test_ship_chunk_no_stats_and_no_took_when_disabled():
    # stats=None (default) -> no recording, and the fast path requests only "errors" (no took).
    es = _RecordingES([{"errors": False}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(), counts, [])
    assert counts["written"] == 2 and es.calls == [(["a", "b"], "errors")]


def test_ship_chunk_records_took_on_full_path():
    # Full path (id_field=None) records the send too, reading took from the full response.
    es = _RecordingES([{"items": [{"index": {"status": 201}}, {"index": {"status": 201}}], "took": 3}])
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_chunk(es, ["a", "b"], _cfg(id_field=None), counts, stats=stats, error_samples=[])
    assert counts["written"] == 2 and len(stats) == 1
    assert stats[0][0] == 2 and stats[0][2] == 3


def test_ship_ndjson_lines_merges_stats_across_workers():
    from databricks_es_connector.bulk import _ship_ndjson_lines
    es = _FastFakeES()   # returns {"errors": False} on the probe; no took, so took_ms is None
    lines = [f"L{i}" for i in range(10)]
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    stats = []
    _ship_ndjson_lines(es, lines, _cfg(write_concurrency=3, chunk_size=2), counts, [], stats=stats)
    # write_concurrency=3 strides 10 lines into slices of 4/3/3, each chunked by 2 -> 2+2+2 = 6 sends;
    # every line is shipped exactly once (docs sum to 10), and every send is recorded.
    assert len(stats) == 6 and sum(s[0] for s in stats) == 10
    assert all(s[1] >= 0.0 for s in stats)


def test_writer_emits_per_partition_bulk_stats_columns(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    es = _FastFakeES()
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)
    writer = make_ndjson_partition_writer(_cfg(bulk_stats=True, chunk_size=2))
    out = list(writer(iter([pd.DataFrame({"_ndjson": ["a", "b", "c", "d", "e"]})])))
    row = out[0].iloc[0]
    assert "n_sends" in out[0].columns and "rtt_ms_p95" in out[0].columns
    assert int(row["n_sends"]) == 3 and int(row["docs_sent"]) == 5   # chunks [2,2,1]
    assert float(row["rtt_ms_max"]) >= 0.0


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
         "rtt_ms_mean": 12.0, "rtt_ms_p50": 12.0, "rtt_ms_p95": 18.0, "rtt_ms_max": 20.0,
         "took_ms_mean": 4.0, "took_ms_p50": 4.0, "took_ms_p95": 6.0, "took_ms_max": 7.0},
        {"written": 6, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
         "total_input": 6, "error_samples": "[]", "n_sends": 3, "docs_sent": 6,
         "rtt_ms_mean": 9.0, "rtt_ms_p50": 9.0, "rtt_ms_p95": 11.0, "rtt_ms_max": 12.0,
         "took_ms_mean": 3.0, "took_ms_p50": 3.0, "took_ms_p95": 4.0, "took_ms_max": 5.0},
    ]
    result = _merge_partition_results(rows)
    assert result["written"] == 10
    assert "bulk_stats" in result and len(result["bulk_stats"]) == 2
    assert result["bulk_stats"][0]["n_sends"] == 2 and result["bulk_stats"][1]["rtt_ms_max"] == 12.0


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

    def bulk(self, operations=None, filter_path=None, **kw):
        ops = list(operations)
        with self._lock:
            self.calls.append((ops, filter_path))
            self.all_ops.extend(ops)
        if filter_path and "errors" in filter_path:      # "errors" or "errors,took" (stats mode)
            return {"errors": False, "took": 1}
        return {"items": [{"index": {"status": 201}} for _ in ops], "took": 1}


def test_fast_path_under_fan_out_ships_each_line_once_via_probe():
    # The fan-out (write_concurrency) composes with the fast path: every worker probes with
    # filter_path="errors", every line ships exactly once, all counted written, no full re-ship.
    from databricks_es_connector.bulk import _ship_ndjson_lines
    es = _FastFakeES()
    lines = [f"L{i}" for i in range(10)]
    counts = {"written": 0, "deleted": 0, "ignored": 0, "errors": 0}
    _ship_ndjson_lines(es, lines, _cfg(write_concurrency=3, chunk_size=2), counts, [])
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
