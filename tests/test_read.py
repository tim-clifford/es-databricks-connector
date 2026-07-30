"""Unit tests for read.py's pure seams — EsReadConfig validation and the hit->row coercion — that
don't need Spark or a live ES. The scroll/PIT loop and createDataFrame are exercised live in the
integration tier; here we cover the logic around them.
"""
import datetime as dt

import pytest

from databricks_es_connector import EsReadConfig
from databricks_es_connector.read import (
    _coerce_hit, _schema_field_tokens, _resolve_num_slices, _make_slice_reader,
)


def _rcfg(**kw):
    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i")
    base.update(kw)
    return EsReadConfig(**base)


# --- EsReadConfig ----------------------------------------------------------------------------

def test_read_config_defaults():
    c = _rcfg()
    assert c.query is None and c.num_slices is None
    assert c.batch_size == 1000 and c.pit_keep_alive == "1m" and c.include_id is True
    assert "hosts" in c.client_kwargs()   # connection fields are shared from EsConnection

def test_read_config_requires_hosts():
    with pytest.raises(ValueError, match="hosts"):
        EsReadConfig(index="i", basic_auth=("u", "p"), hosts="")

def test_read_config_requires_auth():
    with pytest.raises(ValueError, match="api_key or basic_auth"):
        EsReadConfig(hosts="https://h:9200", index="i")

def test_read_config_rejects_nonpositive_batch():
    with pytest.raises(ValueError, match="batch_size"):
        _rcfg(batch_size=0)

def test_read_config_rejects_bad_num_slices():
    with pytest.raises(ValueError, match="num_slices"):
        _rcfg(num_slices=0)


# --- _coerce_hit: map one ES hit to a schema-shaped row --------------------------------------

_TOKENS = [("doc_id", "string"), ("event_ts", "timestamp"), ("n", "long")]


def test_coerce_hit_applies_inverse_transforms():
    src = {"doc_id": "k1", "event_ts": 1609459200000, "n": 5}   # epoch-ms for 2021-01-01Z
    row = _coerce_hit(src, "k1", _TOKENS, id_field="doc_id", include_id=True)
    assert row["doc_id"] == "k1"
    assert row["event_ts"] == dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    assert row["n"] == 5

def test_coerce_hit_missing_field_is_none():
    row = _coerce_hit({"doc_id": "k1"}, "k1", _TOKENS, id_field="doc_id", include_id=True)
    assert row["event_ts"] is None and row["n"] is None

def test_coerce_hit_falls_back_to_es_id_when_source_omits_it():
    # If the id column isn't in _source, take it from the hit's _id (writes keep it in _source,
    # but a caller could have dropped it or read a foreign index).
    row = _coerce_hit({"n": 9}, "the-id", _TOKENS, id_field="doc_id", include_id=True)
    assert row["doc_id"] == "the-id"

def test_coerce_hit_uses_source_id_over_es_id_when_present():
    row = _coerce_hit({"doc_id": "from-source", "n": 1}, "es-meta-id",
                      _TOKENS, id_field="doc_id", include_id=True)
    assert row["doc_id"] == "from-source"

def test_coerce_hit_no_id_field():
    # id_field=None => no special-casing; every column read straight from _source.
    row = _coerce_hit({"doc_id": "x", "n": 2}, "ignored", _TOKENS, id_field=None, include_id=True)
    assert row["doc_id"] == "x" and row["n"] == 2


# --- _resolve_num_slices: configured value wins; else shard count; else fallback 1 -----------

class _FakeIndices:
    def __init__(self, settings):
        self._settings = settings
    def get_settings(self, index, flat_settings=False):
        if self._settings is None:
            raise RuntimeError("boom")
        return self._settings

class _FakeES:
    def __init__(self, settings=None):
        self.indices = _FakeIndices(settings)


def test_resolve_num_slices_uses_configured():
    es = _FakeES(settings={"i": {"settings": {"index.number_of_shards": "5"}}})
    assert _resolve_num_slices(es, "i", 8) == 8   # configured overrides shard count

def test_resolve_num_slices_defaults_to_shard_count():
    es = _FakeES(settings={"i": {"settings": {"index.number_of_shards": "3"}}})
    assert _resolve_num_slices(es, "i", None) == 3

def test_resolve_num_slices_falls_back_to_one_on_error():
    es = _FakeES(settings=None)   # get_settings raises
    assert _resolve_num_slices(es, "i", None) == 1

def test_resolve_num_slices_warns_on_fallback(caplog):
    # The serial single-slice fallback must not be silent — it forfeits parallelism.
    import logging
    es = _FakeES(settings=None)   # get_settings raises
    with caplog.at_level(logging.WARNING, logger="databricks_es_connector.read"):
        assert _resolve_num_slices(es, "my_index", None) == 1
    assert any("my_index" in r.message and "single" in r.message.lower()
               for r in caplog.records), "expected a WARNING naming the index on fallback"

def test_resolve_num_slices_does_not_warn_when_configured(caplog):
    # An explicit num_slices skips the shard lookup entirely — no warning even if get_settings would fail.
    import logging
    es = _FakeES(settings=None)
    with caplog.at_level(logging.WARNING, logger="databricks_es_connector.read"):
        assert _resolve_num_slices(es, "my_index", 4) == 4
    assert caplog.records == []

def test_resolve_num_slices_floors_configured_at_one():
    es = _FakeES(settings={"i": {"settings": {"index.number_of_shards": "3"}}})
    assert _resolve_num_slices(es, "i", 0) == 1   # never < 1


# --- _make_slice_reader: yields schema-shaped pandas frames, sliced correctly ----------------

def test_slice_reader_yields_coerced_frames(monkeypatch):
    pd = pytest.importorskip("pandas")
    import databricks_es_connector.read as read_mod

    # Stub _slice_hits so no ES is needed; capture the slice_spec it was called with per slice.
    seen_specs = []
    def _fake_slice_hits(es, pit_id, query, batch_size, keep_alive, slice_spec=None):
        seen_specs.append(slice_spec)
        sid = slice_spec["id"] if slice_spec else 0
        # one hit per slice, id encodes the slice so we can assert per-partition output
        return iter([{"_id": f"k{sid}", "_source": {"doc_id": f"k{sid}", "n": sid}}])
    monkeypatch.setattr(read_mod, "_slice_hits", _fake_slice_hits)

    cfg = _rcfg(num_slices=3)
    tokens = [("doc_id", "string"), ("n", "long")]
    reader = _make_slice_reader(cfg, "pit-123", {"match_all": {}}, tokens, num_slices=3)

    # mapInPandas hands the fn an iterator of pandas frames; here one frame carrying slice ids 0..2.
    out = list(reader(iter([pd.DataFrame({"id": [0, 1, 2]})])))
    assert len(out) == 3                                   # one frame per slice
    assert [df.iloc[0]["doc_id"] for df in out] == ["k0", "k1", "k2"]
    assert list(out[0].columns) == ["doc_id", "n"]         # schema column order
    # With num_slices=3 each read must be a sliced scroll {"id": i, "max": 3}.
    assert seen_specs == [{"id": 0, "max": 3}, {"id": 1, "max": 3}, {"id": 2, "max": 3}]

def test_slice_reader_single_slice_is_unsliced(monkeypatch):
    pd = pytest.importorskip("pandas")
    import databricks_es_connector.read as read_mod
    seen_specs = []
    def _fake_slice_hits(es, pit_id, query, batch_size, keep_alive, slice_spec=None):
        seen_specs.append(slice_spec)
        return iter([{"_id": "k0", "_source": {"doc_id": "k0", "n": 0}}])
    monkeypatch.setattr(read_mod, "_slice_hits", _fake_slice_hits)

    reader = _make_slice_reader(_rcfg(num_slices=1), "pit", {"match_all": {}},
                                [("doc_id", "string"), ("n", "long")], num_slices=1)
    list(reader(iter([pd.DataFrame({"id": [0]})])))
    # A single slice must NOT pass a slice spec (ES rejects max=1); it reads the PIT unsliced.
    assert seen_specs == [None]

def test_slice_reader_empty_slice_yields_nothing(monkeypatch):
    pd = pytest.importorskip("pandas")
    import databricks_es_connector.read as read_mod
    monkeypatch.setattr(read_mod, "_slice_hits", lambda *a, **k: iter([]))
    reader = _make_slice_reader(_rcfg(num_slices=2), "pit", {"match_all": {}},
                                [("doc_id", "string")], num_slices=2)
    out = list(reader(iter([pd.DataFrame({"id": [0, 1]})])))
    assert out == []   # no rows in any slice => no frames yielded


# --- _slice_hits: the PIT + search_after paging loop -----------------------------------------
# A fake ES client returns canned pages, recording the search kwargs so we can assert the loop
# pages correctly (search_after advances, PIT carried, sort/slice passed) and terminates on empty.

class _FakeSearchES:
    """Fake Elasticsearch: .search() returns queued pages in order; records each call's kwargs."""
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []
        self.closed = []
        self.client_closed = 0
    def search(self, **kwargs):
        self.calls.append(kwargs)
        page = self._pages.pop(0) if self._pages else {"hits": {"hits": []}}
        return page
    # PIT lifecycle (used by the entry-point tests below)
    def open_point_in_time(self, index, keep_alive):
        return {"id": "pit-abc"}
    def close_point_in_time(self, id):
        self.closed.append(id)
    def close(self):
        # The driver-side client is closed to avoid a connection-pool leak (distinct from the PIT).
        self.client_closed += 1

def _page(*hit_ids, pit_id=None):
    hits = [{"_id": h, "_source": {"doc_id": h}, "sort": [i]} for i, h in enumerate(hit_ids)]
    page = {"hits": {"hits": hits}}
    if pit_id:
        page["pit_id"] = pit_id
    return page


def test_slice_hits_pages_until_empty():
    from databricks_es_connector.read import _slice_hits
    es = _FakeSearchES([_page("a", "b"), _page("c"), _page()])  # 2 + 1 + terminate
    got = [h["_id"] for h in _slice_hits(es, "pit-1", {"match_all": {}}, 500, "1m")]
    assert got == ["a", "b", "c"]
    # 3 search calls: two returning hits, one empty that stops the loop.
    assert len(es.calls) == 3
    # First call has no search_after; later calls carry the previous page's last sort value.
    assert "search_after" not in es.calls[0]
    assert es.calls[1]["search_after"] == [1]   # last hit of page 1 (index 1)
    assert es.calls[2]["search_after"] == [0]   # last hit of page 2 (single hit, index 0)

def test_slice_hits_carries_pit_and_sort_no_index():
    from databricks_es_connector.read import _slice_hits
    es = _FakeSearchES([_page("a"), _page()])
    list(_slice_hits(es, "pit-xyz", {"term": {"x": 1}}, 250, "2m"))
    c = es.calls[0]
    assert c["pit"] == {"id": "pit-xyz", "keep_alive": "2m"}
    assert c["sort"] == [{"_shard_doc": "asc"}]
    assert c["size"] == 250 and c["query"] == {"term": {"x": 1}}
    assert "index" not in c              # PIT scopes the index; passing index= would error
    assert "slice" not in c              # no slice unless requested

def test_slice_hits_passes_slice_spec_when_given():
    from databricks_es_connector.read import _slice_hits
    es = _FakeSearchES([_page()])
    list(_slice_hits(es, "pit", {"match_all": {}}, 500, "1m", slice_spec={"id": 2, "max": 4}))
    assert es.calls[0]["slice"] == {"id": 2, "max": 4}

def test_slice_hits_follows_refreshed_pit_id():
    # ES can return a refreshed pit_id per page; the next request must use it.
    from databricks_es_connector.read import _slice_hits
    es = _FakeSearchES([_page("a", pit_id="pit-2"), _page()])
    list(_slice_hits(es, "pit-1", {"match_all": {}}, 500, "1m"))
    assert es.calls[0]["pit"]["id"] == "pit-1"   # first uses the opened id
    assert es.calls[1]["pit"]["id"] == "pit-2"   # second uses the refreshed id


# --- entry-point orchestration + PIT lifecycle (fake Spark + stubbed ES) ---------------------

class _FakeSpark:
    """Minimal SparkSession stand-in. createDataFrame records (rows, schema) and echoes them back;
    range(n, numPartitions=n) returns a fake DF whose mapInPandas records how it was called."""
    def __init__(self):
        self.created = None
        self.range_n = None
        self.mapped = None
    def createDataFrame(self, rows, schema):
        self.created = (list(rows), schema)
        return ("collected-df", self.created)
    def range(self, n, numPartitions=None):
        self.range_n = n
        return _FakeRangeDF(self, nparts=numPartitions)

class _FakeRangeDF:
    def __init__(self, spark, nparts=None):
        self._spark = spark
        self._nparts = nparts
    def mapInPandas(self, fn, schema):
        self._spark.mapped = {"fn": fn, "schema": schema, "nparts": self._nparts}
        return ("lazy-distributed-df", self._spark.mapped)


class _Schema:
    """Duck-typed Spark schema: has .fields, each with .name and .dataType.simpleString()."""
    def __init__(self, fields):  # fields: list of (name, token)
        self.fields = [_Field(n, t) for n, t in fields]

class _Field:
    def __init__(self, name, token):
        self.name = name
        self.dataType = _DT(token)

class _DT:
    def __init__(self, token): self._t = token
    def simpleString(self): return self._t


def _install_fake_es(monkeypatch, es):
    """Make `from elasticsearch import Elasticsearch` inside read.py return our fake."""
    import elasticsearch
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda **kw: es)


def test_read_index_collect_opens_and_closes_pit(monkeypatch):
    import databricks_es_connector.read as read_mod
    es = _FakeSearchES([_page("a", "b"), _page()])
    _install_fake_es(monkeypatch, es)
    spark = _FakeSpark()
    schema = _Schema([("doc_id", "string")])

    out = read_mod.read_index_collect(spark, _rcfg(), schema)
    # Driver path collects rows and builds a DataFrame with the declared schema.
    rows, used_schema = spark.created
    assert [r["doc_id"] for r in rows] == ["a", "b"]
    assert used_schema is schema
    # PIT was opened and CLOSED, and the driver client was closed too (no connection-pool leak).
    assert es.closed == ["pit-abc"]
    assert es.client_closed == 1

def test_read_index_collect_closes_pit_even_on_error(monkeypatch):
    import databricks_es_connector.read as read_mod
    class _BoomES(_FakeSearchES):
        def search(self, **kw):
            raise RuntimeError("es down")
    es = _BoomES([])
    _install_fake_es(monkeypatch, es)
    with pytest.raises(RuntimeError, match="es down"):
        read_mod.read_index_collect(_FakeSpark(), _rcfg(), _Schema([("doc_id", "string")]))
    assert es.closed == ["pit-abc"]   # finally still closed the PIT
    assert es.client_closed == 1      # and the driver client

def test_read_index_distributed_opens_pit_and_fans_out(monkeypatch):
    import databricks_es_connector.read as read_mod
    es = _FakeSearchES([])
    _install_fake_es(monkeypatch, es)
    # Force a known slice count so we can assert the fan-out width.
    monkeypatch.setattr(read_mod, "_resolve_num_slices", lambda es, index, cfg_n: 4)
    spark = _FakeSpark()
    schema = _Schema([("doc_id", "string")])

    df = read_mod.read_index(spark, _rcfg(), schema)
    # Distributed: spark.range(num_slices, numPartitions=num_slices) + mapInPandas(reader, schema);
    # NOT createDataFrame.
    assert spark.range_n == 4
    assert spark.mapped["nparts"] == 4                 # one partition per slice, straight from range
    assert spark.mapped["schema"] is schema
    assert callable(spark.mapped["fn"])                # the slice reader
    assert spark.created is None                        # data never collected to the driver
    # The driver client is closed before returning (executors build their own), but the PIT is NOT
    # closed here — the lazy DataFrame's executors read it later.
    assert es.client_closed == 1
    # Deliberately does NOT close the PIT (lazy DF must outlive this call).
    assert es.closed == []

def test_read_index_validates_before_touching_es(monkeypatch):
    import databricks_es_connector.read as read_mod
    es = _FakeSearchES([])
    _install_fake_es(monkeypatch, es)
    # Empty schema => raise, and ES/PIT must not have been opened.
    with pytest.raises(ValueError, match="StructType schema"):
        read_mod.read_index(_FakeSpark(), _rcfg(), _Schema([]))
    with pytest.raises(ValueError, match="index is required"):
        read_mod.read_index(_FakeSpark(), _rcfg(index=""), _Schema([("doc_id", "string")]))

def test_read_index_collect_swallows_pit_close_failure(monkeypatch):
    # A failure closing the PIT OR the client must not fail the read — the rows are already
    # collected, and the PIT will expire on its own. (Covers both defensive excepts in the finally.)
    import databricks_es_connector.read as read_mod
    class _CloseBoomES(_FakeSearchES):
        def close_point_in_time(self, id):
            raise RuntimeError("pit close failed")
        def close(self):
            raise RuntimeError("client close failed")
    es = _CloseBoomES([_page("a"), _page()])
    _install_fake_es(monkeypatch, es)
    spark = _FakeSpark()
    out = read_mod.read_index_collect(spark, _rcfg(), _Schema([("doc_id", "string")]))
    rows, _ = spark.created
    assert [r["doc_id"] for r in rows] == ["a"]   # read succeeded despite both close failures

def test_read_index_distributed_swallows_client_close_failure(monkeypatch):
    # In the distributed path the driver client close is best-effort too: a failure must not stop
    # us returning the lazy DataFrame. (Covers the except around es.close() in read_index.)
    import databricks_es_connector.read as read_mod
    class _CloseBoomES(_FakeSearchES):
        def close(self):
            raise RuntimeError("client close failed")
    es = _CloseBoomES([])
    _install_fake_es(monkeypatch, es)
    monkeypatch.setattr(read_mod, "_resolve_num_slices", lambda es, index, cfg_n: 2)
    spark = _FakeSpark()
    df = read_mod.read_index(spark, _rcfg(), _Schema([("doc_id", "string")]))
    assert spark.mapped["nparts"] == 2     # returned the lazy DF despite the close failure
    assert es.closed == []                 # PIT still not closed (lazy)
