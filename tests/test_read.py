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
