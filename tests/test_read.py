"""Unit tests for read.py's pure seams — EsReadConfig validation and the hit->row coercion — that
don't need Spark or a live ES. The scroll/PIT loop and createDataFrame are exercised live in the
integration tier; here we cover the logic around them.
"""
import datetime as dt

import pytest

from databricks_es_connector import EsReadConfig
from databricks_es_connector.read import _coerce_hit, _schema_field_tokens


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
