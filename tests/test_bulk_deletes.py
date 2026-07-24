"""Unit tests for delete support: EsConfig validation + the pure bulk-result classifier.

No Spark, no ES client. The classifier is the exact rule that decides whether a
delete-404 is a silent no-op or a counted error, so it gets dedicated coverage here.
"""
import pytest

from databricks_es_connector.config import EsConfig
from databricks_es_connector.bulk import (
    classify_bulk_result, make_partition_writer, WRITTEN, DELETED, IGNORED, ERROR,
)


# --- EsConfig validation for the delete params -------------------------------------------

def _cfg(**kw):
    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i")
    base.update(kw)
    return EsConfig(**base)


def test_config_default_has_no_deletes():
    c = _cfg(id_field="doc_id")
    assert c.has_deletes is False and c.delete_flag_column is None


def test_config_has_deletes_requires_flag_column():
    with pytest.raises(ValueError, match="delete_flag_column"):
        _cfg(id_field="doc_id", has_deletes=True)


def test_config_has_deletes_requires_id_field():
    with pytest.raises(ValueError, match="id_field"):
        _cfg(has_deletes=True, delete_flag_column="d")


def test_config_flag_column_without_has_deletes_raises():
    # A flag column with deletes off would silently do nothing — reject it.
    with pytest.raises(ValueError, match="has_deletes is False"):
        _cfg(id_field="doc_id", delete_flag_column="d")


def test_config_valid_delete_setup():
    c = _cfg(id_field="doc_id", has_deletes=True, delete_flag_column="d")
    assert c.has_deletes is True and c.delete_flag_column == "d"


# --- the pure classifier: the scoped 404-delete suppression ------------------------------

def test_ok_index_is_written():
    assert classify_bulk_result(True, "index", 200) == WRITTEN
    assert classify_bulk_result(True, "create", 201) == WRITTEN
    assert classify_bulk_result(True, "update", 200) == WRITTEN


def test_ok_delete_is_deleted():
    assert classify_bulk_result(True, "delete", 200) == DELETED


def test_delete_404_is_ignored_not_error():
    # THE no-op case: deleting a doc that isn't there is expected on CDF replays / filtered rows.
    assert classify_bulk_result(False, "delete", 404) == IGNORED


def test_index_404_is_still_an_error():
    # A 404 on a NON-delete op must NOT be suppressed — suppression is scoped to deletes only.
    assert classify_bulk_result(False, "index", 404) == ERROR
    assert classify_bulk_result(False, "update", 404) == ERROR


def test_delete_non_404_is_still_an_error():
    # A delete failing for a reason other than 'not found' is a real error.
    for status in (409, 429, 500, 503):
        assert classify_bulk_result(False, "delete", status) == ERROR


def test_index_non_404_failures_are_errors():
    for status in (400, 409, 429, 500):
        assert classify_bulk_result(False, "index", status) == ERROR


# --- the writer glue: streaming_bulk results -> {written, deleted, errors} ----------------
# Exercises the loop in make_partition_writer without Spark or a live ES, by monkeypatching
# the elasticsearch module the writer imports internally. Guards the wiring (result counting
# + the yielded pandas schema) that classify_bulk_result's unit tests don't cover.

def test_partition_writer_counts_and_schema(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import elasticsearch.helpers

    # A mixed batch: 2 indexed, 1 real delete (200), 1 delete-404 (no-op), 1 index error (409),
    # 1 delete error (409). Expect written=2, deleted=1, errors=2 (the 404 is ignored).
    canned = [
        (True,  {"index":  {"status": 201}}),
        (True,  {"index":  {"status": 200}}),
        (True,  {"delete": {"status": 200}}),
        (False, {"delete": {"status": 404}}),   # no-op — must NOT count
        (False, {"index":  {"status": 409}}),   # real error
        (False, {"delete": {"status": 409}}),   # real error (non-404 delete)
    ]

    class _FakeES:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk",
                        lambda es, actions, **kw: iter(canned))

    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id")
    writer = make_partition_writer(cfg)
    # One non-empty pandas chunk drives one streaming_bulk call (its content is irrelevant here
    # since streaming_bulk is stubbed; it just needs >=1 row so the actions list is non-empty).
    out = list(writer(iter([pd.DataFrame({"id": ["a"]})])))

    assert len(out) == 1
    row = out[0].iloc[0]
    assert list(out[0].columns) == ["written", "deleted", "errors"]   # mapInPandas schema
    assert (int(row["written"]), int(row["deleted"]), int(row["errors"])) == (2, 1, 2)


def test_partition_writer_empty_chunk_yields_zeros(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import elasticsearch.helpers

    class _FakeES:
        def __init__(self, **kw):
            pass

    called = {"n": 0}
    def _stub(es, actions, **kw):
        called["n"] += 1
        return iter([])
    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", _stub)

    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id")
    writer = make_partition_writer(cfg)
    # An empty pandas chunk => no actions => streaming_bulk is skipped, counts are all zero.
    out = list(writer(iter([pd.DataFrame({"id": []})])))
    row = out[0].iloc[0]
    assert (int(row["written"]), int(row["deleted"]), int(row["errors"])) == (0, 0, 0)
    assert called["n"] == 0   # no bulk call for an empty chunk
