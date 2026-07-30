"""Unit tests for delete support: EsConfig validation + the pure bulk-result classifier.

No Spark, no ES client. The classifier is the exact rule that decides whether a
delete-404 is a silent no-op or a counted error, so it gets dedicated coverage here.
"""
import pytest

from databricks_es_connector.config import EsConfig
from databricks_es_connector.bulk import (
    classify_bulk_result, make_partition_writer, _merge_partition_results,
    ERROR_SAMPLE_CAP, WRITTEN, DELETED, IGNORED, ERROR,
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
    # A flag column with deletes off would silently do nothing, reject it.
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
    # A 404 on a NON-delete op must NOT be suppressed: suppression is scoped to deletes only.
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
        (False, {"delete": {"status": 404}}),   # no-op, must NOT count
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
    # mapInPandas schema: counts + reconciliation total + bounded error samples (JSON string)
    assert list(out[0].columns) == ["written", "deleted", "errors", "total_input", "error_samples"]
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
    assert int(row["total_input"]) == 0
    assert called["n"] == 0   # no bulk call for an empty chunk


# --- #5 reconciliation: total_input, and #2: bounded error samples ------------------------

def test_partition_writer_reports_total_input(monkeypatch):
    # total_input must equal the number of ROWS fed in (so a caller can reconcile
    # written+deleted+errors+ignored against it and detect silent chunk-level loss).
    pd = pytest.importorskip("pandas")
    import elasticsearch, elasticsearch.helpers

    canned = [(True, {"index": {"status": 201}}), (True, {"index": {"status": 201}}),
              (True, {"index": {"status": 201}})]

    class _FakeES:
        def __init__(self, **kw): pass
    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", lambda es, actions, **kw: iter(canned))

    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id")
    writer = make_partition_writer(cfg)
    out = list(writer(iter([pd.DataFrame({"id": ["a", "b", "c"]})])))
    row = out[0].iloc[0]
    assert int(row["total_input"]) == 3
    assert int(row["written"]) == 3


def test_partition_writer_captures_error_samples(monkeypatch):
    # A failed doc must leave a diagnostic breadcrumb (id, op, status, reason), not just a count.
    pd = pytest.importorskip("pandas")
    import json as _json
    import elasticsearch, elasticsearch.helpers

    canned = [
        (True,  {"index": {"status": 201, "_id": "ok1"}}),
        (False, {"index": {"status": 400, "_id": "bad1",
                           "error": {"type": "mapper_parsing_exception", "reason": "boom"}}}),
    ]

    class _FakeES:
        def __init__(self, **kw): pass
    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", lambda es, actions, **kw: iter(canned))

    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id")
    writer = make_partition_writer(cfg)
    out = list(writer(iter([pd.DataFrame({"id": ["ok1", "bad1"]})])))
    row = out[0].iloc[0]
    assert int(row["errors"]) == 1
    samples = _json.loads(row["error_samples"])
    assert len(samples) == 1
    s = samples[0]
    assert s["_id"] == "bad1" and s["op_type"] == "index" and s["status"] == 400
    assert "boom" in s["reason"]


def test_partition_writer_error_samples_are_bounded(monkeypatch):
    # Many failures must NOT blow up memory: samples are capped, but the error COUNT is exact.
    pd = pytest.importorskip("pandas")
    import json as _json
    import elasticsearch, elasticsearch.helpers
    from databricks_es_connector.bulk import ERROR_SAMPLE_CAP

    n = ERROR_SAMPLE_CAP + 25
    canned = [(False, {"index": {"status": 400, "_id": f"e{i}",
                                 "error": {"reason": f"r{i}"}}}) for i in range(n)]

    class _FakeES:
        def __init__(self, **kw): pass
    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", lambda es, actions, **kw: iter(canned))

    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="id")
    writer = make_partition_writer(cfg)
    out = list(writer(iter([pd.DataFrame({"id": [f"e{i}" for i in range(n)]})])))
    row = out[0].iloc[0]
    assert int(row["errors"]) == n                       # count is exact
    samples = _json.loads(row["error_samples"])
    assert len(samples) == ERROR_SAMPLE_CAP              # sample list is capped


# --- driver-side merge of per-partition summaries (pure, no Spark) ------------------------

def _prow(written=0, deleted=0, errors=0, total_input=0, samples=None):
    import json
    return {"written": written, "deleted": deleted, "errors": errors,
            "total_input": total_input, "error_samples": json.dumps(samples or [])}


def test_merge_sums_counts_and_totals():
    rows = [_prow(written=5, total_input=5), _prow(written=3, deleted=1, errors=1, total_input=5)]
    out = _merge_partition_results(rows)
    assert out["written"] == 8 and out["deleted"] == 1 and out["errors"] == 1
    assert out["total_input"] == 10


def test_merge_reconciliation_gap_is_visible():
    # A partition that lost rows below the per-doc level: 10 in, only 7 accounted for.
    out = _merge_partition_results([_prow(written=7, total_input=10)])
    assert out["total_input"] - (out["written"] + out["deleted"] + out["errors"]) == 3


def test_merge_concatenates_and_caps_samples():
    # Two partitions each with samples; merged list is capped at ERROR_SAMPLE_CAP.
    half = ERROR_SAMPLE_CAP
    s1 = [{"_id": f"a{i}", "op_type": "index", "status": 400, "reason": "x"} for i in range(half)]
    s2 = [{"_id": f"b{i}", "op_type": "index", "status": 400, "reason": "y"} for i in range(half)]
    out = _merge_partition_results([_prow(errors=half, samples=s1), _prow(errors=half, samples=s2)])
    assert out["errors"] == 2 * half                 # count exact
    assert len(out["error_samples"]) == ERROR_SAMPLE_CAP   # merged list capped


def test_merge_empty_is_all_zero():
    out = _merge_partition_results([])
    assert out == {"written": 0, "deleted": 0, "errors": 0, "total_input": 0, "error_samples": []}


# --- bulk_write orchestration wiring (fake Spark; symmetry with read_index's unit test) --------
# bulk_write is linear glue: sanitize_for_arrow(df) -> df.mapInPandas(writer, schema).collect() ->
# _merge_partition_results. Real Arrow/Spark behavior is covered live in the integration tier; here
# we only assert the wiring: sanitize is applied, the mapInPandas result schema is the 5-field
# contract, and the collected partition rows are merged into the final dict.

def test_bulk_write_wires_sanitize_mapinpandas_and_merge(monkeypatch):
    import databricks_es_connector.bulk as bulk_mod

    sanitized = {"marker": "post-sanitize"}
    captured = {}

    class _FakeDF:
        def mapInPandas(self, writer, schema):
            captured["writer"] = writer
            captured["schema"] = schema
            # Stand-in for the per-partition summary rows collect() would return.
            class _Mapped:
                def collect(self_inner):
                    return [_prow(written=4, total_input=5, errors=1,
                                  samples=[{"_id": "b", "op_type": "index", "status": 400,
                                            "reason": "boom"}])]
            return _Mapped()

    # sanitize_for_arrow is Spark-side; stub it to return our fake DF and prove it's applied first.
    def _fake_sanitize(df):
        captured["sanitized_in"] = df
        return _FakeDF()
    monkeypatch.setattr(bulk_mod, "sanitize_for_arrow", _fake_sanitize)

    cfg = EsConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="doc_id")
    out = bulk_mod.bulk_write("original-df", cfg)

    assert captured["sanitized_in"] == "original-df"       # sanitize applied to the caller's df
    assert captured["schema"] == \
        "written long, deleted long, errors long, total_input long, error_samples string"
    # The collected partition rows are merged into the final result dict.
    assert out["written"] == 4 and out["errors"] == 1 and out["total_input"] == 5
    assert out["error_samples"][0]["_id"] == "b"
