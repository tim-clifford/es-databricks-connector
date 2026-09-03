"""Unit tests for the streaming foreachBatch helper. No Spark, no ES.

make_foreach_batch is thin glue: skip empty micro-batches, otherwise delegate to bulk_write
and (optionally) report the result to on_batch. We stub bulk_write and use a fake DataFrame
with a controllable isEmpty(), so the glue (empty-skip, delegation, and the callback contract)
is testable without a session.

The key contract under test: the dict handed to on_batch has the SAME keys whether the batch
was empty or not (written/deleted/errors/total_input/error_samples), so a caller reading e.g.
result["total_input"] never KeyErrors on an empty batch. That inconsistency was the bug this
covers.
"""
import databricks_es_connector.stream as stream_mod
from databricks_es_connector.config import EsConfig
from databricks_es_connector.stream import make_foreach_batch


class _FakeDF:
    """Stand-in for a Spark DataFrame: only isEmpty() is exercised by the helper."""
    def __init__(self, empty: bool):
        self._empty = empty

    def isEmpty(self) -> bool:
        return self._empty


def _cfg(**kw):
    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="doc_id")
    base.update(kw)
    return EsConfig(**base)


# --- the keys a bulk_write result (and the empty-batch stand-in) must always carry ---
_RESULT_KEYS = {"written", "deleted", "errors", "total_input", "error_samples"}


def test_empty_batch_skips_bulk_write_and_reports_full_shape(monkeypatch):
    # An empty micro-batch must NOT call bulk_write, and the callback dict must carry the same
    # keys a real result does (this is the regression: total_input/error_samples were missing).
    called = {"bulk": 0}
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg, batch_id=None: called.__setitem__("bulk", called["bulk"] + 1) or {})

    seen = []
    fb = make_foreach_batch(_cfg(), on_batch=lambda bid, r: seen.append((bid, r)))
    fb(_FakeDF(empty=True), 7)

    assert called["bulk"] == 0, "bulk_write must not run on an empty batch"
    assert len(seen) == 1
    bid, result = seen[0]
    assert bid == 7
    assert _RESULT_KEYS <= set(result), f"empty-batch dict missing keys: {_RESULT_KEYS - set(result)}"
    assert result["empty"] is True
    assert result["total_input"] == 0 and result["error_samples"] == []


def test_empty_batch_without_callback_is_a_noop(monkeypatch):
    # No on_batch => nothing to call; must not touch bulk_write or raise.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg, batch_id=None: (_ for _ in ()).throw(AssertionError("should not run")))
    fb = make_foreach_batch(_cfg())
    fb(_FakeDF(empty=True), 0)   # returns cleanly


def test_non_empty_batch_delegates_to_bulk_write(monkeypatch):
    # A non-empty batch calls bulk_write(df, cfg, batch_id=...) and forwards its result to on_batch.
    result = {"written": 3, "deleted": 1, "errors": 0, "total_input": 4, "error_samples": []}
    captured = {}
    def _fake_bulk(df, cfg, batch_id=None):
        captured["df"] = df
        captured["cfg"] = cfg
        captured["batch_id"] = batch_id
        return result
    monkeypatch.setattr(stream_mod, "bulk_write", _fake_bulk)

    seen = []
    cfg = _cfg()
    df = _FakeDF(empty=False)
    fb = make_foreach_batch(cfg, on_batch=lambda bid, r: seen.append((bid, r)))
    fb(df, 42)

    assert captured["df"] is df and captured["cfg"] is cfg   # passed through unchanged
    assert captured["batch_id"] == 42                        # micro-batch id forwarded for the log table
    assert seen == [(42, result)]                            # result forwarded verbatim


def test_non_empty_batch_without_callback_still_writes(monkeypatch):
    # Even with no on_batch, a non-empty batch must be written (the callback is optional metrics,
    # not the write trigger).
    called = {"bulk": 0}
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg, batch_id=None: called.__setitem__("bulk", called["bulk"] + 1) or {})
    fb = make_foreach_batch(_cfg())
    fb(_FakeDF(empty=False), 1)
    assert called["bulk"] == 1


def test_empty_and_nonempty_callback_dicts_share_the_contract_keys(monkeypatch):
    # The heart of the fix: whatever a caller reads off the result must be present in BOTH the
    # empty-batch dict and a real bulk_write result. Assert the empty-batch keys are a superset of
    # the required contract keys (it also adds `empty`, which non-empty results don't have).
    real_result = {"written": 5, "deleted": 0, "errors": 0, "total_input": 5, "error_samples": []}
    monkeypatch.setattr(stream_mod, "bulk_write", lambda df, cfg, batch_id=None: real_result)

    seen = []
    fb = make_foreach_batch(_cfg(), on_batch=lambda bid, r: seen.append(r))
    fb(_FakeDF(empty=True), 0)    # empty
    fb(_FakeDF(empty=False), 1)   # non-empty

    empty_dict, nonempty_dict = seen
    assert _RESULT_KEYS <= set(empty_dict)
    assert _RESULT_KEYS <= set(nonempty_dict)
