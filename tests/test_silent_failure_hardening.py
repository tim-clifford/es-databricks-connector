"""Regression tests for the 0.6.0 silent-failure hardening. No Spark, no live ES.

Each test here pins a behavior that used to fail SILENTLY: the write succeeded-looking while data
was lost, coerced, or sent somewhere unintended. Every one of these was verified red against the
0.5.0 implementation before the fix landed, so they are regression guards, not confirmations.

Grouped by the failure they prevent:
  1. a rejected micro-batch advancing the Structured Streaming checkpoint (the worst one)
  2. per-document 429s never being retried
  3. reconciliation being unable to tell a delete-404 no-op from a lost row
 10. a typo'd index name auto-creating a dynamically-mapped index
 11/12. read_coerce returning plausible-looking wrong values on a schema mismatch
 13. a misspelled drop_fields entry silently pruning nothing (PII egress control failing open)
 15. inf/-inf/NaN becoming JSON null with no count
 16. an ambiguous delete-flag string silently meaning "not a delete"
"""
import json

import pytest

import databricks_es_connector.bulk as bulk_mod
import databricks_es_connector.stream as stream_mod
from databricks_es_connector.bulk import (
    EsWriteError, _merge_partition_results, make_partition_writer, reconcile_or_raise,
)
from databricks_es_connector.config import EsConfig, EsReadConfig, EsWriteConfig
from databricks_es_connector.read_transform import ReadSchemaMismatch, read_coerce
from databricks_es_connector.stream import IGNORE, LOG, RAISE, make_foreach_batch
from databricks_es_connector.transform import AmbiguousDeleteFlag, build_action, coerce_value


def _cfg(**kw):
    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="doc_id")
    base.update(kw)
    return EsConfig(**base)


class _FakeDF:
    """Stand-in for a micro-batch DataFrame: only isEmpty() is exercised by the helper."""

    def __init__(self, empty=False):
        self._empty = empty

    def isEmpty(self):
        return self._empty


def _result(**kw):
    base = {"written": 0, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
            "total_input": 0, "unaccounted": 0, "error_samples": []}
    base.update(kw)
    return base


# =====================================================================================
# Item 1: a failed micro-batch must FAIL, so the checkpoint does not advance past lost rows
# =====================================================================================

def test_streaming_raises_when_every_doc_is_rejected(monkeypatch):
    # THE bug. foreachBatch returning normally is what commits the checkpoint offset, so a batch
    # where ES rejected all 1000 docs used to be recorded as a success and the rows were never
    # retried. It must now raise.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(errors=1000, total_input=1000))
    fb = make_foreach_batch(_cfg())
    with pytest.raises(EsWriteError, match="1000 document"):
        fb(_FakeDF(), 42)


def test_streaming_raises_on_partial_rejection(monkeypatch):
    # Not just the all-or-nothing case: one rejected doc is still silent loss of that row.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=999, errors=1, total_input=1000))
    fb = make_foreach_batch(_cfg())
    with pytest.raises(EsWriteError):
        fb(_FakeDF(), 7)


def test_streaming_raises_on_unaccounted_rows(monkeypatch):
    # Loss BELOW the per-doc level (a chunk-level transport failure): errors==0 but rows vanished.
    # The per-doc error count structurally cannot see this, which is why unaccounted exists.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=7, total_input=10, unaccounted=3))
    fb = make_foreach_batch(_cfg())
    with pytest.raises(EsWriteError, match="unaccounted"):
        fb(_FakeDF(), 1)


def test_streaming_error_carries_the_result_for_a_dead_letter_path(monkeypatch):
    # A caller catching the error still needs the counts and samples (e.g. to write a DLQ row).
    res = _result(written=1, errors=2, total_input=3,
                  error_samples=[{"_id": "x", "op_type": "index", "status": 400, "reason": "boom"}])
    monkeypatch.setattr(stream_mod, "bulk_write", lambda df, cfg: res)
    fb = make_foreach_batch(_cfg())
    with pytest.raises(EsWriteError) as exc:
        fb(_FakeDF(), 3)
    assert exc.value.result["errors"] == 2
    assert exc.value.result["error_samples"][0]["_id"] == "x"


def test_streaming_does_not_raise_on_a_clean_batch(monkeypatch):
    # The guard must not fire on success, or every healthy stream breaks.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=500, total_input=500))
    make_foreach_batch(_cfg())(_FakeDF(), 0)   # returns cleanly


def test_streaming_does_not_raise_on_expected_delete_404s(monkeypatch):
    # A CDF replay deleting already-absent docs is the COMMON case; it must not fail the stream.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=1, deleted=2, ignored=5, total_input=8))
    make_foreach_batch(_cfg())(_FakeDF(), 0)   # returns cleanly


def test_streaming_on_batch_still_sees_a_failing_batch(monkeypatch):
    # The metrics hook may BE the dead-letter path, so it must run before the raise.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(errors=3, total_input=3))
    seen = []
    fb = make_foreach_batch(_cfg(), on_batch=lambda bid, r: seen.append((bid, r["errors"])))
    with pytest.raises(EsWriteError):
        fb(_FakeDF(), 11)
    assert seen == [(11, 3)], "on_batch must observe the failure before it raises"


def test_streaming_on_error_log_advances_and_warns(monkeypatch, caplog):
    # The documented opt-out. It must be loud about the consequence: rows are NOT retried.
    import logging
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(errors=4, total_input=4))
    fb = make_foreach_batch(_cfg(), on_error=LOG)
    with caplog.at_level(logging.WARNING, logger="databricks_es_connector.stream"):
        fb(_FakeDF(), 5)     # does NOT raise
    assert any("will not be retried" in r.message.lower().replace("_", " ") or
               "WILL advance" in r.message for r in caplog.records), caplog.text


def test_streaming_on_error_ignore_is_silent(monkeypatch, caplog):
    # Restores pre-0.6.0 behavior for a deliberately loss-tolerant pipeline. No raise, no log.
    import logging
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(errors=4, total_input=4))
    fb = make_foreach_batch(_cfg(), on_error=IGNORE)
    with caplog.at_level(logging.WARNING, logger="databricks_es_connector.stream"):
        fb(_FakeDF(), 5)
    assert caplog.records == []


def test_streaming_rejects_an_unknown_on_error_policy():
    # A typo'd policy must not silently fall through to the unsafe path.
    with pytest.raises(ValueError, match="on_error"):
        make_foreach_batch(_cfg(), on_error="warn")


def test_streaming_empty_batch_result_has_the_new_keys(monkeypatch):
    # The empty-batch stand-in must carry the same keys a real result does, including the new ones,
    # so a callback reading result["unaccounted"] never KeyErrors on an empty micro-batch.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: pytest.fail("must not write an empty batch"))
    seen = []
    make_foreach_batch(_cfg(), on_batch=lambda b, r: seen.append(r))(_FakeDF(empty=True), 0)
    for key in ("written", "deleted", "errors", "ignored", "coerced_nonfinite", "total_input",
                "unaccounted", "error_samples"):
        assert key in seen[0], f"empty-batch dict missing {key}"
    assert seen[0]["empty"] is True


# =====================================================================================
# Item 3 / 3a: reconciliation, and the ignored count that makes it usable
# =====================================================================================

def test_reconcile_passes_a_clean_result():
    res = _result(written=10, total_input=10)
    assert reconcile_or_raise(res) is res    # returned unchanged so it can be used inline


def test_reconcile_raises_on_errors():
    with pytest.raises(EsWriteError, match="rejected"):
        reconcile_or_raise(_result(written=9, errors=1, total_input=10))


def test_reconcile_raises_on_unaccounted():
    with pytest.raises(EsWriteError, match="unaccounted"):
        reconcile_or_raise(_result(written=7, total_input=10, unaccounted=3))


def test_reconcile_ignores_delete_404_noops():
    # Without the `ignored` count this would look like 2 lost rows and fire spuriously.
    reconcile_or_raise(_result(written=1, ignored=2, total_input=3))


def test_reconcile_message_names_the_index_and_shows_samples():
    with pytest.raises(EsWriteError) as exc:
        reconcile_or_raise(_result(errors=1, total_input=1,
                                   error_samples=[{"_id": "bad", "reason": "mapper_parsing"}]),
                           index="my-index")
    msg = str(exc.value)
    assert "my-index" in msg and "bad" in msg


def test_negative_unaccounted_does_not_raise():
    # Defensive: more outcomes than inputs would be a counting bug, not data loss. Don't fail a
    # healthy write over it (the streaming path would then be unusable until a library fix).
    reconcile_or_raise(_result(written=11, total_input=10, unaccounted=-1))


# =====================================================================================
# Item 2: per-document retries for retryable rejections (429)
# =====================================================================================

def test_writer_passes_per_doc_retry_settings_to_streaming_bulk(monkeypatch):
    # streaming_bulk's own default is max_retries=0, so an item-level 429 got ZERO retries. The
    # transport-level EsConnection.max_retries does not cover it: _bulk returns HTTP 200 even when
    # individual documents fail, so the transport retry never sees them.
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import elasticsearch.helpers

    captured = {}

    class _FakeES:
        def __init__(self, **kw):
            pass

    def _stub(es, actions, **kw):
        captured.update(kw)
        return iter([(True, {"index": {"status": 201}})])

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", _stub)

    writer = make_partition_writer(_cfg(max_retries_per_doc=5, retry_on_doc_status=(429, 503)))
    list(writer(iter([pd.DataFrame({"doc_id": ["a"]})])))

    assert captured["max_retries"] == 5
    assert tuple(captured["retry_on_status"]) == (429, 503)


def test_config_default_retries_per_doc_is_nonzero():
    # The whole point: the default must not be elasticsearch-py's 0.
    assert _cfg().max_retries_per_doc == 3
    assert 429 in _cfg().retry_on_doc_status


def test_config_rejects_negative_retries():
    with pytest.raises(ValueError, match="max_retries_per_doc"):
        _cfg(max_retries_per_doc=-1)


def test_config_rejects_retries_with_no_retryable_status():
    # Would silently never retry despite asking for retries.
    with pytest.raises(ValueError, match="retry_on_doc_status"):
        _cfg(max_retries_per_doc=3, retry_on_doc_status=())


def test_warns_when_transport_retries_raised_but_per_doc_retries_off():
    # `max_retries` and `max_retries_per_doc` are one character apart and mean different things.
    # Someone hardening against ES backpressure by raising the transport knob has almost certainly
    # not intended to leave rejected documents with zero retries. Documenting the difference is not
    # enough: the config itself has to say so.
    with pytest.warns(UserWarning, match="max_retries_per_doc"):
        _cfg(max_retries=10, max_retries_per_doc=0)


@pytest.mark.parametrize("kw", [
    {},                                              # defaults
    {"max_retries": 10, "max_retries_per_doc": 5},   # both raised: coherent
    {"max_retries_per_doc": 0},                      # per-doc off, transport at default
    {"max_retries": 1, "max_retries_per_doc": 0},    # per-doc off, transport LOWERED: deliberate
])
def test_no_spurious_retry_warning(kw, recwarn):
    # A warning that fires on sane configurations trains people to ignore warnings.
    _cfg(**kw)
    assert [w for w in recwarn if "max_retries_per_doc" in str(w.message)] == []


# =====================================================================================
# Item 13: a misspelled drop_fields entry must fail CLOSED (it is a PII/egress control)
# =====================================================================================

class _ColsDF:
    def __init__(self, cols):
        self.columns = list(cols)


def test_preflight_rejects_unknown_drop_field():
    # 'ssnn' is a typo for 'ssn': the field would ship to ES anyway, with the caller believing it
    # was withheld. Fail rather than silently prune nothing.
    cfg = _cfg(drop_fields=("ssnn",), require_existing_index=False)
    with pytest.raises(ValueError, match="drop_fields"):
        bulk_mod._preflight(_ColsDF(["doc_id", "ssn", "name"]), cfg)


def test_preflight_error_lists_the_available_columns():
    # Actionable: the caller needs to see the name they meant.
    cfg = _cfg(drop_fields=("ssnn",), require_existing_index=False)
    with pytest.raises(ValueError) as exc:
        bulk_mod._preflight(_ColsDF(["doc_id", "ssn"]), cfg)
    assert "ssn" in str(exc.value)


def test_preflight_accepts_correct_drop_fields():
    cfg = _cfg(drop_fields=("ssn",), require_existing_index=False)
    bulk_mod._preflight(_ColsDF(["doc_id", "ssn", "name"]), cfg)   # no raise


def test_preflight_skips_drop_field_check_when_not_strict():
    # The documented opt-out for one config reused across differing schemas.
    cfg = _cfg(drop_fields=("nope",), strict_drop_fields=False, require_existing_index=False)
    bulk_mod._preflight(_ColsDF(["doc_id"]), cfg)   # no raise


# =====================================================================================
# Item 10: a missing index must not be silently auto-created by ES
# =====================================================================================

class _FakeIndices:
    def __init__(self, exists):
        self._exists = exists
        self.asked = []

    def exists(self, index):
        self.asked.append(index)
        return self._exists


class _FakeESClient:
    last = None

    def __init__(self, exists):
        self.indices = _FakeIndices(exists)
        self.closed = False

    def close(self):
        self.closed = True


def _patch_es(monkeypatch, exists):
    import elasticsearch
    holder = {}

    def _factory(**kw):
        client = _FakeESClient(exists)
        holder["client"] = client
        return client

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _factory)
    return holder


def test_preflight_rejects_a_missing_index(monkeypatch):
    # ES auto-creates a missing index (action.auto_create_index defaults true), so a TYPO'd name
    # produced a new dynamically-mapped index and a perfect `written` count.
    _patch_es(monkeypatch, exists=False)
    with pytest.raises(ValueError, match="does not exist"):
        bulk_mod._preflight(_ColsDF(["doc_id"]), _cfg(index="typoo-index"))


def test_preflight_missing_index_error_explains_the_risk(monkeypatch):
    _patch_es(monkeypatch, exists=False)
    with pytest.raises(ValueError) as exc:
        bulk_mod._preflight(_ColsDF(["doc_id"]), _cfg(index="typoo-index"))
    msg = str(exc.value)
    assert "typoo-index" in msg
    assert "require_existing_index=False" in msg      # names the escape hatch


def test_preflight_accepts_an_existing_index(monkeypatch):
    holder = _patch_es(monkeypatch, exists=True)
    bulk_mod._preflight(_ColsDF(["doc_id"]), _cfg(index="real-index"))
    assert holder["client"].indices.asked == ["real-index"]
    assert holder["client"].closed, "the driver-side client must be closed, not leaked"


def test_preflight_skips_index_check_when_disabled(monkeypatch):
    # No client should even be built when the caller opted out.
    import elasticsearch

    def _boom(**kw):
        raise AssertionError("must not build a client when require_existing_index=False")

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _boom)
    bulk_mod._preflight(_ColsDF(["doc_id"]), _cfg(require_existing_index=False))


# =====================================================================================
# Item 15: inf / -inf / NaN silently becoming JSON null must be COUNTED
# =====================================================================================

def test_coerce_value_counts_infinity():
    stats = {}
    assert coerce_value(float("inf"), stats) is None
    assert coerce_value(float("-inf"), stats) is None
    assert stats["coerced_nonfinite"] == 2


def test_coerce_value_counts_nan():
    # A NaN is a real value that became null, unlike a None that was already null.
    stats = {}
    assert coerce_value(float("nan"), stats) is None
    assert stats["coerced_nonfinite"] == 1


def test_coerce_value_does_not_count_a_genuine_none():
    stats = {}
    assert coerce_value(None, stats) is None
    assert stats.get("coerced_nonfinite", 0) == 0


def test_coerce_value_counts_nested_nonfinite():
    # A divide-by-zero inside a struct/array is just as invisible as one at the top level.
    stats = {}
    coerce_value({"a": [1.0, float("inf")], "b": {"c": float("-inf")}}, stats)
    assert stats["coerced_nonfinite"] == 2


def test_coerce_value_without_stats_still_works():
    # The counter is optional; the pure transform must keep its old signature behavior.
    assert coerce_value(float("inf")) is None


def test_build_action_threads_stats_through():
    stats = {}
    build_action({"doc_id": "a", "ratio": float("inf")}, index="i", id_field="doc_id", stats=stats)
    assert stats["coerced_nonfinite"] == 1


def test_writer_reports_coerced_nonfinite(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import elasticsearch.helpers

    class _FakeES:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk",
                        lambda es, actions, **kw: iter([(True, {"index": {"status": 201}})]))

    writer = make_partition_writer(_cfg())
    out = list(writer(iter([pd.DataFrame({"doc_id": ["a"], "ratio": [float("inf")]})])))
    assert int(out[0].iloc[0]["coerced_nonfinite"]) == 1


# =====================================================================================
# Item 16: an ambiguous delete-flag string must raise, not quietly mean "not a delete"
# =====================================================================================

@pytest.mark.parametrize("flag", ["on", "enabled", "delete", "2", "-1", "yep", "0.0"])
def test_ambiguous_delete_flag_raises(flag):
    # Each of these reads as truthy to a human but used to mean "keep the document", leaving a doc
    # in ES that should have been deleted, with no error and no count.
    with pytest.raises(AmbiguousDeleteFlag):
        build_action({"doc_id": "a", "_is_delete": flag}, index="i", id_field="doc_id",
                     has_deletes=True, delete_flag_column="_is_delete")


@pytest.mark.parametrize("flag", ["true", "True", "TRUE", " t ", "1", "yes", "Y"])
def test_recognized_true_strings_still_delete(flag):
    action = build_action({"doc_id": "a", "_is_delete": flag}, index="i", id_field="doc_id",
                          has_deletes=True, delete_flag_column="_is_delete")
    assert action["_op_type"] == "delete"


@pytest.mark.parametrize("flag", ["false", "False", "f", "0", "no", "N", "", "   "])
def test_recognized_false_strings_still_index(flag):
    action = build_action({"doc_id": "a", "_is_delete": flag}, index="i", id_field="doc_id",
                          has_deletes=True, delete_flag_column="_is_delete")
    assert action.get("_op_type") != "delete"


def test_null_delete_flag_is_not_a_delete():
    # A missing flag must never be read as a delete.
    action = build_action({"doc_id": "a", "_is_delete": None}, index="i", id_field="doc_id",
                          has_deletes=True, delete_flag_column="_is_delete")
    assert action.get("_op_type") != "delete"


def test_boolean_delete_flag_is_unambiguous():
    for value, is_delete in ((True, True), (False, False), (1, True), (0, False)):
        action = build_action({"doc_id": "a", "_is_delete": value}, index="i", id_field="doc_id",
                              has_deletes=True, delete_flag_column="_is_delete")
        assert (action.get("_op_type") == "delete") is is_delete


# =====================================================================================
# Items 11 / 12: read_coerce must not return plausible-looking WRONG values
# =====================================================================================

def test_multivalued_field_declared_scalar_raises():
    # ES has no array type: ANY field can hold a list under the same mapping, and the mapping gives
    # no hint. This used to return the literal string "['prod', 'urgent']".
    with pytest.raises(ReadSchemaMismatch, match="multiple values"):
        read_coerce(["prod", "urgent"], "string")


def test_multivalue_error_names_the_array_fix():
    with pytest.raises(ReadSchemaMismatch, match=r"array<string>"):
        read_coerce(["a", "b"], "string")


def test_object_declared_scalar_raises():
    # Used to return "{'nested': 'object'}".
    with pytest.raises(ReadSchemaMismatch, match="object"):
        read_coerce({"nested": "object"}, "string")


def test_non_integral_float_declared_int_raises():
    # Used to silently truncate 3.7 -> 3.
    with pytest.raises(ReadSchemaMismatch, match="not an integer"):
        read_coerce(3.7, "int")


def test_integral_float_declared_int_is_allowed():
    # 3.0 -> 3 is exact and legitimate: JSON has one number type, so an integer often arrives as a
    # float. Only a LOSSY conversion is rejected.
    assert read_coerce(3.0, "int") == 3
    assert read_coerce(-7.0, "long") == -7


def test_declared_array_still_reads_a_list():
    # The correct declaration keeps working, including ES's scalar-as-one-element case.
    assert read_coerce(["prod", "urgent"], ("array", "string")) == ["prod", "urgent"]
    assert read_coerce("prod", ("array", "string")) == ["prod"]


@pytest.mark.parametrize("token", ["timestamp", "timestamp_ntz", "date", "binary",
                                   "decimal(10,2)", "long", "double", "boolean"])
def test_every_scalar_token_rejects_a_list(token):
    # The guard must be uniform: a multi-valued field is a mismatch for EVERY scalar type, not just
    # the string branch that happened to stringify it.
    with pytest.raises(ReadSchemaMismatch):
        read_coerce(["a", "b"], token)


def test_nulls_are_still_none_for_every_type():
    # A missing/null field must stay None, not trip the new guards.
    for token in ("string", "long", "timestamp", ("array", "string"), ("struct", [("a", "long")])):
        assert read_coerce(None, token) is None


def test_scalar_numbers_still_stringify_for_a_string_column():
    # Only containers are rejected; a number declared string is a meaningful, lossless rendering.
    assert read_coerce(42, "string") == "42"
    assert read_coerce(True, "string") == "True"


# =====================================================================================
# Item 14: the serial-read fallback must not be silent (EsReadConfig plumbing)
# =====================================================================================

def test_read_config_defaults_to_strict_slices():
    assert EsReadConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i").strict_slices


def test_read_config_allows_opting_out_of_strict_slices():
    cfg = EsReadConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i",
                       strict_slices=False)
    assert cfg.strict_slices is False
