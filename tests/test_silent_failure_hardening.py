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
import inspect
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
    # `transport_max_retries` and `max_retries_per_doc` mean different things (a whole HTTP request
    # vs one rejected document). Someone hardening against ES backpressure by raising the transport
    # knob has almost certainly not intended to leave rejected documents with zero retries.
    # Documenting the difference is not enough: the config itself has to say so.
    with pytest.warns(UserWarning, match="max_retries_per_doc"):
        _cfg(transport_max_retries=10, max_retries_per_doc=0)


@pytest.mark.parametrize("kw", [
    {},                                                        # defaults
    {"transport_max_retries": 10, "max_retries_per_doc": 5},    # both raised: coherent
    {"max_retries_per_doc": 0},                                 # per-doc off, transport at default
    {"transport_max_retries": 1, "max_retries_per_doc": 0},     # per-doc off, transport LOWERED
])
def test_no_spurious_retry_warning(kw, recwarn):
    # A warning that fires on sane configurations trains people to ignore warnings.
    _cfg(**kw)
    assert [w for w in recwarn if "max_retries_per_doc" in str(w.message)] == []


# --- the two retry layers: umbrella vs granular (0.6.0) ----------------------------------------
# `max_retries` is a supported convenience that sets EVERY layer at once; the per-layer fields tune
# them independently. Mixing the two is rejected so the effective count never depends on an invisible
# precedence rule.

def test_transport_max_retries_reaches_the_client():
    # Our field is `transport_max_retries`; the elasticsearch-py client kwarg is still `max_retries`.
    assert _cfg(transport_max_retries=7).client_kwargs()["max_retries"] == 7


def test_umbrella_sets_every_layer():
    cfg = _cfg(max_retries=5)
    assert cfg.transport_max_retries == 5
    assert cfg.max_retries_per_doc == 5
    assert cfg.client_kwargs()["max_retries"] == 5


def test_umbrella_is_not_a_deprecated_alias():
    # It is a first-class way to configure retries, so it must NOT warn.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # any warning becomes an exception
        _cfg(max_retries=5)


def test_granular_layers_are_independent():
    cfg = _cfg(transport_max_retries=5, max_retries_per_doc=2)
    assert (cfg.transport_max_retries, cfg.max_retries_per_doc) == (5, 2)


@pytest.mark.parametrize("kw", [
    {"max_retries": 5, "max_retries_per_doc": 2},                             # umbrella + per-doc
    {"max_retries": 5, "transport_max_retries": 1},                           # umbrella + transport
    {"max_retries": 5, "transport_max_retries": 1, "max_retries_per_doc": 2},  # umbrella + both
])
def test_mixing_umbrella_and_granular_is_rejected(kw):
    # Accepting both would make the effective value depend on precedence nobody can see at the call
    # site. The error names the conflicting fields and the equivalent granular form.
    with pytest.raises(ValueError, match="not both"):
        _cfg(**kw)


def test_mixing_error_names_the_conflicting_field():
    with pytest.raises(ValueError) as exc:
        _cfg(max_retries=5, max_retries_per_doc=2)
    msg = str(exc.value)
    assert "max_retries_per_doc" in msg and "max_retries=5" in msg


def test_reading_max_retries_when_layers_agree():
    assert _cfg(max_retries=4).max_retries == 4
    assert _cfg().max_retries == 3                 # both defaults are 3


def test_reading_max_retries_is_none_when_layers_differ():
    # Honest rather than arbitrary: under a granular config there is no single number to report, so
    # do not pick one of the two and imply it describes both.
    assert _cfg(transport_max_retries=5, max_retries_per_doc=2).max_retries is None


def test_max_retries_is_not_a_stored_field():
    # It expands into the real fields at construction, so no stored value can drift from the layers.
    import dataclasses
    assert "max_retries" not in {f.name for f in dataclasses.fields(_cfg())}


def test_a_further_subclass_loses_the_umbrella_but_fails_loudly():
    # Documented boundary: @dataclass regenerates __init__ for every subclass, so a caller's own
    # subclass is not wrapped. What matters is that it fails LOUDLY (TypeError) rather than silently
    # ignoring the argument and leaving retries at their defaults, and that the per-layer fields
    # still work. Pinned so the failure mode cannot degrade into a silent one.
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _Custom(EsWriteConfig):
        extra: str = "x"

    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i")
    with pytest.raises(TypeError, match="max_retries"):
        _Custom(**base, max_retries=7)
    granular = _Custom(**base, transport_max_retries=7, max_retries_per_doc=7)
    assert (granular.transport_max_retries, granular.max_retries_per_doc) == (7, 7)


@pytest.mark.parametrize("name", ["EsConnection", "EsWriteConfig", "EsReadConfig"])
def test_max_retries_is_advertised_in_the_signature(name):
    # Because it is not a dataclass field, the generated __init__ signature that functools.wraps
    # copies does not mention it -- so help(), IDE completion and any introspecting tool would hide a
    # supported argument. The wrapper appends it explicitly.
    import databricks_es_connector.config as cfg_mod
    params = inspect.signature(getattr(cfg_mod, name)).parameters
    assert "max_retries" in params, f"{name} does not advertise max_retries"
    assert params["max_retries"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name,extra", [
    ("EsConnection", {}),
    ("EsWriteConfig", {"index": "i"}),
    ("EsReadConfig", {"index": "i"}),
])
def test_umbrella_works_on_every_config_class(name, extra):
    # Each subclass regenerates its own __init__, so the umbrella has to be applied to all three or a
    # subclass silently loses it.
    import databricks_es_connector.config as cfg_mod
    obj = getattr(cfg_mod, name)(hosts="https://h:9200", basic_auth=("u", "p"), **extra,
                                 max_retries=9)
    assert obj.transport_max_retries == 9


def test_umbrella_on_read_config_covers_only_the_layers_it_has():
    # A read has no per-document layer at all (there are no documents being written to retry), so the
    # umbrella must not invent one, and the conflict message must name only the real field.
    from databricks_es_connector import EsReadConfig
    cfg = EsReadConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", max_retries=7)
    assert cfg.transport_max_retries == 7
    assert not hasattr(cfg, "max_retries_per_doc")
    assert cfg.max_retries == 7
    with pytest.raises(ValueError, match="transport_max_retries"):
        EsReadConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i",
                     max_retries=7, transport_max_retries=1)


def test_configs_are_still_frozen_dataclasses_after_the_wrapper():
    # The umbrella wraps the GENERATED __init__ after @dataclass, so the dataclass machinery
    # (frozen, fields, defaults, pickling) must be intact. Pickling matters: these configs are
    # captured in closures and shipped to Spark executors.
    import dataclasses
    import pickle
    cfg = _cfg(max_retries=6)
    assert dataclasses.is_dataclass(cfg)
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    assert isinstance(hash(cfg), int)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.transport_max_retries = 1


def test_negative_transport_retries_rejected():
    with pytest.raises(ValueError, match="transport_max_retries"):
        _cfg(transport_max_retries=-1)


def test_negative_umbrella_rejected():
    # The umbrella must not bypass per-layer validation.
    with pytest.raises(ValueError):
        _cfg(max_retries=-1)


def test_retry_warning_points_at_the_caller_not_the_library():
    # A warning whose filename/lineno land inside config.py blames the library for the caller's
    # configuration and is useless for finding the offending line. Wrapping __init__ for the umbrella
    # added a stack frame, which silently made the old stacklevel point inside config.py; this pins
    # the correct one so that regression cannot come back unnoticed.
    import warnings
    # Constructed DIRECTLY here, not via the _cfg() helper: the helper would itself be the immediate
    # caller, so the warning would correctly name the helper's line and this test could not tell a
    # right stacklevel from a wrong one.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EsWriteConfig(hosts="https://h:9200", basic_auth=("u", "p"), index="i", id_field="doc_id",
                      transport_max_retries=10, max_retries_per_doc=0)
        expected_line = inspect.currentframe().f_lineno - 2   # the EsWriteConfig( line

    relevant = [w for w in caught if w.category is UserWarning]
    assert relevant, "expected a UserWarning"
    w = relevant[0]
    assert w.filename == __file__, f"warning blamed {w.filename}, not the caller's file"
    assert w.lineno == expected_line, f"warning blamed line {w.lineno}, expected {expected_line}"


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


# =====================================================================================
# Item 17: a config field naming a MISSING column must fail closed (the class, not one case)
#
# `drop_fields` was hardened in 0.6.0 (item 13) but the other two column-naming fields were not.
# `delete_flag_column` was the live data-loss case: with has_deletes=True and a misspelled flag
# column, `row.get(flag)` returns None for EVERY row, so no row is ever routed to a delete and
# every intended deletion is applied as an UPSERT instead. Proven live against real Spark + ES:
# deleted=0, errors=0, unaccounted=0, raise_on_error=True passed clean, and the documents that
# should have been erased were still in the index (with the flag column indexed alongside them).
#
# This is invisible below _preflight by construction: a column that is PRESENT with a null value
# must not be a delete (test_null_flag_is_not_a_delete pins that), and `.get()` cannot tell that
# apart from a column that is ABSENT. `_preflight` is the only layer that sees df.columns, so it is
# the only place the two can be distinguished.
# =====================================================================================

def test_preflight_rejects_missing_delete_flag_column():
    # The live silent failure: every intended delete silently became an upsert.
    cfg = _cfg(has_deletes=True, delete_flag_column="_is_deleted",   # typo: real column is _is_delete
               require_existing_index=False)
    with pytest.raises(ValueError, match="delete_flag_column"):
        bulk_mod._preflight(_ColsDF(["doc_id", "n", "_is_delete"]), cfg)


def test_preflight_delete_flag_error_is_actionable():
    cfg = _cfg(has_deletes=True, delete_flag_column="_is_deleted", require_existing_index=False)
    with pytest.raises(ValueError) as exc:
        bulk_mod._preflight(_ColsDF(["doc_id", "_is_delete"]), cfg)
    msg = str(exc.value)
    assert "_is_deleted" in msg and "_is_delete" in msg      # names the typo AND the real columns
    assert "upsert" in msg or "delete" in msg                # says what goes wrong


def test_preflight_accepts_a_present_delete_flag_column():
    cfg = _cfg(has_deletes=True, delete_flag_column="_is_delete", require_existing_index=False)
    bulk_mod._preflight(_ColsDF(["doc_id", "_is_delete"]), cfg)   # must not raise


def test_preflight_ignores_delete_flag_when_deletes_are_off():
    # has_deletes=False means the flag column is irrelevant (the config forbids setting it anyway).
    cfg = _cfg(require_existing_index=False)
    bulk_mod._preflight(_ColsDF(["doc_id"]), cfg)                 # must not raise


def test_preflight_rejects_missing_id_field():
    # A missing id_field raises per-row in _require_id today, mid-write, after some partitions may
    # already have committed. Preflight turns it into one clear driver-side failure before any write.
    cfg = _cfg(id_field="docid", require_existing_index=False)     # typo: real column is doc_id
    with pytest.raises(ValueError, match="id_field"):
        bulk_mod._preflight(_ColsDF(["doc_id", "n"]), cfg)


def test_preflight_covers_every_column_naming_config_field():
    """The class-closure guard: this test FAILS when a new column-naming field is added.

    The 0.6.0 hardening fixed `drop_fields` and left `delete_flag_column` open because nothing
    enumerated the class. This pins the enumeration, so a fourth field cannot be added without a
    deliberate decision about whether _preflight must validate it.
    """
    assert bulk_mod._COLUMN_NAMING_FIELDS == ("id_field", "drop_fields", "delete_flag_column")


# =====================================================================================
# Item 18: read_index must reject declared types it cannot honor (char/varchar/void)
#
# These produce simpleString() tokens ('varchar(10)', 'char(5)', 'void') that no read_coerce branch
# matches, so the value falls through the unknown-token passthrough untouched -- bypassing the
# _reject_non_scalar guard that StringType correctly applies. Proven live: a multi-valued ES field
# declared varchar reached Spark as a Python list and died with
# `PySparkNotImplementedError: Invalid return type in mapInPandas`, an error that names neither the
# field nor the real problem.
# =====================================================================================

def test_read_coerce_rejects_varchar_and_char():
    for token in ("varchar(10)", "char(5)"):
        with pytest.raises(ReadSchemaMismatch, match="StringType"):
            read_coerce("abc", token)


def test_read_coerce_rejects_void():
    # NullType: Spark cannot carry it through mapInPandas either.
    with pytest.raises(ReadSchemaMismatch):
        read_coerce("abc", "void")


def test_read_coerce_still_accepts_string():
    assert read_coerce("abc", "string") == "abc"


# =====================================================================================
# Item 19: a BooleanType read must not turn the string "false" into True
#
# bool("false") is True, so an ES field storing the STRING "false" (common when reading an index
# the connector did not write) silently inverts. Proven live: Row(doc_id='b1', flag=True).
# Mirrors the write side's _is_delete_flagged, which already refuses to guess.
# =====================================================================================

def test_read_coerce_boolean_parses_false_strings():
    for s in ("false", "False", "FALSE", "f", "0", "no", "n"):
        assert read_coerce(s, "boolean") is False, f"{s!r} must read as False"


def test_read_coerce_boolean_parses_true_strings():
    for s in ("true", "True", "t", "1", "yes", "y"):
        assert read_coerce(s, "boolean") is True, f"{s!r} must read as True"


def test_read_coerce_boolean_rejects_ambiguous_strings():
    # 'maybe'/'2'/'on' read as truthy to bool() but mean nothing definite. Raise instead of guess.
    for s in ("maybe", "2", "on", "enabled", "-1"):
        with pytest.raises(ReadSchemaMismatch):
            read_coerce(s, "boolean")


def test_read_coerce_boolean_rejects_empty_and_whitespace_strings():
    """The one input where the read side deliberately does NOT match the write side.

    `transform._is_delete_flagged("")` returns False, because there an empty string means "no flag
    present" and an absent flag must never be read as a delete. Here the caller has DECLARED the
    column boolean and Elasticsearch stored a string, so "" is a value that does not parse rather
    than an absence: a real null reads as None long before this branch. Returning False would invent
    a datum the source does not contain, in the column type where nobody re-checks.

    Pinned because nothing covered it: an "obvious" symmetry fix would silently turn unparseable
    data into False, and this is the test that would stop it.
    """
    for s in ("", " ", "   ", "\t", "\n"):
        with pytest.raises(ReadSchemaMismatch):
            read_coerce(s, "boolean")


def test_write_side_still_treats_an_empty_delete_flag_as_absent():
    # The other half of the asymmetry, pinned so a later "make these consistent" change has to
    # break a test rather than silently start deleting (or refusing) rows. An empty flag means the
    # row is not a delete, matching the null rule.
    from databricks_es_connector.transform import _is_delete_flagged

    for s in ("", " ", "   ", "\t"):
        assert _is_delete_flagged(s) is False, f"{s!r} must mean 'not a delete', not raise"


def test_read_and_write_boolean_parsing_agree_on_every_recognized_string():
    # Whatever the empty-string difference, the two allow-lists must not drift apart on the values
    # they DO recognize: a string that deletes a row on write must read back as True, and vice
    # versa. Divergence there would be a genuine round-trip inversion.
    from databricks_es_connector.transform import _is_delete_flagged

    for s in ("true", "True", "TRUE", "t", "1", "yes", "y", " t "):
        assert _is_delete_flagged(s) is True and read_coerce(s.strip(), "boolean") is True, s
    for s in ("false", "False", "FALSE", "f", "0", "no", "n"):
        assert _is_delete_flagged(s) is False and read_coerce(s, "boolean") is False, s


def test_read_coerce_boolean_keeps_real_booleans_and_numbers():
    assert read_coerce(True, "boolean") is True
    assert read_coerce(False, "boolean") is False
    assert read_coerce(1, "boolean") is True
    assert read_coerce(0, "boolean") is False


# =====================================================================================
# Item 20: a NEGATIVE unaccounted must be SURFACED, but must not fail the write
#
# unaccounted = total_input - (written+deleted+errors+ignored). A negative value means more outcomes
# than inputs: structurally impossible, so it would be a counting bug in THIS library rather than
# anything wrong with the caller's data. Two wrong answers here:
#   - silence (the pre-fix behavior): a broken accounting identity reconciles clean and nobody knows.
#   - raising: fails a healthy write, and on the streaming path that is an infinite retry loop on a
#     batch that can never pass, escapable only via on_error="log", which also switches off real
#     loss detection. test_negative_unaccounted_does_not_raise pins that it must not raise.
# So it logs at ERROR and lets the write stand.
# =====================================================================================

def test_negative_unaccounted_is_logged_not_raised(caplog):
    result = {"written": 12, "deleted": 0, "errors": 0, "ignored": 0, "total_input": 10,
              "unaccounted": -2, "error_samples": []}
    with caplog.at_level("ERROR", logger="databricks_es_connector.bulk"):
        assert reconcile_or_raise(result, index="my-index") is result   # returns clean, no raise
    assert caplog.records, "an impossible count must not pass in total silence"
    msg = caplog.records[0].getMessage()
    assert "my-index" in msg and "impossible" in msg
    assert "accounting bug" in msg      # tells the user it is our bug, not their data


def test_merge_then_reconcile_surfaces_impossible_counts(caplog):
    rows = [{"written": 12, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
             "total_input": 10, "error_samples": "[]"}]
    merged = _merge_partition_results(rows)
    # The over-count is reported in its OWN field, never as a negative `unaccounted`: a negative
    # would subtract from real loss found in another partition (see the cancellation test below).
    assert merged["overcounted"] == 2
    assert merged["unaccounted"] == 0
    with caplog.at_level("ERROR", logger="databricks_es_connector.bulk"):
        reconcile_or_raise(merged, index="i")
    assert caplog.records


def test_overcount_in_one_partition_cannot_cancel_real_loss_in_another(caplog):
    """The cancellation bug: two opposite discrepancies netting to a clean verdict.

    `unaccounted` used to be derived ONCE from the summed counts, so a partition that over-counted
    subtracted from a partition that genuinely lost rows. Both signals vanished and the write
    reported complete success while rows were gone. They are now accumulated per partition and kept
    apart, because they mean opposite things: one is the caller's data going missing, the other is a
    defect in this library.
    """
    lost = {"written": 95, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
            "total_input": 100, "error_samples": "[]"}          # 5 rows produced no outcome
    overcounted = {"written": 105, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
                   "total_input": 100, "error_samples": "[]"}   # 5 outcomes too many

    merged = _merge_partition_results([lost, overcounted])
    # Pre-fix this was `total_input(200) - written(200) == 0`, i.e. perfectly clean.
    assert merged["unaccounted"] == 5, "real loss must survive an over-count elsewhere"
    assert merged["overcounted"] == 5, "the library bug must be reported in its own right"

    with caplog.at_level("ERROR", logger="databricks_es_connector.bulk"):
        with pytest.raises(EsWriteError, match="unaccounted"):
            reconcile_or_raise(merged, index="i")
    assert caplog.records, "the impossible count must still be surfaced alongside the raise"


def test_partition_order_does_not_change_the_verdict():
    # Sign-splitting must be order-independent: whichever partition is seen first, the same two
    # totals come out. An order-dependent verdict would be its own silent-failure mode.
    lost = {"written": 95, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
            "total_input": 100, "error_samples": "[]"}
    over = {"written": 105, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
            "total_input": 100, "error_samples": "[]"}
    a = _merge_partition_results([lost, over])
    b = _merge_partition_results([over, lost])
    assert (a["unaccounted"], a["overcounted"]) == (b["unaccounted"], b["overcounted"]) == (5, 5)


def test_many_partitions_accumulate_both_signs_independently():
    # Three lossy partitions and two over-counting ones: each side sums on its own.
    lossy = [{"written": 8, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
              "total_input": 10, "error_samples": "[]"} for _ in range(3)]      # 2 lost each
    over = [{"written": 13, "deleted": 0, "errors": 0, "ignored": 0, "coerced_nonfinite": 0,
             "total_input": 10, "error_samples": "[]"} for _ in range(2)]       # 3 extra each
    merged = _merge_partition_results(lossy + over)
    assert merged["unaccounted"] == 6      # 3 x 2
    assert merged["overcounted"] == 6      # 2 x 3


def test_delete_404_no_ops_are_still_not_counted_as_loss():
    # `ignored` must remain part of the per-partition identity. A delete-404 is an expected no-op, so
    # a partition of pure no-ops is CLEAN, not 3 rows lost.
    rows = [{"written": 0, "deleted": 0, "errors": 0, "ignored": 3, "coerced_nonfinite": 0,
             "total_input": 3, "error_samples": "[]"}]
    merged = _merge_partition_results(rows)
    assert merged["unaccounted"] == 0 and merged["overcounted"] == 0
    assert reconcile_or_raise(merged, index="i") is merged


def test_legacy_signed_unaccounted_is_still_surfaced(caplog):
    # A result dict from before this split (single signed `unaccounted`, no `overcounted` key) must
    # degrade to the new semantics rather than silently reading as clean.
    legacy = {"written": 12, "deleted": 0, "errors": 0, "ignored": 0, "total_input": 10,
              "unaccounted": -2, "error_samples": []}
    with caplog.at_level("ERROR", logger="databricks_es_connector.bulk"):
        assert reconcile_or_raise(legacy, index="i") is legacy    # still must not raise
    assert caplog.records and "impossible" in caplog.records[0].getMessage()


def test_every_result_producer_agrees_on_the_key_set():
    """The result dict's shape is stated in three places; they must not drift apart.

    `_merge_partition_results` builds it, `bulk_write`'s docstring documents it, and `stream.py`'s
    empty-batch stand-in mimics it. When `overcounted` was added, all three needed it: a callback
    doing `result["overcounted"]` would KeyError on an empty micro-batch otherwise. A subset check
    would not have caught that, so this compares the sets exactly.
    """
    import re

    produced = set(_merge_partition_results([]))

    # 1. bulk_write's docstring must list exactly the keys that are produced.
    doc = bulk_mod.bulk_write.__doc__
    listed = set(re.findall(r"'(\w+)'", doc[doc.index("Returns {"):doc.index("}:")]))
    assert listed == produced, (
        f"bulk_write docstring and _merge_partition_results disagree: "
        f"only in docstring={listed - produced}, only produced={produced - listed}")

    # 2. The streaming empty-batch stand-in must carry every produced key (plus its `empty` flag),
    #    since callbacks read the same keys whether or not the batch had rows.
    src = inspect.getsource(stream_mod)
    stand_in = src[src.index('"written": 0'):src.index('"empty": True')]
    for key in produced - {"error_samples"}:
        assert f'"{key}"' in stand_in, (
            f"stream.py's empty-batch result is missing {key!r}; a callback reading it would "
            "KeyError on an empty micro-batch")


def test_overcount_is_surfaced_under_both_raise_and_log(monkeypatch, caplog):
    """The signal must not depend on the on_error policy.

    `reconcile_or_raise` logs an impossible count on the RAISE path. The LOG path has its own
    reporting branch and checked only errors/unaccounted, so an over-count was surfaced in the
    default mode and silently dropped in the other: the same "fixed one branch, left its sibling"
    shape this suite exists to catch. Neither mode fails the batch over it.
    """
    import logging

    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=105, total_input=100, overcounted=5))

    for policy in (RAISE, LOG):
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            make_foreach_batch(_cfg(), on_error=policy)(_FakeDF(), 1)   # must NOT raise
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "impossible" in msgs, f"on_error={policy!r} swallowed the impossible count"
        assert "accounting bug" in msgs, f"on_error={policy!r} must name it as OUR bug"


def test_overcount_stays_silent_under_ignore(monkeypatch, caplog):
    # "ignore" is documented as silent by definition; pinned so the fix above does not leak into it.
    import logging

    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=105, total_input=100, overcounted=5))
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        make_foreach_batch(_cfg(), on_error=IGNORE)(_FakeDF(), 1)
    assert caplog.records == [], f"on_error='ignore' must stay silent, got {caplog.text}"


def test_an_overcount_alone_never_fails_a_batch(monkeypatch):
    # Guard-rail on the fix: surfacing the over-count must not start FAILING batches over a library
    # bug, which on the streaming path is an unescapable retry loop.
    monkeypatch.setattr(stream_mod, "bulk_write",
                        lambda df, cfg: _result(written=105, total_input=100, overcounted=5))
    make_foreach_batch(_cfg(), on_error=RAISE)(_FakeDF(), 1)   # returns cleanly


def test_positive_unaccounted_still_raises():
    # The guard-rail for the above: loosening the negative case must not weaken real loss detection.
    result = {"written": 7, "deleted": 0, "errors": 0, "ignored": 0, "total_input": 10,
              "unaccounted": 3, "error_samples": []}
    with pytest.raises(EsWriteError, match="unaccounted"):
        reconcile_or_raise(result, index="i")


# =====================================================================================
# Item 21: to_es_source's `id_field` parameter was accepted but had NO effect (dead parameter)
# =====================================================================================

def test_to_es_source_has_no_dead_id_field_parameter():
    import databricks_es_connector.transform as tmod
    params = inspect.signature(tmod.to_es_source).parameters
    assert "id_field" not in params, (
        "to_es_source accepted id_field but ignored it entirely; a public parameter that lies is "
        "worse than no parameter")
