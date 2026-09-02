"""Unit tests for per-partition write concurrency (EsWriteConfig.write_concurrency).

No Spark, no live ES: the elasticsearch helpers the writer imports internally are monkeypatched,
same as test_bulk_deletes.py. These cover the three things the concurrency path must guarantee:
  1. write_concurrency == 1 is the original serial path (a single streaming_bulk stream).
  2. write_concurrency > 1 fans the actions across that many streaming_bulk streams and merges every
     per-document result exactly once, with the connector's retry settings passed to each stream.
  3. a worker exception is re-raised (fail closed), so a partial write can never report clean counts.
"""
import threading

import pytest

from databricks_es_connector.config import EsConfig
from databricks_es_connector.bulk import (
    make_partition_writer, _iter_bulk_results, _streaming_bulk,
)


def _cfg(**kw):
    base = dict(hosts="https://h:9200", basic_auth=("u", "p"), index="i")
    base.update(kw)
    return EsConfig(**base)


# --- config validation -------------------------------------------------------------------

def test_write_concurrency_defaults_to_one():
    assert _cfg().write_concurrency == 1


def test_write_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="write_concurrency must be >= 1"):
        _cfg(write_concurrency=0)


def test_client_kwargs_sizes_connection_pool_to_concurrency():
    # Serial path unchanged: no pool override, so the client keeps elastic_transport's default.
    assert "connections_per_node" not in _cfg().client_kwargs()
    # Concurrent path: pool is sized to the concurrency so the fanned workers aren't capped below it.
    assert _cfg(write_concurrency=8).client_kwargs()["connections_per_node"] == 8


# --- a recording stub for helpers.streaming_bulk -----------------------------------------
# Yields one (ok, result) tuple per action so a test can recover exactly which actions each
# stream processed, and records every call's kwargs (thread-safely) so the fan-out and the
# retry settings are both observable.

def _install_stub(monkeypatch, *, boom_id=None):
    import elasticsearch
    import elasticsearch.helpers

    calls = []            # one (actions_seen, kwargs) per streaming_bulk invocation
    lock = threading.Lock()

    def _stub(es, actions, **kw):
        seen = list(actions)
        with lock:
            calls.append(([a["n"] for a in seen], kw))
        for a in seen:
            if boom_id is not None and a["n"] == boom_id:
                raise RuntimeError("transport died mid-stream")
            yield (True, {"index": {"_id": a["n"], "status": 201}})

    class _FakeES:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", _stub)
    return calls


def _actions(n):
    return [{"n": k} for k in range(n)]


# --- _iter_bulk_results: serial vs fanned -------------------------------------------------

def test_serial_path_uses_one_stream(monkeypatch):
    calls = _install_stub(monkeypatch)
    cfg = _cfg(write_concurrency=1)
    out = list(_iter_bulk_results(object(), _actions(10), cfg))
    assert len(calls) == 1                              # exactly one streaming_bulk stream
    assert sorted(r["index"]["_id"] for _, r in out) == list(range(10))


def test_concurrent_path_fans_out_and_merges_every_doc(monkeypatch):
    calls = _install_stub(monkeypatch)
    cfg = _cfg(write_concurrency=4)
    out = list(_iter_bulk_results(object(), _actions(40), cfg))
    # One stream per concurrency unit, and the union of what they processed is every action exactly
    # once (strided slices partition the input, no overlap, no gap).
    assert len(calls) == 4
    seen = sorted(n for actions_seen, _ in calls for n in actions_seen)
    assert seen == list(range(40))
    # And the merged per-document results carry every doc exactly once.
    assert sorted(r["index"]["_id"] for _, r in out) == list(range(40))


def test_concurrent_streams_get_the_retry_settings(monkeypatch):
    calls = _install_stub(monkeypatch)
    cfg = _cfg(write_concurrency=3, chunk_size=250, max_retries_per_doc=4, retry_on_doc_status=(429, 503))
    list(_iter_bulk_results(object(), _actions(9), cfg))
    assert len(calls) == 3
    for _, kw in calls:
        # Each fanned stream must be the SAME request the serial path would issue: identical chunking
        # and per-document 429 retry, so error accounting is unchanged by concurrency.
        assert kw["chunk_size"] == 250
        assert kw["max_retries"] == 4
        assert kw["retry_on_status"] == (429, 503)
        assert kw["raise_on_error"] is False and kw["yield_ok"] is True


def test_worker_exception_propagates_fail_closed(monkeypatch):
    # RED-BEFORE-GREEN guard: if _iter_bulk_results swallowed a worker failure (dropped the futures'
    # result() check), the docs the dead stream never sent would vanish while the partition reported
    # a clean partial count. Deleting the `f.result()` loop makes this test fail.
    _install_stub(monkeypatch, boom_id=17)
    cfg = _cfg(write_concurrency=4)
    with pytest.raises(RuntimeError, match="transport died"):
        list(_iter_bulk_results(object(), _actions(40), cfg))


def test_early_consumer_abort_does_not_deadlock(monkeypatch):
    # RED-BEFORE-GREEN guard for the abort deadlock: fill the bounded queue, then abandon the
    # generator. Without the stop-flag release + drain, the workers stay blocked on put() and
    # ThreadPoolExecutor.__exit__'s shutdown(wait=True) hangs the partition forever. The workers must
    # be released so close() returns promptly. 400 actions >> the maxsize-8 queue, so producers are
    # blocked by the time we abort.
    _install_stub(monkeypatch)
    cfg = _cfg(write_concurrency=4)
    gen = _iter_bulk_results(object(), _actions(400), cfg)
    next(gen)                              # start the workers, then stop draining

    done = threading.Event()

    def _close():
        gen.close()                        # GeneratorExit at the paused yield -> must not hang
        done.set()

    threading.Thread(target=_close, daemon=True).start()
    assert done.wait(timeout=15), "generator.close() hung: early-abort deadlock"


# --- end-to-end through the writer, concurrent -------------------------------------------

def test_partition_writer_counts_are_correct_under_concurrency(monkeypatch):
    pd = pytest.importorskip("pandas")
    import elasticsearch
    import elasticsearch.helpers

    # Shape-agnostic stub: the writer feeds REAL build_action output (no "n" key), so yield one
    # success per action without inspecting it. Fanned across 4 streams, the merged counts must
    # still total every input row exactly once.
    def _stub(es, actions, **kw):
        for _ in actions:
            yield (True, {"index": {"status": 201}})

    class _FakeES:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(elasticsearch, "Elasticsearch", _FakeES)
    monkeypatch.setattr(elasticsearch.helpers, "streaming_bulk", _stub)

    cfg = _cfg(id_field="id", write_concurrency=4)
    writer = make_partition_writer(cfg)
    df = pd.DataFrame({"id": [str(k) for k in range(40)]})
    out = list(writer(iter([df])))
    row = out[0].iloc[0]
    assert int(row["written"]) == 40
    assert int(row["errors"]) == 0
    assert int(row["total_input"]) == 40
