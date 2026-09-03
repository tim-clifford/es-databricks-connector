"""Unit tests for the optional write-log table (EsWriteConfig.log_table + bulk.build_log_rows).

Pure Python, no Spark, no ES: build_log_rows is the pure row builder that _write_log (Spark-side,
proven in the live tier) hands to createDataFrame. These lock the row shape, the batch-vs-partition
split, and the config validation.
"""
import datetime as dt

import pytest

from databricks_es_connector import EsWriteConfig
from databricks_es_connector.bulk import build_log_rows, LOG_COLUMNS


_RESULT = {"total_input": 300, "written": 295, "deleted": 0, "errors": 3,
           "ignored": 2, "unaccounted": 0, "overcounted": 0}
# What .asDict() yields from the collected mapInPandas summary Rows.
_PARTS = [
    {"partition_id": 0, "partition_duration_ms": 1200, "total_input": 100,
     "written": 100, "deleted": 0, "errors": 0, "ignored": 0},
    {"partition_id": 1, "partition_duration_ms": 3400, "total_input": 200,
     "written": 195, "deleted": 0, "errors": 3, "ignored": 2},
]
_EVENT = dt.datetime(2026, 9, 3, 7, 0, 0, tzinfo=dt.timezone.utc)


def test_config_accepts_log_table_and_defaults_none():
    assert EsWriteConfig(hosts="https://h:9200", api_key="k", index="i").log_table is None
    cfg = EsWriteConfig(hosts="https://h:9200", api_key="k", index="i", log_table="cat.sch.es_log")
    assert cfg.log_table == "cat.sch.es_log"


def test_config_rejects_empty_log_table():
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="log_table is set but empty"):
            EsWriteConfig(hosts="https://h:9200", api_key="k", index="i", log_table=bad)


def test_one_batch_row_plus_one_row_per_partition():
    rows = build_log_rows(_RESULT, _PARTS, _EVENT, 4600, batch_id=7, index="idx")
    assert len(rows) == 1 + len(_PARTS)
    assert [r["scope"] for r in rows] == ["batch", "partition", "partition"]
    # every row carries exactly the declared columns
    for r in rows:
        assert set(r.keys()) == set(LOG_COLUMNS)


def test_batch_row_carries_aggregate_and_wall_clock():
    batch = build_log_rows(_RESULT, _PARTS, _EVENT, 4600, batch_id=7, index="idx")[0]
    assert batch["scope"] == "batch"
    assert batch["partition_id"] is None          # not a partition-scoped row
    assert batch["duration_ms"] == 4600           # whole-call wall clock, not a partition's
    assert batch["event_time"] == _EVENT
    assert batch["batch_id"] == 7
    assert batch["index"] == "idx"
    assert (batch["total_input"], batch["written"], batch["errors"], batch["ignored"],
            batch["unaccounted"]) == (300, 295, 3, 2, 0)


def test_partition_rows_carry_their_own_timing_and_counts():
    rows = build_log_rows(_RESULT, _PARTS, _EVENT, 4600, batch_id=7, index="idx")
    p1 = rows[2]
    assert p1["scope"] == "partition"
    assert p1["partition_id"] == 1
    assert p1["duration_ms"] == 3400              # THIS partition's duration, not the batch's
    assert (p1["total_input"], p1["written"], p1["errors"]) == (200, 195, 3)
    # unaccounted is a cross-partition reconciliation, meaningless per-partition => None
    assert p1["unaccounted"] is None


def test_batch_id_none_is_preserved_not_coerced():
    # A non-streaming bulk_write passes no batch_id; the column must be a real null, not 0.
    rows = build_log_rows(_RESULT, _PARTS, _EVENT, 10, batch_id=None, index="idx")
    assert all(r["batch_id"] is None for r in rows)


def test_no_partitions_still_yields_the_batch_row():
    rows = build_log_rows(_RESULT, [], _EVENT, 5, batch_id=1, index="idx")
    assert len(rows) == 1 and rows[0]["scope"] == "batch"


def test_missing_partition_counts_default_to_zero():
    # A summary row missing a count key degrades to 0 rather than raising (mirrors the tolerant
    # reading in _merge_partition_results).
    rows = build_log_rows(_RESULT, [{"partition_id": 0, "partition_duration_ms": 5}],
                          _EVENT, 5, batch_id=1, index="idx")
    part = rows[1]
    assert part["written"] == 0 and part["total_input"] == 0 and part["errors"] == 0
