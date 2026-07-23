"""Unit tests for delete support: EsConfig validation + the pure bulk-result classifier.

No Spark, no ES client. The classifier is the exact rule that decides whether a
delete-404 is a silent no-op or a counted error, so it gets dedicated coverage here.
"""
import pytest

from databricks_es_connector.config import EsConfig
from databricks_es_connector.bulk import (
    classify_bulk_result, WRITTEN, DELETED, IGNORED, ERROR,
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
