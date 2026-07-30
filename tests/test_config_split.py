"""Unit tests for the 0.4.0 config split: EsConnection base + EsWriteConfig / EsReadConfig, with
EsConfig kept as a backward-compatible alias. No Spark, no ES."""
import pytest

from databricks_es_connector import EsConnection, EsWriteConfig, EsReadConfig, EsConfig


def test_esconfig_is_alias_for_write_config():
    # Pre-0.4.0 code used EsConfig for writes; it must remain exactly EsWriteConfig.
    assert EsConfig is EsWriteConfig


def test_write_and_read_share_the_connection_base():
    assert issubclass(EsWriteConfig, EsConnection)
    assert issubclass(EsReadConfig, EsConnection)


def test_connection_validation_is_shared():
    # hosts + one-of-auth are enforced on both configs via the shared base.
    for cls in (EsWriteConfig, EsReadConfig):
        with pytest.raises(ValueError, match="hosts"):
            cls(hosts="", basic_auth=("u", "p"))
        with pytest.raises(ValueError, match="api_key or basic_auth"):
            cls(hosts="https://h:9200")
        with pytest.raises(ValueError, match="pick one"):
            cls(hosts="https://h:9200", basic_auth=("u", "p"), verify_certs=False, ca_certs="/x")


def test_client_kwargs_includes_ca_certs_when_pinned():
    # ca_certs (CA-bundle pinning) must be passed through to the ES client kwargs.
    w = EsWriteConfig(hosts="https://h:9200", api_key="k", index="i", ca_certs="/etc/ca.pem")
    kw = w.client_kwargs()
    assert kw["ca_certs"] == "/etc/ca.pem"
    # And omitted (not None) when not set, so the client uses its default trust store.
    assert "ca_certs" not in EsWriteConfig(hosts="https://h:9200", api_key="k", index="i").client_kwargs()


def test_client_kwargs_identical_for_same_connection():
    # The client kwargs come from the shared base, so a read and write config with the same
    # connection produce the same client configuration.
    conn = dict(hosts="https://h:9200", api_key="k", request_timeout=42)
    assert EsWriteConfig(**conn, index="i").client_kwargs() == \
           EsReadConfig(**conn, index="i").client_kwargs()


def test_write_only_fields_absent_from_read_config():
    # Delete/write-shaping knobs must not leak onto the read config.
    r = EsReadConfig(hosts="https://h:9200", api_key="k", index="i")
    for f in ("has_deletes", "delete_flag_column", "drop_fields", "chunk_size"):
        assert not hasattr(r, f), f"read config unexpectedly has write field {f}"


def test_read_only_fields_absent_from_write_config():
    w = EsWriteConfig(hosts="https://h:9200", api_key="k", index="i")
    for f in ("query", "num_slices", "batch_size", "pit_keep_alive", "include_id"):
        assert not hasattr(w, f), f"write config unexpectedly has read field {f}"


def test_write_delete_validation_still_enforced_on_write_config():
    # The delete-routing validation moved with the fields onto EsWriteConfig — still enforced.
    with pytest.raises(ValueError, match="delete_flag_column"):
        EsWriteConfig(hosts="https://h:9200", api_key="k", index="i",
                      id_field="doc_id", has_deletes=True)
