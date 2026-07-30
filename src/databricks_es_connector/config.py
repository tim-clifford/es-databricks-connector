"""Connection + read/write configuration for the Elasticsearch connector.

Plain, serializable frozen dataclasses so they can be captured in a closure and shipped to Spark
executors (mapInPandas / foreachBatch) without dragging a live client. The Elasticsearch client is
constructed *inside* each partition from a config, never on the driver, so nothing non-serializable
crosses the wire.

Three types, sharing one connection base:
  - EsConnection: connection + client tuning (hosts, auth, TLS, timeouts, compression).
  - EsWriteConfig: connection + write behavior (index, id_field, chunking, deletes). Used by
                     bulk_write / make_foreach_batch.
  - EsReadConfig: connection + read behavior (index, query, slicing, paging). Used by read_index.

`EsConfig` remains as a backward-compatible alias for `EsWriteConfig` (the pre-0.4.0 name), so
existing write code keeps working; new code should name the read/write config explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EsConnection:
    """Connection + client tuning shared by reads and writes.

    The Elasticsearch client kwargs are built from this (on the executor), so every field here is a
    client-level concern. `hosts` is the only required field; auth (exactly one of api_key /
    basic_auth) is validated in __post_init__.
    """
    # --- connection ---
    hosts: str                              # e.g. "https://host:9200"
    api_key: Optional[str] = None           # preferred: base64 id:key or encoded key
    basic_auth: Optional[tuple] = None      # ("user", "pass"), sandbox only
    verify_certs: bool = True               # False for self-signed sandbox boxes
    ca_certs: Optional[str] = None          # path to CA bundle when pinning

    # --- client tuning (apply to both read and write requests) ---
    http_compress: bool = True              # gzip both ways: request body on write, ES response on read
    request_timeout: int = 60
    max_retries: int = 3
    retry_on_timeout: bool = True

    def __post_init__(self):
        if not self.hosts:
            raise ValueError("hosts is required")
        if self.api_key is None and self.basic_auth is None:
            raise ValueError("EsConfig requires either api_key or basic_auth")
        if self.verify_certs is False and self.ca_certs:
            raise ValueError("ca_certs is set but verify_certs is False, pick one")

    def client_kwargs(self) -> dict:
        """Kwargs for elasticsearch.Elasticsearch(...). Built on the executor."""
        kw = {
            "hosts": self.hosts,
            "http_compress": self.http_compress,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "retry_on_timeout": self.retry_on_timeout,
            "verify_certs": self.verify_certs,
        }
        if self.api_key is not None:
            kw["api_key"] = self.api_key
        if self.basic_auth is not None:
            kw["basic_auth"] = self.basic_auth
        if self.ca_certs is not None:
            kw["ca_certs"] = self.ca_certs
        return kw


@dataclass(frozen=True)
class EsWriteConfig(EsConnection):
    """Connection + write behavior for bulk_write / make_foreach_batch."""
    # --- write behavior ---
    index: str = ""                         # target index (or use a per-row _index)
    id_field: Optional[str] = None          # column used as deterministic _id (idempotency)
    chunk_size: int = 500                   # docs per bulk request

    # --- doc shaping ---
    drop_fields: tuple = field(default_factory=tuple)  # columns to prune before indexing (egress lever)

    # --- deletes ---
    # has_deletes=False (default) is the historical behavior: every row is an index/upsert.
    # Set has_deletes=True *and* delete_flag_column to route rows whose flag is truthy to an
    # ES delete-by-id instead of an index. Requires id_field (you cannot delete without an _id).
    has_deletes: bool = False
    delete_flag_column: Optional[str] = None  # boolean-ish column: truthy => delete this _id

    def __post_init__(self):
        super().__post_init__()
        # Delete routing is all-or-nothing and needs an _id to target.
        if self.has_deletes:
            if not self.delete_flag_column:
                raise ValueError("has_deletes=True requires delete_flag_column")
            if self.id_field is None:
                raise ValueError("has_deletes=True requires id_field (deletes target a doc _id)")
        elif self.delete_flag_column is not None:
            # A flag column set with deletes off would silently do nothing, reject the misconfig
            # rather than let a caller believe deletes are happening.
            raise ValueError("delete_flag_column is set but has_deletes is False, enable has_deletes or drop it")


@dataclass(frozen=True)
class EsReadConfig(EsConnection):
    """Connection + read behavior for read_index.

    v0.4.0: the caller declares the Spark schema separately (no mapping inference); this config
    carries only connection + which index / query / how to page and slice it.
    """
    index: str = ""                         # source index (required by read_index)
    id_field: Optional[str] = None          # if the declared schema names this, it is filled from _id
    query: Optional[dict] = None            # raw ES query DSL; None => match_all. No Spark pushdown in v1.
    num_slices: Optional[int] = None        # distributed-reader parallelism (None => shard count)
    batch_size: int = 1000                  # docs per scroll/PIT page
    # Point-in-Time lifetime. This is a SLIDING window, not a total budget: every page request
    # re-sends keep_alive and resets the PIT's clock (verified against the ES PIT API), so it only
    # has to cover the longest gap between consecutive touches of the PIT, not the whole job. The
    # binding gap for read_index is open-PIT -> first executor page (the DataFrame is lazy, so Spark
    # may not schedule the read tasks immediately, and serverless executor cold-start adds to it).
    # Default 5m gives that gap headroom; raise it if a slow downstream consumer stretches the gap
    # between pages. See read_index's docstring.
    pit_keep_alive: str = "5m"
    include_id: bool = True                 # expose ES _id when the schema declares the id_field

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("EsReadConfig.batch_size must be positive")
        if self.num_slices is not None and self.num_slices < 1:
            raise ValueError("EsReadConfig.num_slices must be >= 1 when set")


# Backward-compatible alias: pre-0.4.0, the (write) config was named EsConfig. Keep it working so
# existing write code and tests don't break; new code should use EsWriteConfig / EsReadConfig.
EsConfig = EsWriteConfig
