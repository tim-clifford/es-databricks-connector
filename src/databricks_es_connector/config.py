"""Connection + write configuration for the Elasticsearch bulk sink.

A plain, serializable dataclass so it can be captured in a closure and shipped
to Spark executors (mapInPandas / foreachBatch) without dragging a live client.
The Elasticsearch client is constructed *inside* each partition from this config,
never on the driver, so nothing non-serializable crosses the wire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EsConfig:
    # --- connection ---
    hosts: str                              # e.g. "https://host:9200"
    api_key: Optional[str] = None           # preferred: base64 id:key or encoded key
    basic_auth: Optional[tuple] = None      # ("user", "pass") — sandbox only
    verify_certs: bool = True               # False for self-signed sandbox boxes
    ca_certs: Optional[str] = None          # path to CA bundle when pinning

    # --- write behavior ---
    index: str = ""                         # target index (or use a per-row _index)
    id_field: Optional[str] = None          # column used as deterministic _id (idempotency)
    http_compress: bool = True              # gzip the bulk request body (egress lever)
    chunk_size: int = 500                   # docs per bulk request
    request_timeout: int = 60
    max_retries: int = 3
    retry_on_timeout: bool = True

    # --- doc shaping ---
    drop_fields: tuple = field(default_factory=tuple)  # columns to prune before indexing (egress lever)

    # --- change data feed (CDC) ---
    # When change_feed=True, the input DataFrame is expected to be a Delta Change Data Feed
    # (read with .option("readChangeFeed","true")). The connector then routes each change to
    # the right ES op: insert/update_postimage -> index (upsert on _id), delete -> delete;
    # update_preimage rows are dropped. Within each batch, multiple changes to the same _id are
    # collapsed to the single latest one (highest commit version). Default False keeps the plain
    # append/upsert behavior unchanged.
    change_feed: bool = False
    change_type_field: str = "_change_type"      # CDF column naming the change kind
    commit_version_field: str = "_commit_version"  # CDF column giving per-commit ordering

    def __post_init__(self):
        if not self.hosts:
            raise ValueError("EsConfig.hosts is required")
        if self.api_key is None and self.basic_auth is None:
            raise ValueError("EsConfig requires either api_key or basic_auth")
        if self.verify_certs is False and self.ca_certs:
            raise ValueError("ca_certs is set but verify_certs is False — pick one")
        if self.change_feed and not self.id_field:
            # No deterministic _id => cannot upsert or delete a specific doc, and cannot
            # collapse multiple changes for the same record.
            raise ValueError("change_feed=True requires id_field (deterministic _id)")

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
