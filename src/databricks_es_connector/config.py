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

import functools
import warnings
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
    # TRANSPORT-level retries: they fire on the HTTP status of a whole REQUEST (a connection reset, a
    # load-balancer 503, a gateway timeout, or a 429 on the bulk call itself), and they re-send the
    # ENTIRE request. They do NOT retry an individual rejected document, because the `_bulk` API
    # answers HTTP 200 even when items inside it fail: the per-item statuses live in the response
    # BODY, which the transport layer never inspects. Per-document retries are a separate knob,
    # `EsWriteConfig.max_retries_per_doc`.
    #
    # Named `transport_max_retries` (renamed from `max_retries` in 0.6.0) precisely because the old
    # name was one character from `max_retries_per_doc` while meaning something entirely different:
    # both defaulted to 3, so the pair was indistinguishable at a glance. `max_retries=` is still
    # accepted with a DeprecationWarning (see __init__ below).
    transport_max_retries: int = 3
    retry_on_timeout: bool = True

    @property
    def max_retries(self) -> int:
        """Deprecated read alias for `transport_max_retries` (renamed in 0.6.0).

        Kept so existing code that READS cfg.max_retries keeps working. Writing it (passing
        `max_retries=` to the constructor) is handled by `_accept_deprecated_max_retries` below.
        """
        return self.transport_max_retries

    def __post_init__(self):
        if not self.hosts:
            raise ValueError("hosts is required")
        if self.api_key is None and self.basic_auth is None:
            raise ValueError("EsConfig requires either api_key or basic_auth")
        if self.verify_certs is False and self.ca_certs:
            raise ValueError("ca_certs is set but verify_certs is False, pick one")
        if self.transport_max_retries < 0:
            raise ValueError("transport_max_retries must be >= 0")

    def client_kwargs(self) -> dict:
        """Kwargs for elasticsearch.Elasticsearch(...). Built on the executor."""
        kw = {
            "hosts": self.hosts,
            "http_compress": self.http_compress,
            "request_timeout": self.request_timeout,
            # The elasticsearch-py client's own kwarg is still called `max_retries`; only OUR field
            # was renamed. This is the one place the two spellings meet.
            "max_retries": self.transport_max_retries,
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
    index: str = ""                         # target index; one index per write (no per-row routing)
    id_field: Optional[str] = None          # column used as deterministic _id (idempotency)
    chunk_size: int = 500                   # docs per bulk request

    # Per-DOCUMENT retries for rows Elasticsearch rejects with a retryable status (429
    # es_rejected_execution_exception: the ES write queue is full). This is NOT the same as
    # EsConnection.max_retries, which is transport-level and only fires on the HTTP status of the
    # whole request: the _bulk API returns HTTP 200 even when individual items fail, so the
    # transport retry never sees a rejected document. Without this, a 429'd row is a hard error on
    # its first and only attempt. Backoff is exponential (elasticsearch-py: initial_backoff * 2**n).
    max_retries_per_doc: int = 3
    retry_on_doc_status: tuple = (429,)     # per-doc statuses worth retrying (429 = ES queue full)

    # --- doc shaping ---
    drop_fields: tuple = field(default_factory=tuple)  # columns to prune before indexing (egress lever)
    # drop_fields is frequently used as a PII/egress control, and a misspelled column name would
    # silently prune NOTHING (the field ships to ES anyway). True => bulk_write validates every
    # drop_fields name against the DataFrame schema and raises on an unknown one, so the control
    # fails CLOSED. Set False only if you deliberately reuse one config across DataFrames with
    # differing columns.
    strict_drop_fields: bool = True

    # Elasticsearch auto-creates a missing index on write (action.auto_create_index defaults to
    # true), so a TYPO'd index name silently produces a brand-new dynamically-mapped index and a
    # perfect-looking written count. True => bulk_write checks the index exists first and raises if
    # not. Set False for a pipeline that intentionally relies on auto-creation or an index template.
    require_existing_index: bool = True

    # --- deletes ---
    # has_deletes=False (default) is the historical behavior: every row is an index/upsert.
    # Set has_deletes=True *and* delete_flag_column to route rows whose flag is truthy to an
    # ES delete-by-id instead of an index. Requires id_field (you cannot delete without an _id).
    has_deletes: bool = False
    delete_flag_column: Optional[str] = None  # boolean-ish column: truthy => delete this _id

    def __post_init__(self):
        super().__post_init__()
        if self.max_retries_per_doc < 0:
            raise ValueError("EsWriteConfig.max_retries_per_doc must be >= 0")
        # `max_retries` and `max_retries_per_doc` are one character apart and mean different things.
        # Someone who raises the transport-level knob to harden against ES backpressure has almost
        # certainly NOT intended to disable per-document retries, which are the ones that actually
        # cover a 429'd document (the _bulk API returns HTTP 200 even when items inside fail, so the
        # transport retry cannot see them). Warn rather than raise: the combination is legal, just
        # very unlikely to be what was meant.
        if self.max_retries_per_doc == 0 and self.transport_max_retries > 3:
            warnings.warn(
                f"transport_max_retries={self.transport_max_retries} is raised above the default but "
                "max_retries_per_doc=0, so documents Elasticsearch rejects with a 429 get NO "
                "retries. The _bulk API returns HTTP 200 even when individual items fail, so "
                "transport-level retries never see them. Set max_retries_per_doc (default 3) if you "
                "meant to retry rejected documents.",
                UserWarning, stacklevel=3)
        if not self.retry_on_doc_status and self.max_retries_per_doc:
            # Retries requested but no status to retry on would silently never retry.
            raise ValueError("max_retries_per_doc is set but retry_on_doc_status is empty, "
                             "name at least one status (e.g. (429,)) or set max_retries_per_doc=0")
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
    # When num_slices is None the reader looks up the index's shard count. If that lookup fails it
    # can only fall back to ONE unsliced reader: correct but serial, silently forfeiting the
    # parallelism read_index exists for (and a `logging` warning is easy to miss entirely on
    # serverless). True => raise instead of degrading quietly. Set False to accept the serial read.
    strict_slices: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("EsReadConfig.batch_size must be positive")
        if self.num_slices is not None and self.num_slices < 1:
            raise ValueError("EsReadConfig.num_slices must be >= 1 when set")


def _accept_deprecated_max_retries(cls):
    """Let `max_retries=` still construct a config, with a DeprecationWarning.

    `max_retries` was renamed `transport_max_retries` in 0.6.0 because the old name sat one character
    from `max_retries_per_doc` while meaning something entirely different (whole HTTP request vs one
    rejected document), and both defaulted to 3, so the pair was indistinguishable at a glance.

    Applied AFTER @dataclass so it wraps the GENERATED __init__: defining __init__ in the class body
    would make @dataclass skip generating one at all, and there is no `__dataclass_init__` to fall
    back to (verified). Wrapping keeps field order, defaults and inheritance exactly as generated.
    """
    original = cls.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if "max_retries" in kwargs:
            if "transport_max_retries" in kwargs:
                raise ValueError(
                    "pass only one of transport_max_retries or max_retries (deprecated alias for it)")
            warnings.warn(
                f"{cls.__name__}.max_retries was renamed to transport_max_retries in 0.6.0, to "
                "distinguish it from EsWriteConfig.max_retries_per_doc: this one retries a whole HTTP "
                "REQUEST, the other retries an individual DOCUMENT that Elasticsearch rejected (the "
                "_bulk API returns HTTP 200 even when items inside it fail, so the transport retry "
                "cannot see them). The old name still works but will be removed in a future release.",
                DeprecationWarning, stacklevel=2)
            kwargs["transport_max_retries"] = kwargs.pop("max_retries")
        original(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


for _cls in (EsConnection, EsWriteConfig, EsReadConfig):
    _accept_deprecated_max_retries(_cls)


# Backward-compatible alias: pre-0.4.0, the (write) config was named EsConfig. Keep it working so
# existing write code and tests don't break; new code should use EsWriteConfig / EsReadConfig.
EsConfig = EsWriteConfig
