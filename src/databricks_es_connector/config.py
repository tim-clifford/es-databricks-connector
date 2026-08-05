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

RETRIES: ONE UMBRELLA OR TWO LAYERS
-----------------------------------
Elasticsearch can fail a write in two independent places, so there are two retry layers:

  transport_max_retries   a whole HTTP REQUEST failed (connection reset, LB 503, gateway timeout,
                          or a 429 on the bulk call itself). Re-sends the entire request. Applies to
                          reads and writes. -> elastic_transport.Transport
  max_retries_per_doc     ONE DOCUMENT inside a successful request was rejected (429, ES write queue
                          full). Re-sends only the failed subset. Writes only.
                          -> elasticsearch.helpers.streaming_bulk

They exist separately because the `_bulk` API answers **HTTP 200 even when documents inside it
fail**: the per-item statuses live in the response body, which the transport layer never inspects.
A 429 therefore means different things at each layer, and only the per-document one loses data
quietly.

Configure them either way, but not both ways at once:

    EsWriteConfig(..., max_retries=5)                 # UMBRELLA: 5 at every layer
    EsWriteConfig(..., transport_max_retries=5,       # GRANULAR: tune each layer
                       max_retries_per_doc=2)

`max_retries` is a constructor-level convenience, not a stored field: it expands into the real
per-layer fields. Passing it together with either per-layer field raises, so the effective retry
count never depends on an invisible precedence rule. Reading `cfg.max_retries` gives the shared value
when the layers agree, and `None` when they were set granularly (there is no single number to report).

One asymmetry worth knowing before raising the umbrella: per-document retries sleep BLOCKINGLY in the
executor with exponential backoff (elasticsearch-py: 2s, doubling, capped at 600s), so
`max_retries=8` is up to ~8.5 minutes of sleep per partition, while the same value costs the
transport layer almost nothing. Prefer the granular form when you need a high transport count.
"""
from __future__ import annotations

import functools
import inspect
import warnings
from dataclasses import dataclass, field, fields
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
    # BODY, which the transport layer never inspects. Per-document retries are
    # `EsWriteConfig.max_retries_per_doc`.
    #
    # `max_retries` is the UMBRELLA that sets both layers at once (see the module docstring). Leave
    # this field alone and set `max_retries` for one simple knob; set this one to tune the transport
    # layer independently. Setting both is rejected, so the effective value is never ambiguous.
    transport_max_retries: int = 3
    retry_on_timeout: bool = True

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
        # The `max_retries` umbrella cannot produce this combination (it sets both layers to the same
        # value), so reaching here means the layers were configured GRANULARLY: the transport layer
        # was hardened while per-document retries were switched off. That is almost certainly not
        # what someone protecting against ES backpressure intended, because per-document retries are
        # the only ones that cover a 429'd document (the _bulk API returns HTTP 200 even when items
        # inside it fail, so the transport retry never sees them). Warn rather than raise: the
        # combination is legal and might be deliberate, just very unlikely.
        if self.max_retries_per_doc == 0 and self.transport_max_retries > 3:
            warnings.warn(
                f"transport_max_retries={self.transport_max_retries} is raised above the default but "
                "max_retries_per_doc=0, so documents Elasticsearch rejects with a 429 get NO "
                "retries. The _bulk API returns HTTP 200 even when individual items fail, so "
                "transport-level retries never see them. Set max_retries_per_doc, or use "
                "max_retries=<n> to set both layers at once.",
                # stacklevel=4, not 3: __post_init__ <- generated __init__ <- the umbrella wrapper
                # <- the caller. The wrapper added a frame, so 3 pointed inside config.py, which
                # blames the library for the caller's configuration. Pinned by a test that asserts
                # the warning's filename/lineno, since this is exactly the kind of off-by-one that
                # silently returns if the call chain changes again.
                UserWarning, stacklevel=4)
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


# The per-layer fields `max_retries` fans out to. Only the ones a class actually HAS are set, so the
# umbrella works on EsReadConfig (no per-document layer exists for a read) as well as write configs.
_RETRY_LAYER_FIELDS = ("transport_max_retries", "max_retries_per_doc")


def _support_max_retries_umbrella(cls):
    """Give `cls` a `max_retries=` constructor argument that sets EVERY retry layer at once.

    Two ways to configure retries, and you pick one:

      EsWriteConfig(..., max_retries=5)                  # simple: 5 at both layers
      EsWriteConfig(..., transport_max_retries=5,        # granular: tune each layer
                         max_retries_per_doc=2)

    Mixing them is REJECTED rather than resolved by precedence. If `max_retries=5` and
    `max_retries_per_doc=2` were both accepted, the effective per-document value would depend on an
    ordering rule nobody can see at the call site, which is exactly the class of ambiguity this
    config keeps trying to eliminate. An error naming both spellings is unmissable.

    `max_retries` is not a dataclass field: it is a pure constructor-level convenience that expands
    into the real fields. So `cfg.max_retries` is a read-only property (see below) rather than stored
    state, and there is never a stored value that could disagree with the layers it set.

    Applied AFTER @dataclass so it wraps the GENERATED __init__. Defining __init__ in the class body
    would make @dataclass skip generating one at all, leaving no `__dataclass_init__` to delegate to
    (verified). Wrapping keeps field order, defaults, inheritance, pickling and frozen-ness intact.
    """
    original = cls.__init__
    own_fields = {f.name for f in fields(cls)}
    layers = tuple(f for f in _RETRY_LAYER_FIELDS if f in own_fields)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if "max_retries" in kwargs:
            conflicts = sorted(f for f in layers if f in kwargs)
            if conflicts:
                raise ValueError(
                    f"pass either max_retries (the umbrella, which sets {' and '.join(layers)}) or "
                    f"the per-layer field(s) {', '.join(conflicts)}, not both. Mixing them would "
                    "make the effective retry count depend on an invisible precedence rule. "
                    f"For {cls.__name__}, max_retries={kwargs['max_retries']!r} is equivalent to "
                    + ", ".join(f"{f}={kwargs['max_retries']!r}" for f in layers) + ".")
            umbrella = kwargs.pop("max_retries")
            for f in layers:
                kwargs[f] = umbrella
        original(self, *args, **kwargs)

    # Advertise `max_retries` in the signature. functools.wraps copies the GENERATED __init__'s
    # signature, which does not mention `max_retries` (it is not a field), so `help(EsWriteConfig)`,
    # IDE completion and anything else using inspect.signature would not reveal a supported argument.
    # Append it as keyword-only so introspection matches what the constructor actually accepts.
    try:
        _sig = inspect.signature(original)
        _params = list(_sig.parameters.values())
        _params.append(inspect.Parameter("max_retries", inspect.Parameter.KEYWORD_ONLY,
                                         annotation=int))
        __init__.__signature__ = _sig.replace(parameters=_params)
    except (TypeError, ValueError):  # pragma: no cover - introspection is best-effort, never fatal
        pass

    cls.__init__ = __init__

    def _max_retries(self):
        """The umbrella retry setting, if every layer currently agrees; else None.

        Read-only and derived, because `max_retries` is a constructor convenience rather than stored
        state. It returns None under a granular configuration (the layers differ), which is honest:
        there is no single number that describes it.
        """
        values = {getattr(self, f) for f in layers}
        return next(iter(values)) if len(values) == 1 else None

    cls.max_retries = property(_max_retries)
    return cls


# Each class is wrapped independently, because @dataclass regenerates __init__ for every subclass, so
# wrapping only the base would leave the subclasses without the umbrella.
#
# KNOWN BOUNDARY: a FURTHER subclass defined by a caller (e.g. `@dataclass(frozen=True) class
# MyConfig(EsWriteConfig)`) regenerates __init__ again and is not wrapped, so `max_retries=` raises
# TypeError there while the per-layer fields keep working. That fails loudly rather than silently
# mis-configuring retries, and subclassing these configs is not part of the public API (construct
# them, don't extend them), so it is left as-is rather than solved with metaclass machinery.
# `_support_max_retries_umbrella` is importable if a caller genuinely needs it on their own subclass.
for _cls in (EsConnection, EsWriteConfig, EsReadConfig):
    _support_max_retries_umbrella(_cls)


# Backward-compatible alias: pre-0.4.0, the (write) config was named EsConfig. Keep it working so
# existing write code and tests don't break; new code should use EsWriteConfig / EsReadConfig.
EsConfig = EsWriteConfig
