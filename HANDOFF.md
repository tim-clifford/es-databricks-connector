# Production Readiness / Known Limitations (0.8.0)

`databricks-es-connector` proves the **mechanism** in both directions: serverless Databricks can
bulk-write to Elasticsearch with gzip compression (measured ~7x on event-log NDJSON) and idempotent
deterministic IDs (every Spark datatype is exportable), and can read an index back into a DataFrame
against a declared schema with write→read fidelity. This document lists what a production customer
deployment still needs, so the gaps are explicit rather than assumed.

Treat everything under "Open items" as **not yet addressed**: work to be scoped and tracked
before a customer relies on it.

---

## Open items: cross-cloud & networking

- **Cross-partition connectivity (e.g. commercial AWS → AWS GovCloud).** GovCloud is a separate
  AWS partition (`aws-us-gov`). **PrivateLink does not cross partitions**, so a commercial→GovCloud
  hop cannot be a private VPC endpoint: it traverses the public internet or a customer-operated
  proxy/transit path. **Validate the actual network path in the customer environment before
  promising anything.**
- **Serverless egress IPs are a shifting NAT range, not a stable IP.** Do **not** allowlist a
  single observed egress address on the Elasticsearch firewall. Allowlist the documented
  serverless egress **CIDR range** for the workspace region (Databricks publishes these; they
  rotate over time and require periodic re-sync), or use private connectivity.
- **Egress cost premise is only half-measured.** Compression savings are measured (~7x). The
  cross-partition **data-transfer-out cost** and whether the path even works are not. Close the
  business case with a real cross-cloud test.

## Open items: security & compliance

- **Ship secure configuration, not the sandbox posture.** The library defaults are secure
  (`verify_certs=True`). Production must use `verify_certs=True` with the customer's CA
  (`EsConfig.ca_certs` supports pinning a CA bundle). Do not carry `verify_certs=False` +
  `basic_auth` (sandbox-only settings) into a customer deployment.
- **Auth: API key, not basic auth.** Production should use a least-privilege Elastic **API key**
  stored in a Databricks secret scope (`EsConfig(api_key=...)` is supported; no worked example
  ships yet).
- **FIPS / FedRAMP.** Unaddressed. GovCloud security workloads frequently require FIPS-validated
  crypto and FIPS endpoints: potentially a hard compliance gate, not a tuning knob. Confirm the
  `elasticsearch-py`/OpenSSL posture and TLS floor required.

## Open items: data durability & operations (deferred hardening)

Hardening still needed before production for SIEM/audit data:

- **Error capture / dead-letter.** Partially addressed. `bulk_write` returns
  `{written, deleted, errors, ignored, coerced_nonfinite, total_input, unaccounted, error_samples}`:
  `error_samples` is a **bounded** sample (≤20) of per-doc failure diagnostics (`_id`, op, status, ES
  reason), and `unaccounted` reports below-per-doc loss directly (every row yields exactly one of
  written/deleted/errors/ignored). The streaming path raises on either signal, so a failure fails the
  batch instead of advancing the checkpoint past it. **Still missing:** a durable **dead-letter path**
  that captures *every* failed row (not just a sample): for SIEM/audit, silent loss of the un-sampled
  failures beyond the cap is a correctness gap. Route failures to a Delta table / DLQ (the `on_batch`
  hook runs before the raise, which is the seam for this).
- **Backpressure / concurrency cap.** `mapInPandas` opens one ES client per Spark partition; a
  large Databricks cluster can overrun a modest ES cluster (429s). Per-document retries with
  exponential backoff (`max_retries_per_doc`, default 3) absorb a transient 429, and a batch that still
  fails after retries fails the stream. That mitigates the symptom; a **coordinated** throttle
  (bounding total concurrent writers across executors) is **not built**.
- **Index templates + ILM / data streams.** The connector writes to a single target index.
  Time-series SIEM data wants an index template + ILM (rollover, retention, tiers) or data streams.
- **Streaming: run as a job, and mind the semantics.** `make_foreach_batch` + `availableNow` +
  a UC-Volume checkpoint is the supported streaming shape, run inside a Databricks **job** (call
  `spark.streams.awaitAnyTermination()`). Driving it interactively cell-by-cell on serverless /
  Spark Connect is unreliable: `query.start()` can intermittently hang, and `awaitTermination(timeout)`
  does not honor its timeout there. Do not certify the interactive path.
- **Checkpoint semantics: what is and is not retried.** Structured Streaming commits a micro-batch's
  offset when `foreachBatch` **returns normally**, with no visibility into the documents inside. So
  `make_foreach_batch` raises `EsWriteError` by default (`on_error="raise"`), failing the batch so
  Spark retries it and the offset holds; with `id_field` set that retry is an idempotent upsert.
  Operators who opt into `on_error="log"`/`"ignore"` are accepting **unretried loss** and need external
  alerting on the `errors`/`unaccounted` counts. Triage note: `ConnectionError`/`ConnectionTimeout`/
  `SerializationError` are not `ApiError` subclasses, so transport failures propagate and are retried
  by Spark regardless of policy; the policy governs per-document rejections.
- **Streaming freshness expectations.** On serverless only `availableNow`/`Once` triggers work, so
  end-to-end latency is job-cadence (≈30s to 2min), not seconds. If a SIEM needs near-real-time, use a
  classic job cluster with `processingTime`, or a Kafka bridge. Set this expectation explicitly.
- **Per-run counts on serverless.** `query.recentProgress` read after an `availableNow` query
  terminates is empty client-side (Spark Connect), and a driver-local list mutated inside
  `on_batch` does not propagate back (`foreachBatch` runs server-side), so neither is a reliable
  "rows written this run" counter. Measure the ES doc-count delta, or push metrics from inside the
  batch (StatsD, a Delta audit table).
- **Monitoring.** The `on_batch` hook is a seam for metrics but nothing sinks throughput / error
  rate / lag. A failed batch is visible as a failed Spark job, which is a floor, not a monitoring
  story.
- **Verifying a mapping needs an INDEXED read, not a round-trip.** A
  write-then-`read_index` comparison cannot detect a mapping mismatch: reads come from `_source`,
  which ES stores verbatim, so the check compares your data against a copy of itself. Verified live on
  8.19: `amount` dynamically mapped `long`, writing `100.75` gives `errors=0` and round-trips as
  `100.75`, while the indexed value is `100` and a `range > 100.5` query finds nothing. Any acceptance
  test that claims to validate mapping must read indexed values (`fields`, an aggregation, or
  `GET /_mapping`). Pre-creating explicit mappings is the real fix; an acceptance test should assert
  on the indexed value rather than on a `read_index` round-trip.
- **Updates & deletes.** Inserts/upserts via deterministic `_id`, and deletes via `has_deletes` +
  `delete_flag_column` (emitting delete-by-`_id` bulk actions with scoped 404 no-op suppression),
  are supported. A `delete_flag_column` that is not a column in the DataFrame fails the write before
  any document is touched, because otherwise no row is flagged and every intended delete is applied
  as an upsert while the counts still reconcile. The connector deletes exactly the rows the caller
  flags: it does **not** dedup or
  order Change Data Feed rows itself. That is deliberate: dedup needs the caller's business
  sequencing column and should run distributed in Spark, not in executor memory (see the
  Change-Data-Feed pattern in the README). **Known limitation:** a caller that dedups per
  micro-batch is only correct within a batch: late-arriving data whose `event_ts` is older than a
  doc already in ES, but committed in a *later* batch, can clobber the newer doc. The memory-free
  fix (not yet implemented) is ES external versioning (`version` = `event_ts` epoch-millis,
  `version_type=external_gte`).

## Open items: read path (`read_index`)

- **Explicit schema required; no mapping inference.** `read_index` requires the caller to declare a
  Spark `StructType`. This is deliberate: several write transforms are one-way (epoch-millis,
  decimal→float, base64, variant/interval→string) and can't be inverted from `_source` alone. A
  best-effort `GET /_mapping` inference for the unambiguous types is a possible future enhancement,
  but must never silently guess the ambiguous ones.
- **PIT keep-alive is a sliding window, and there is no automatic renewal.** `read_index` returns a
  *lazy* distributed DataFrame, so its Point-in-Time snapshot can't be driver-closed (that would kill
  the still-lazy read); it expires via `pit_keep_alive`. Each page request re-sends `pit_keep_alive`
  and resets the PIT's expiry, so it need only cover the longest gap *between* consecutive touches of
  the PIT (the lazy open-to-first-page scheduling gap, or a slow downstream consumer applying
  backpressure between pages), not the whole job's wall-clock. A gap longer than the window fails the
  late slice with an expired-PIT error; raise `pit_keep_alive` if a stage is slow to schedule or
  consume. Nothing renews the PIT outside those page touches.
- **No Spark predicate/column pushdown.** `read_index` accepts a raw ES query DSL dict
  (`EsReadConfig.query`) for server-side filtering, but does not translate Spark
  `.filter(...)`/`.select(...)` into ES queries. A caller wanting pushdown must express it as the raw
  query. A proper Spark `DataSource` with pushdown is out of scope (and blocked on serverless anyway).
- **Read throughput not yet benchmarked.** The sliced-scroll fan-out is proven correct (all docs,
  once, across shards) but has no throughput/large-index benchmark. `num_slices` defaults to the
  shard count; the optimal slice count and `batch_size` for a large export are untuned.
- **A declared type that doesn't fit raises `ReadSchemaMismatch`.** The trap to raise with a customer:
  **ES has no array type**, so any field can hold multiple values under the same mapping and
  `GET /_mapping` will not tell you which ones do. Declare `array<T>` wherever multi-values are
  possible. Integral floats (`3.0` → `3`) are accepted; lossy conversions are not.
- **A failed shard-count lookup raises rather than degrading silently.** With `num_slices=None` the
  reader looks up the shard count; if that call fails, the only correct fallback is a single serial
  reader, which forfeits all parallelism (and a `logging` warning is easy to miss on serverless, so a
  large export just looks slow). Pass `num_slices` explicitly in production to skip the lookup, or set
  `strict_slices=False` to accept the serial read.

## Open items: packaging & portability

- **Wheel install path is per-environment.** Notebooks `%pip install` from a specific UC Volume
  path; `%pip` cannot read a widget, so this line is edited per workspace. Documented in the
  README ("Deploying to a workspace").
- **Datatype coverage is complete, with two fidelity caveats.** Every Spark column is exportable
  with no caller pre-processing: `coerce_value` handles all Arrow-crossable types (numerics,
  `decimal`→float, `binary`→base64, `date`/`timestamp`→epoch-millis, array/map/struct, non-finite
  floats→null, plus a str fallback for anything unforeseen); `sanitize_for_arrow` (called by
  `bulk_write`) serializes the types that can't cross Arrow at all (`variant`, `interval`, at any
  nesting depth) to a string: `variant`→JSON string, scalar `interval`→its Spark string form.
  Field pruning
  (`drop_fields`) is a client opt-out to shrink payload, never a capability limit. Non-string
  `map` keys are rendered to strings via the same value transform; Spark maps are
  homogeneously typed so keys stay distinct. Caveats to raise with a customer: (1) `decimal`→float
  loses precision beyond ~15-17 sig figs: cast to string in Spark if exact decimals matter
  (money/IDs); (2) added fields need matching ES mapping entries or ES dynamic-maps and guesses the
  type; (3) a Spark `FLOAT` (32-bit) stores its exact widened value (`0.1`→`0.10000000149011612`),
  not the source literal: use `DOUBLE` if that matters; (4) `timestamp`→epoch-millis is floored to
  the millisecond (sub-ms precision dropped; ES `date` is ms-resolution: use `date_nanos` for finer).
  **`timestamp` timezone-independence:** a `timestamp`'s stored epoch is its true UTC instant
  regardless of `spark.sql.session.timeZone` (the connector converts via Spark `unix_millis` before
  export); `timestamp_ntz` is interpreted as UTC and reads back naive, `date` is unaffected.
- **ES version compatibility.** The client is pinned `elasticsearch>=8,<9`; the 8.x client refuses a
  9.x cluster. Confirm the customer's ES major version and adjust.

---

## Pre-production checklist (minimum)

- [ ] Cross-partition network path validated in the customer environment
- [ ] `verify_certs=True` + customer CA, worked example
- [ ] API-key auth via secret scope, worked example
- [ ] FIPS/FedRAMP requirements confirmed
- [ ] Index template + ILM defined
- [ ] Durable dead-letter path added (bounded `error_samples` + `unaccounted` exist and the stream fails on either; a full DLQ for every failed row is still needed)
- [ ] Streaming run as a job with checkpoint + restart-idempotency
- [ ] Streaming left on the default `on_error="raise"` (or external alerting on `errors`/`unaccounted` if not)
- [ ] Explicit index mappings pre-created, and verified by reading INDEXED values (not a `_source` round-trip)
- [ ] `num_slices` passed explicitly for large reads (so a shard-count lookup failure can't serialize the export)
- [ ] ES major-version compatibility confirmed
