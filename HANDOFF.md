# Production Readiness / Known Limitations (0.2.0)

`databricks-es-connector` 0.1.0 proves the **mechanism**: serverless Databricks can bulk-write
to Elasticsearch with gzip compression (measured ~7x on event-log NDJSON) and idempotent
deterministic IDs, and every Spark datatype is exportable. This document lists what a production
customer deployment still needs, so the gaps are explicit rather than assumed.

Treat everything under "Open items" as **not yet addressed** — work to be scoped and tracked
before a customer relies on it.

---

## Open items — cross-cloud & networking

- **Cross-partition connectivity (e.g. commercial AWS → AWS GovCloud).** GovCloud is a separate
  AWS partition (`aws-us-gov`). **PrivateLink does not cross partitions**, so a commercial→GovCloud
  hop cannot be a private VPC endpoint — it traverses the public internet or a customer-operated
  proxy/transit path. **Validate the actual network path in the customer environment before
  promising anything.**
- **Serverless egress IPs are a shifting NAT range, not a stable IP.** Do **not** allowlist a
  single observed egress address on the Elasticsearch firewall. Allowlist the documented
  serverless egress **CIDR range** for the workspace region (Databricks publishes these; they
  rotate over time and require periodic re-sync), or use private connectivity.
- **Egress cost premise is only half-measured.** Compression savings are measured (~7x). The
  cross-partition **data-transfer-out cost** and whether the path even works are not. Close the
  business case with a real cross-cloud test.

## Open items — security & compliance

- **Ship secure configuration, not the sandbox posture.** The library defaults are secure
  (`verify_certs=True`). Production must use `verify_certs=True` with the customer's CA
  (`EsConfig.ca_certs` supports pinning a CA bundle). Do not carry `verify_certs=False` +
  `basic_auth` (sandbox-only settings) into a customer deployment.
- **Auth: API key, not basic auth.** Production should use a least-privilege Elastic **API key**
  stored in a Databricks secret scope (`EsConfig(api_key=...)` is supported; no worked example
  ships yet).
- **FIPS / FedRAMP.** Unaddressed. GovCloud security workloads frequently require FIPS-validated
  crypto and FIPS endpoints — potentially a hard compliance gate, not a tuning knob. Confirm the
  `elasticsearch-py`/OpenSSL posture and TLS floor required.

## Open items — data durability & operations (deferred hardening)

These were consciously deferred to keep 0.1.0 focused. Needed before production for SIEM/audit data:

- **Error capture / dead-letter.** `bulk_write` returns `{written, deleted, errors}` but the per-doc
  error bodies from `streaming_bulk` are counted and discarded — a failed write is currently
  undiagnosable, and errored events are silently dropped. Add error-reason logging (at minimum) and
  a dead-letter path (for SIEM, silent event loss is a correctness/audit problem).
- **Backpressure / concurrency cap.** `mapInPandas` opens one ES client per Spark partition; a
  large Databricks cluster can overrun a modest ES cluster (429s). No coordinated throttle exists.
- **Index templates + ILM / data streams.** The connector writes to a single target index.
  Time-series SIEM data wants an index template + ILM (rollover, retention, tiers) or data streams.
- **Streaming: run as a job, and mind the semantics.** `make_foreach_batch` + `availableNow` +
  a UC-Volume checkpoint is the supported streaming shape, run inside a Databricks **job** (call
  `spark.streams.awaitAnyTermination()`). Driving it interactively cell-by-cell on serverless /
  Spark Connect is unreliable: `query.start()` can intermittently hang, and `awaitTermination(timeout)`
  does not honor its timeout there. Do not certify the interactive path.
- **Streaming freshness expectations.** On serverless only `availableNow`/`Once` triggers work, so
  end-to-end latency is job-cadence (≈30s–2min), not seconds. If a SIEM needs near-real-time, use a
  classic job cluster with `processingTime`, or a Kafka bridge. Set this expectation explicitly.
- **Per-run counts on serverless.** `query.recentProgress` read after an `availableNow` query
  terminates is empty client-side (Spark Connect), and a driver-local list mutated inside
  `on_batch` does not propagate back (`foreachBatch` runs server-side) — so neither is a reliable
  "rows written this run" counter. Measure the ES doc-count delta, or push metrics from inside the
  batch (StatsD, a Delta audit table).
- **Monitoring.** The `on_batch` hook is a seam for metrics but nothing sinks throughput / error
  rate / lag.
- **Updates & deletes.** Inserts/upserts via deterministic `_id` shipped in 0.1.0; **deletes shipped
  in 0.2.0** (`has_deletes` + `delete_flag_column`, emitting delete-by-`_id` bulk actions with
  scoped 404 no-op suppression). The connector deletes exactly the rows the caller flags — it does
  **not** dedup or order Change Data Feed rows itself. That is deliberate: dedup needs the caller's
  business sequencing column and should run distributed in Spark, not in executor memory (see the
  Change-Data-Feed pattern in the README). **Known limitation:** a caller that dedups per
  micro-batch is only correct within a batch — late-arriving data whose `event_ts` is older than a
  doc already in ES, but committed in a *later* batch, can clobber the newer doc. The memory-free
  fix is ES external versioning (`version` = `event_ts` epoch-millis, `version_type=external_gte`),
  deferred to keep the 0.2.0 API surface minimal.

## Open items — packaging & portability

- **Wheel install path is per-environment.** Notebooks `%pip install` from a specific UC Volume
  path; `%pip` cannot read a widget, so this line is edited per workspace. Documented in the
  README ("Deploying to a workspace").
- **Datatype coverage is complete, with two fidelity caveats.** Every Spark column is exportable
  with no caller pre-processing (verified end-to-end across all types): `coerce_value` handles all
  Arrow-crossable types (numerics, `decimal`→float, `binary`→base64, `date`/`timestamp`→epoch-millis,
  array/map/struct, plus a str fallback for anything unforeseen); `sanitize_for_arrow` (called by
  `bulk_write`) serializes the types that can't cross Arrow at all (`variant`, `interval`, at any
  nesting depth) to a JSON string. Field pruning
  (`drop_fields`) is a client opt-out to shrink payload, never a capability limit. Caveats to raise
  with a customer: (1) `decimal`→float loses precision beyond ~15-17 sig figs — cast to string in
  Spark if exact decimals matter (money/IDs); (2) added fields need matching ES mapping entries or
  ES dynamic-maps and guesses the type.
- **ES version compatibility.** The client is pinned `elasticsearch>=8,<9`; the 8.x client refuses a
  9.x cluster. Confirm the customer's ES major version and adjust.

---

## Pre-production checklist (minimum)

- [ ] Cross-partition network path validated in the customer environment
- [ ] `verify_certs=True` + customer CA, worked example
- [ ] API-key auth via secret scope, worked example
- [ ] FIPS/FedRAMP requirements confirmed
- [ ] Index template + ILM defined
- [ ] Error/dead-letter handling added
- [ ] Streaming run as a job with checkpoint + restart-idempotency
- [ ] ES major-version compatibility confirmed
