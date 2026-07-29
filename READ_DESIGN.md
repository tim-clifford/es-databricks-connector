# Read path design — Elasticsearch index → Spark DataFrame (v0.4.0)

**Status:** design + Option-A spike (evaluation). Not the shipped distributed reader yet.
**Scope decision (locked):** full-index export at scale; read v1 requires an **explicit caller schema**
(no mapping inference in v1); fidelity mirrors the write path's documented contract.

The connector today is write-only (`bulk_write`, `make_foreach_batch`). This adds the reverse:
pull an ES index into a Spark DataFrame, serverless-safe, with the same fidelity guarantees we
already prove for writes.

---

## 1. Design constraints

### Serverless kills the obvious approach
A custom Spark `DataSource` with partition planning is how the old `elasticsearch-hadoop` built its
reader — via RDD APIs (`sparkContext.runJob`, custom RDDs). Those are **blocked on serverless**, the
same wall that forced the write path onto `mapInPandas`. So the distributed reader must use the same
escape hatch: `spark.range(n).mapInPandas(...)`, one task per ES slice, each task building its own
ES client from a serializable config. No driver-collected data on the scaled path.

### Fidelity mirrors the write contract (not more, not less)
The read path must return what the write path put in — **except** for the transforms the README
already documents as one-way. That is the whole fidelity bar: `write(df)` then `read(schema=df.schema)`
should reproduce the original DataFrame, modulo the documented deltas (decimal precision loss, etc.).

The consequence is decisive: **several stored values are ambiguous and cannot be inverted from
`_source` alone**, so the reader must be *told* the target type.

| Stored in ES `_source` | Could legitimately be | Invertible without a caller hint? |
|---|---|---|
| epoch-millis `long` | `timestamp`, `date`, or a genuine `long` | **No** |
| `float`/`double` (from decimal) | `decimal(p,s)` or `double` | **No** (and decimal was already lossy — documented) |
| base64 `string` (from binary) | `binary` or a real `keyword`/`text` | **No** |
| JSON `string` (from variant/interval) | `variant`/`interval` or a plain string | **No** |
| a scalar vs. a 1-element array | ES has no array type; any field may be multi-valued | **No** |

This is why **v1 requires an explicit `StructType`**. Mapping inference (`GET /_mapping`) can only
recover the unambiguous types (string, bool, int/long, double, object/nested) and would silently
violate the fidelity contract for the four rows above. Inference is deferred to a later version as a
clearly-labeled best-effort convenience, never the default for the ambiguous types.

---

## 2. Public API (v1)

```python
from databricks_es_connector import EsConfig, EsReadConfig, read_index
from pyspark.sql.types import StructType, StructField, TimestampType, StringType, LongType

schema = StructType([
    StructField("doc_id", StringType()),
    StructField("event_ts", TimestampType()),   # stored as epoch-millis long -> back to timestamp
    StructField("n", LongType()),
])

df = read_index(
    spark,
    EsConfig(hosts=..., api_key=..., index="my-index"),   # reuse the existing connection config
    schema,                                                # REQUIRED in v1
    read=EsReadConfig(
        query={"match_all": {}},        # optional ES query DSL (filter/pushdown)
        num_slices=None,                # default = index shard count; the parallelism knob
        batch_size=1000,                # docs per scroll/PIT page
        pit_keep_alive="1m",            # PIT lifetime per slice
    ),
)
```

- **`schema` is required and positional.** No inference in v1. A missing/empty schema raises, the same
  way `EsConfig` raises on missing `hosts`.
- **`EsReadConfig`** is a sibling frozen dataclass to `EsConfig` (serializable, shipped to executors).
  Kept separate so the write config surface is untouched. Open question 6.1 covers whether to merge.
- Reads default to including `_id` as a column when the schema declares it (symmetry with how writes
  keep the id in `_source`); TBD in design review.

---

## 3. The coercion layer (`read_transform.py`) — the correctness core

Mirror image of `transform.coerce_value`. Given an ES `_source` value and a target Spark
`DataType`, produce the Python/pandas value Spark expects for that type. Each branch is the
**documented inverse** of a write transform:

| Target Spark type | ES `_source` value | Inverse applied |
|---|---|---|
| `TimestampType` | epoch-millis `long` | `datetime.utcfromtimestamp(ms/1000)` (UTC), mirroring the write floor |
| `DateType` | epoch-millis `long` | epoch-millis → `date` (UTC midnight) |
| `BinaryType` | base64 `str` | `base64.b64decode` |
| `DecimalType(p,s)` | JSON number | `Decimal(str(v))` (precision already lost on write — documented) |
| `StringType` (variant/interval) | JSON/interval `str` | passthrough (caller re-parses variant if needed) |
| `Struct`/`Array`/`Map` | object/array | recurse per field/element/value against the sub-schema |
| null / missing field | absent or JSON null | `None` |
| scalars (bool/int/long/double/string) | same | passthrough with a type check |

This layer is **pure Python, no Spark, no ES** — unit-tested exactly like `transform.py`, and it is
the single oracle both the Option-A driver reader and the Option-B distributed reader share. Building
it first, provable cheaply, de-risks the whole feature before any distributed plumbing.

**Round-trip test (the acceptance bar):** for every type in the datatype-coverage matrix, assert
`read_coerce(coerce_value(x), target_type) == x`, except the documented-lossy cases (decimal,
sub-ms timestamp, float32 widening) which assert equality-within-the-documented-delta.

---

## 4. Transport

### Option A — driver-side scroll/PIT (the spike; also the v1 fallback for small reads)
Driver opens a PIT, pages through with `search_after`, coerces each hit via the layer above, and
`spark.createDataFrame(rows, schema)`. **Not distributed** — all data crosses the driver. Correct and
simple; fine for lookups/reference data up to ~low millions of small docs. Ships as the validation
harness and a documented "small read" entry point.

### Option B — sliced-scroll + `mapInPandas` fan-out (the scaled product)
```
pit_id = open_pit(index, keep_alive)          # driver: one consistent snapshot
spark.range(num_slices)                        # one Spark task per ES slice
     .mapInPandas(make_slice_reader(cfg, read, pit_id, schema), spark_schema)
close_pit(pit_id)                              # driver: after the job
```
Each task runs a [sliced scroll](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html#slice-scroll)
over the shared PIT (`{"slice": {"id": i, "max": num_slices}}`), builds its own ES client from the
serializable `EsConfig` (same pattern as `make_partition_writer`), pages its slice, coerces, and
yields pandas frames typed to `schema`. Distributed, serverless-safe, throughput scales with the
cluster — parity with the old connector's read.

**Hard parts to budget:** PIT lifecycle across tasks (open/close on driver, id shipped to executors),
per-slice resume/retry on transient failures, `num_slices` defaulting to shard count, and
back-pressure/paging. This is the bulk of the engineering.

---

## 5. Known ambiguities carried from the write contract

- **Arrays:** ES has no array type — any field can be multi-valued. The declared schema resolves it:
  a field typed `ArrayType` is read as a list (1-element if ES returned a scalar); a scalar type reads
  the first/only value. No `es.read.field.as.array.include`-style guessing needed *because the schema
  is explicit* — a direct payoff of the required-schema decision.
- **VARIANT/INTERVAL:** stored as strings on write; read back as `StringType`. Reconstructing a Spark
  `VARIANT` (`parse_json`) is a caller-side Spark step, documented — symmetric with the write side.
- **`text` vs `keyword`, multi-fields, `scaled_float`, geo:** all irrelevant in v1 — the caller's
  schema declares the Spark type; we don't consult the mapping.

---

## 6. Open questions for design review

1. **Config surface:** separate `EsReadConfig`, or extend `EsConfig` with read fields? Leaning
   separate (keeps the frozen write config untouched; read-only knobs don't pollute write validation).
2. **`_id` handling:** always expose `_id` as a column, only when the schema names it, or via a config
   flag? Symmetry argues for "when the schema declares the id_field."
3. **Query pushdown depth in v1:** accept a raw ES query DSL dict only, or also translate simple Spark
   filters? v1 = raw DSL only (simplest, explicit); Spark predicate pushdown is a later concern.
4. **Round-trip lossy cases:** confirm the accepted deltas are exactly those in the README datatype
   table (decimal precision, sub-ms timestamp, float32 widening) — nothing new introduced by reads.

---

## 7. Phasing

1. **This doc** + Option-A driver spike (`read.py` driver path + `read_transform.py` coercion layer),
   unit-tested, plus a live round-trip integration test extending the existing tier
   (`write(df)` → `read(df.schema)` == original, modulo documented deltas). Evaluation-grade.
2. **Distributed reader** (Option B): sliced-scroll `mapInPandas`, PIT lifecycle, live scale test.
3. **Docs + release:** README "Reading from Elasticsearch" section; ship in 0.4.0.

Later (not 0.4.0): mapping inference as labeled best-effort; Spark predicate/column pushdown.
