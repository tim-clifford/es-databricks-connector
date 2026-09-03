#!/usr/bin/env python
"""Connector-free Elasticsearch write benchmark (diagnostic; NOT a release gate).

Reads a REAL source table and writes to a REAL Elasticsearch URL, timing each step of the
write path SEPARATELY so you can see which layer dominates. Imports nothing from
databricks_es_connector -- plain Spark + pandas + elasticsearch-py -- so results are
independent of the library under investigation.

Run on the cluster you want to characterize (the one with the throughput problem), either as
a Databricks job `spark_python_task` or pasted into a notebook cell. Set the CONFIG block,
then read the per-stage wall times. The MARGINAL (stage N minus stage N-1) is the cost of the
one layer that stage adds:

  S1 read              read source + materialize every column (noop sink)   -> scan/read
  S2 read+repartition  + repartition(REPARTITION)                           -> the shuffle
  S3 build             + per-partition: pandas->dicts + build bulk actions  -> Arrow feed + per-row build
  S4 build+serialize   + json.dumps each doc                                -> serialization
  S5 build+send        + actually bulk-send to ES (no separate json step)   -> the ES HTTP path

Optional driver probe (RUN_DRIVER_PROBE): pull a sample to the driver and time streaming_bulk
and parallel_bulk against ES with NO Spark at all -> the raw per-process ingest rate of this
ES URL and whether concurrency helps, fully isolated from Spark/Arrow.

METHOD lets you compare "mapInPandas" (Arrow feed, works everywhere) against "foreachPartition"
(RDD, classic compute only) to isolate the JVM<->Python Arrow cost.

This script does NOT create or delete the ES index. Point ES_INDEX at a throwaway index and
clean it up yourself.
"""
import json
import time

from pyspark.sql import SparkSession, functions as F

# ============================= CONFIG (set these) =============================
SOURCE_TABLE = "catalog.schema.table"        # real source table to read
ROW_LIMIT    = 20_000_000                     # cap rows for a bounded run (0 = whole table)
REPARTITION  = 512                            # write parallelism; match prod (0 = no repartition)

ES_HOSTS     = "https://your-es-host:9200"    # ES URL
ES_API_KEY   = ""                             # base64 api key; leave "" to read from the secret below
ES_SECRET_SCOPE = "es_poc"                    # used only if ES_API_KEY == ""
ES_SECRET_KEY   = "api_key"
ES_INDEX     = "write_bench_throwaway"        # THROWAWAY target index (not created/deleted here)
VERIFY_CERTS = False                          # False for self-signed sandbox boxes

CHUNK_SIZE     = 1000                          # docs per _bulk request
WRITE_THREADS  = 1                             # >1 => concurrent bulk streams per partition (parallel_bulk)
METHOD         = "mapInPandas"                 # "mapInPandas" (Arrow) or "foreachPartition" (RDD, classic only)
ID_FIELD       = None                          # column to use as _id (None => ES auto-generates ids)

RUN_ES_STAGE      = True                        # S5: actually hit ES. False => Spark-side stages only
RUN_DRIVER_PROBE  = True                        # driver-only ES ingest probe (no Spark)
DRIVER_PROBE_ROWS = 200_000
# =============================================================================

spark = SparkSession.builder.getOrCreate()

# Resolve the api key ON THE DRIVER; executors close over it as a plain string.
API_KEY = ES_API_KEY
if not API_KEY:
    try:
        from pyspark.dbutils import DBUtils
        API_KEY = DBUtils(spark).secrets.get(ES_SECRET_SCOPE, ES_SECRET_KEY)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: ES_API_KEY empty and secret read failed ({type(e).__name__}: {e})")

results = {}


def timed(label, fn):
    t0 = time.time()
    fn()
    dt = round(time.time() - t0, 1)
    results[label] = dt
    rows = results.get("_rows")
    rate = f"({int(rows / dt):,} rows/s)" if dt and rows else ""
    print(f"[BENCH] {label}: {dt}s {rate}")


def _safe(v):
    """Minimal, connector-free coercion so a raw row value is JSON/ES-serializable."""
    if v is None:
        return None
    if isinstance(v, float):
        return None if v != v else v                      # NaN -> null
    if type(v).__module__ == "numpy":
        try:
            return v.item()
        except Exception:                                 # noqa: BLE001
            return str(v)
    if type(v).__name__ in ("Timestamp", "datetime", "date", "Timedelta"):
        try:
            return v.isoformat()
        except Exception:                                 # noqa: BLE001
            return str(v)
    if isinstance(v, (str, int, bool, dict, list)):
        return v
    return str(v)


def to_action(row):
    action = {"_index": ES_INDEX, "_source": {k: _safe(v) for k, v in row.items()}}
    if ID_FIELD:
        action["_id"] = str(row[ID_FIELD])
    return action


def es_client():
    from elasticsearch import Elasticsearch
    if not VERIFY_CERTS:
        import urllib3
        urllib3.disable_warnings()
    kw = {"hosts": ES_HOSTS, "verify_certs": VERIFY_CERTS, "request_timeout": 120}
    if API_KEY:
        kw["api_key"] = API_KEY
    return Elasticsearch(**kw)


def es_send(client, actions):
    from elasticsearch import helpers
    if WRITE_THREADS > 1:
        for _ in helpers.parallel_bulk(client, actions, thread_count=WRITE_THREADS,
                                       chunk_size=CHUNK_SIZE, raise_on_error=False,
                                       raise_on_exception=False):
            pass
    else:
        for _ in helpers.streaming_bulk(client, actions, chunk_size=CHUNK_SIZE,
                                        raise_on_error=False, raise_on_exception=False,
                                        yield_ok=True):
            pass


def process(records, do_serialize, do_send, client):
    """Per-partition write body, shared by both methods. `records` is an iterable of dicts."""
    actions = [to_action(r) for r in records]             # per-row build (the machinery)
    if do_serialize and not do_send:
        for a in actions:
            json.dumps(a["_source"], default=str)         # serialization only
    if do_send:
        es_send(client, actions)                          # ES HTTP (client serializes internally)
    return len(actions)


def run_mapinpandas(df, do_serialize, do_send):
    def _writer(pdf_iter):
        import pandas as pd
        client = es_client() if do_send else None
        total = 0
        for pdf in pdf_iter:
            total += process(pdf.to_dict("records"), do_serialize, do_send, client)
        yield pd.DataFrame({"n": [total]})
    df.mapInPandas(_writer, "n long").agg(F.sum("n")).collect()


def run_foreachpartition(df, do_serialize, do_send):
    def _part(row_iter):
        client = es_client() if do_send else None
        process((r.asDict(recursive=True) for r in row_iter), do_serialize, do_send, client)
    df.rdd.foreachPartition(_part)                          # classic compute only (RDD API)


def run_stage(df, do_serialize, do_send):
    runner = run_mapinpandas if METHOD == "mapInPandas" else run_foreachpartition
    runner(df, do_serialize, do_send)


def noop(df):
    df.write.format("noop").mode("overwrite").save()        # force full materialization, no output


# ================================== run ==================================
df = spark.read.table(SOURCE_TABLE)
if ROW_LIMIT:
    df = df.limit(ROW_LIMIT)
results["_rows"] = df.count()
print(f"rows={results['_rows']:,} cores(defaultParallelism)={spark.sparkContext.defaultParallelism} "
      f"method={METHOD} repartition={REPARTITION} chunk_size={CHUNK_SIZE} threads={WRITE_THREADS}")

timed("S1_read", lambda: noop(df))
df_w = df.repartition(REPARTITION) if REPARTITION else df
timed("S2_read_repartition", lambda: noop(df_w))
timed("S3_build", lambda: run_stage(df_w, False, False))
timed("S4_build_serialize", lambda: run_stage(df_w, True, False))
if RUN_ES_STAGE:
    timed("S5_build_send_es", lambda: run_stage(df_w, False, True))

for name, a, b in [("shuffle", "S2_read_repartition", "S1_read"),
                   ("arrow_build", "S3_build", "S2_read_repartition"),
                   ("serialize", "S4_build_serialize", "S3_build"),
                   ("es_send", "S5_build_send_es", "S4_build_serialize")]:
    if a in results and b in results:
        results[f"marginal_{name}"] = round(results[a] - results[b], 1)
        print(f"[MARGINAL] {name}: {results[f'marginal_{name}']}s")

# --- driver-only ES ingest probe (no Spark): raw per-process ingest rate of this ES URL ---
if RUN_DRIVER_PROBE:
    from elasticsearch import helpers
    print(f"\n[DRIVER PROBE] pulling {DRIVER_PROBE_ROWS:,} rows to the driver for a pure ES-send test")
    recs = df.limit(DRIVER_PROBE_ROWS).toPandas().to_dict("records")
    actions = [to_action(r) for r in recs]
    client = es_client()

    t0 = time.time()
    for _ in helpers.streaming_bulk(client, list(actions), chunk_size=CHUNK_SIZE,
                                    raise_on_error=False, raise_on_exception=False, yield_ok=True):
        pass
    dt = time.time() - t0
    print(f"[DRIVER PROBE] serial streaming_bulk: {dt:.1f}s ({int(len(actions) / dt):,} rows/s)")

    for tc in (2, 4, 8):
        t0 = time.time()
        for _ in helpers.parallel_bulk(client, list(actions), thread_count=tc, chunk_size=CHUNK_SIZE,
                                       raise_on_error=False, raise_on_exception=False):
            pass
        dt = time.time() - t0
        print(f"[DRIVER PROBE] parallel_bulk threads={tc}: {dt:.1f}s ({int(len(actions) / dt):,} rows/s)")

print("\n[BENCH-RESULTS-JSON]", json.dumps(results))
