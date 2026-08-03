"""databricks-es-connector: serverless-safe, bi-directional transfer between Databricks/Spark
and Elasticsearch.

Write: given a Spark DataFrame and an EsWriteConfig, `bulk_write` writes rows to an
Elasticsearch index via the `elasticsearch-py` client, parallelized across executors with
`mapInPandas` (serverless-safe; RDD APIs are not used), with gzip request compression and
deterministic document IDs. Schema-agnostic on write, every Spark datatype is exportable
with no caller pre-processing.

Read: given an EsReadConfig and a declared Spark schema, `read_index` pulls an index back into
a DataFrame, distributed across executors via a sliced Point-in-Time scroll (same `mapInPandas`
mechanism). The read schema is required (no mapping inference).
"""

from .config import EsConnection, EsWriteConfig, EsReadConfig, EsConfig
from .transform import to_es_source, coerce_value
from .bulk import bulk_write
from .stream import make_foreach_batch
from .spark_prep import sanitize_for_arrow
from .read import read_index
from .read_transform import read_coerce

__all__ = [
    "EsConnection",
    "EsWriteConfig",
    "EsReadConfig",
    "EsConfig",          # backward-compatible alias for EsWriteConfig (pre-0.4.0 name)
    "to_es_source",
    "coerce_value",
    "bulk_write",
    "make_foreach_batch",
    "sanitize_for_arrow",
    "read_index",
    "read_coerce",
]

__version__ = "0.4.2"
