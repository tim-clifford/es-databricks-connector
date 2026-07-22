"""databricks-es-connector: serverless-safe bulk export from Databricks/Spark to Elasticsearch.

Schema-agnostic. Given a Spark DataFrame and an EsConfig, writes rows to an
Elasticsearch index via the `elasticsearch-py` client, parallelized across
executors with `mapInPandas` (serverless-safe; RDD APIs are not used), with
gzip request compression and deterministic document IDs.
"""

from .config import EsConfig
from .transform import to_es_source, coerce_value
from .bulk import bulk_write, make_partition_writer
from .stream import make_foreach_batch
from .spark_prep import cast_unsupported_to_string

__all__ = [
    "EsConfig",
    "to_es_source",
    "coerce_value",
    "bulk_write",
    "make_partition_writer",
    "make_foreach_batch",
    "cast_unsupported_to_string",
]

__version__ = "0.1.0"
