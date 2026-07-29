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
from .spark_prep import sanitize_for_arrow
from .read import read_index, EsReadConfig
from .read_transform import read_coerce

__all__ = [
    "EsConfig",
    "to_es_source",
    "coerce_value",
    "bulk_write",
    "make_partition_writer",
    "make_foreach_batch",
    "sanitize_for_arrow",
    "read_index",
    "EsReadConfig",
    "read_coerce",
]

__version__ = "0.4.0"
