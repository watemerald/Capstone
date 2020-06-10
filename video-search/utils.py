import tensorflow as tf
import os
from functools import lru_cache

# Memoize the number of total records, as it's a very expensive operation to compute
# Since there are only 3 types of records, no more cache is needed
@lru_cache(maxsize=3)
def count_records(folder: str, tp: str) -> int:
    """
        Count the number of TFRecords of certain type
        Args:
            folder: where the records are located
            tp: the type of record ("train", "test", "validate")
        Returns:
            int: the number of records

    """
    ret = 0
    for _ in tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(os.path.join(folder, f"{tp}*tfrecord"))
    ):
        ret += 1

    return ret
