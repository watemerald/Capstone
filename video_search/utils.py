import logging
import os
from functools import lru_cache
from typing import List, Union

import tensorflow as tf


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


def create_logger(name: str, log_files: Union[List[str], str]) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)

    if not isinstance(log_files, list):
        log_files = [log_files]

    for f in log_files:
        # create error file handler and set level to error
        handler = logging.FileHandler(f, "a")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)

    return log
