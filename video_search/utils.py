import logging
import os
import shlex
import subprocess
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
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


def run_process(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)


def url_to_mean_array(url: str) -> Tuple[np.array, np.array]:
    """
    Download a video, run feature extraction using mediapipe on it,
    then calculate the elementwise mean for video and audio features to match
    the format the models were trained on

    Args:
        url: the url of the video to be downloaded

    Returns:
        (array, array): a tuple of numpy arrays, (rgb_features, audio_features)

    """
    try:
        # Create temporary directory to store the file
        temp_dir = "/tmp/video_search"

        # Start Docker container

        cmd = (
            f"docker run -dit -v {temp_dir}:/v_folder --name mediapipe mediapipe:latest"
        )
        run_process(cmd)

        # Download youtube video from given url

        video_out = os.path.join(temp_dir, "vid.mp4")

        cmd = f"youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' '{url}' -o {video_out}"
        run_process(cmd)

        # Build and run the inference on the provided file
        # Following instructions from
        # https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/youtube8m

        cmd = "docker exec mediapipe \
        python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
        --path_to_input_video=/v_folder/vid.mp4 \
        --clip_end_time_sec=600"

        run_process(cmd)

        cmd = "docker exec mediapipe \
        bazel build -c opt --linkopt=-s \
        --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
        mediapipe/examples/desktop/youtube8m:extract_yt8m_features"

        run_process(cmd)

        cmd = "docker exec mediapipe \
        GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
        --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
        --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
        --output_side_packets=output_sequence_example=/v_folder/features.pb"

        run_process(cmd)

        # Because of the attached volume both the video and features file will available
        # from the temp directory

        features_file = os.path.join(temp_dir, "features.pb")

        sequence_example = open(features_file, "rb").read()
        example = tf.train.SequenceExample.FromString(sequence_example)

        rgb_features = example.feature_lists.feature_list["RGB/feature/floats"].feature
        audio_features = example.feature_lists.feature_list[
            "AUDIO/feature/floats"
        ].feature

        rgb_features = np.array(list((f.float_list.value for f in rgb_features)))
        audio_features = np.array(list((f.float_list.value for f in audio_features)))

        # Calculate the elementwize mean for each list of lists
        # to fit the format the models were trained on
        mean_rgb = np.mean(rgb_features, axis=0)
        mean_audio = np.mean(audio_features, axis=0)

        return (mean_rgb, mean_audio)

    finally:
        # Clean up docker state
        run_process("docker stop mediapipe")
        run_process("docker container rm mediapipe")
        run_process(f"rm {video_out}")
