import logging
import os
import re
import shlex
import subprocess
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas
import requests
import tensorflow as tf

from video_search.models.shared import NeuralNet


class NeuralNetwork(str, Enum):
    simple = "simple"
    netvlad = "netvlad"


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
    """
        Utility function to create a logger, setting up both file and stream handlers
        Args:
            name: what the logger name should be
            log_files: file or list of files where to save the logs
        Returns:
            Logger: the logger to be used in printing
    """
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


vocabulary = pandas.read_csv("video_search/vocabulary.csv")


def label_id_to_name(label: int) -> Optional[str]:
    """Converts a single label id number to its full Knowledge Graph Name
        Args:
            label: a lebel id, such as one produced by a model
        Returns:
            None: No such index found (There are 4716 classes possible for the model to infer)
                  but only 3800 of them have corresponding Knowledge Graph points
            str: The full name of the knoledge graph item
    """

    try:
        a = vocabulary.loc[vocabulary["Index"] == label].iloc[0]["Name"]
        if isinstance(a, str):
            return a
        return None
    except IndexError:
        # The model can extract 4716 classes, but the vocabulary only has 3863 classes
        # In this case, just return None
        return None


def expand_vid_id(short_id: Union[bytes, str]) -> Optional[str]:
    """Converts a short id to full YouTube video id

        Connects to YouTube-8M servers , e.g. for short id ABCD, connect to:
        http://data.yt8m.org/2/j/i/AB/ABCD.js
        Sometimes videos are removed from the short_id->long_id lookup, in that case
        return None
        Args:
            short_id: a UTF-8 string (as either str or bytes) of the short id
        Returns:
            str: the expanded full id
            None: no match found
    """
    # If the short_id is passed as bytes, that means that is was
    # decoded from a TFRecord directly, in which case it's a UTF-8
    # string
    if isinstance(short_id, bytes):
        short_id = short_id.decode("UTF-8")

    url = f"http://data.yt8m.org/2/j/i/{short_id[:2]}/{short_id}.js"
    val = requests.get(url)

    if val.status_code != 200:
        return None

    # The return format looks like i("02ab","tvvJFX90eh0");
    # with the short id on the left and full id on the right
    match = re.match(r"i\(\"(?P<short_id>\w{4})\".\"(?P<full_id>\w+)\"\);", val.text)

    return match.group("full_id")


log = create_logger(__name__, "file.log")


def run_process(cmd: str) -> subprocess.CompletedProcess:
    """Utility function to call a shell command
        Args:
            cmd: the command to run in a shell
    """
    log.info(cmd)
    return subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, check=True)


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
        video_out = os.path.join(temp_dir, "vid.mp4")

        # Start Docker container

        cmd = (
            f"docker run -dit -v {temp_dir}:/v_folder --name mediapipe mediapipe:latest"
        )
        run_process(cmd)

        # Download youtube video from given url
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
        bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
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
        run_process("rm -f /tmp/video_search/features.pb")


def predict_url(
    url: str, model: NeuralNet, weights_file: Optional[str] = None, n_best: int = 20
) -> np.array:
    """Make predictions on video located at url
    Args:
        url: the url the video is located at
        model: the model to use in predictions
        weights_file: the weights to load
        n_best: the number of most probable results to return
    Returns:
        array(tuple(int, str, float)): a numpy array of tuples containing
            information the id number, the category name, and the probability
    """
    # Dowenload the video and convert it to the appropriate format
    (rgb, audio) = url_to_mean_array(url)

    # Run predictions using the provided model (and weights file)
    pred = model.predict_single_video(
        mean_rgb=rgb, mean_audio=audio, weights_file=weights_file
    )

    # Reshape array to 1d list
    pred = pred.reshape(-1)

    # Generate the n most probable label ids
    label_ids = np.argpartition(pred, -n_best)[-n_best:]
    # Convert the ids to category names
    labels = list(map(label_id_to_name, label_ids))
    probabilities = pred[label_ids]

    return np.array(list(zip(label_ids, labels, probabilities)))
