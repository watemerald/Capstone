import tensorflow as tf
import requests
import re
from typing import Union, List
import pandas
from dataclasses import dataclass

record = "/media/watemerald/Seagate/data/yt8m/frame/train0001.tfrecord"

features = []
for example in tf.compat.v1.python_io.tf_record_iterator(record):
    tf_example = tf.train.Example.FromString(example)
    features.append(tf_example)


def expand_vid_id(short_id: Union[bytes, str]) -> str:
    """
    """
    # If the short_id is passed as bytes, that means that is was
    # decoded from a TFRecord directly, in which case it's a UTF-8
    # string
    if isinstance(short_id, bytes):
        short_id = short_id.decode("UTF-8")

    url = f"http://data.yt8m.org/2/j/i/{short_id[:2]}/{short_id}.js"
    val = requests.get(url)

    # The return format looks like i("02ab","tvvJFX90eh0");
    # with the short id on the left and full id on the right
    match = re.match(r"i\(\"(?P<short_id>\w{4})\".\"(?P<full_id>\w+)\"\);", val.text)

    return match.group("full_id")


vocabulary = pandas.read_csv("video-search/vocabulary.csv")


def label_id_to_name(label: int) -> str:
    """Converts a single label id number to its full Knowledge Graph Name
    """
    return vocabulary.iloc[label]["Name"]


@dataclass
class VideoInfo:
    short_id: str
    long_id: str
    tags: List[str]


def decode_tf_example(e: tf.train.Example) -> VideoInfo:
    short_id = e.features.feature["id"].bytes_list.value[0]
    labels = e.features.feature["labels"].int64_list.value

    long_id = expand_vid_id(short_id)
    tags = list(map(label_id_to_name, labels))
    return VideoInfo(short_id=short_id.decode("UTF-8"), long_id=long_id, tags=tags,)


from IPython.display import YouTubeVideo


def display_video(vid: VideoInfo) -> YouTubeVideo:

    return YouTubeVideo(vid.long_id)


print()
