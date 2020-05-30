import tensorflow as tf
import requests
import re

record = "/media/watemerald/Seagate/data/yt8m/frame/train0001.tfrecord"

features = []
for example in tf.compat.v1.python_io.tf_record_iterator(record):
    tf_example = tf.train.Example.FromString(example)
    features.append(tf_example)


def expand_vid_id(short_id: str) -> str:
    url = f"http://data.yt8m.org/2/j/i/{short_id[:2]}/{short_id}.js"
    val = requests.get(url)

    # The return format looks like i("02ab","tvvJFX90eh0");
    # with the short id on the left and full id on the right
    match = re.match(r"i\(\"(?P<short_id>\w{4})\".\"(?P<full_id>\w+)\"\);", val.text)

    return match.group("full_id")


print()
