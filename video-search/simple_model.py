from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Dropout,
    LeakyReLU,
    concatenate,
    Layer,
)
from tensorflow.keras.initializers import glorot_normal
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os
import pendulum
from multiprocessing import Pool
from typing import Iterator, Optional, Tuple, Iterator

# Adapted from https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb/code

FOLDER = "/media/watemerald/Seagate/data/yt8m/video/"


def ap_at_n(data: Tuple[np.ndarray, np.ndarray], n: Optional[int] = 20,) -> float:
    """
    Calculate the average precision at n items (ap@n)

    Args:
        data: a tuple of 1D numpy arrays, storing the prediction scores and actual label values
        n: the top n items to be considered for ap@n; If it's None then just calculate average precision over all labels
    
    Returns:
        float: the ap@n
    """
    predictions, actuals = data

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(
                "n must be 'None' or a positive integer." " It was '%s'." % n
            )

    ap = 0.0

    # Get the indexes that would sort the predictions by most confidence
    sortidx = np.argsort(predictions)[::-1]

    # Number of labeled classes
    numpos = np.size(np.where(actuals > 0))

    if numpos == 0:
        return 0

    if n is not None:
        numpos = min(numpos, n)

    delta_recall = 1.0 / numpos

    # Numer of true positives encoundered
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap


def mean_ap(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate the Mean Average Precision (mAP)

    Args:
        pred: the array of predicted labels
        actual: the array of the actual labels
    
    Returns:
        float: the mAP
    """
    lst = zip(list(pred), list(actual))

    # Use a thread pool to calculate the ap@n for each video in parallel
    with Pool() as pool:
        all = pool.map(ap_at_n, lst)

    return np.mean(all)


def tf_itr(
    tp: str = "test", batch: int = 1024
) -> Iterator[Tuple[np.array, np.array, np.array, np.array,]]:
    """
    Iterate over TFRecords of a certain type

    Args:
        tp: the type of record to be iterated over (can be "train", "test", or "validate")
        batch: the number of records to be yielded at once

    Yields:
        (array, array, array, array): a tuple of numpy arrays, (ids, audio, rgb, labels) where for a specific
            index i ids[i], audio[i], rgb[i], labels[i] all correspond to the same records info
    """
    # TFRecord files
    tfiles = sorted(glob.glob(os.path.join(FOLDER, f"{tp}*tfrecord")))

    print(f"total files in {tp} {len(tfiles)}")

    # Initialize the lists to store the ids, audio & visual information, and labels for the current batch
    ids, aud, rgb, lbs = [], [], [], []
    for fn in tfiles:
        for example in tf.data.TFRecordDataset(fn).as_numpy_iterator():
            # Parse a single record from the dataset and extract all its values
            tf_example = tf.train.Example.FromString(example)

            ids.append(
                tf_example.features.feature["id"]
                .bytes_list.value[0]
                .decode(encoding="UTF-8")
            )
            rgb.append(
                np.array(tf_example.features.feature["mean_rgb"].float_list.value)
            )
            aud.append(
                np.array(tf_example.features.feature["mean_audio"].float_list.value)
            )

            # Convert a list of labels into a 1D vector where all the labels are marked as 1
            yss = np.array(tf_example.features.feature["labels"].int64_list.value)
            # Hardcoded number of total classes. Maybe remove them in the future?
            out = np.zeros(4716).astype(np.int8)
            for y in yss:
                out[y] = 1

            lbs.append(out)

            # When the total number of ids reaches the batch number, yield the parsed values
            # to be processed later
            if len(ids) >= batch:
                yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
                ids, aud, rgb, lbs = [], [], [], []


def fc_block(x: Layer, n: int = 1024, d: float = 0.2) -> Layer:
    """
        Passes a TensorFlow layer through a block of layers

        Args:
            x: a TensorFlow layer to be passed through the block
            n: the number of neurons in the dense layer
            d: the dropout rate
        
        Returns:
            Layer: the resulting Tensorflow Layer
    """
    x = Dense(n, kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x


def build_model() -> Model:
    """
        Build a simple model

        Returns:
            Model: A simple YouTube-8M Video Level model
    """
    # Input 1 is the audio information
    in1 = Input((128,), name="x1")
    x1 = fc_block(in1)

    # Input 2 is the video information
    in2 = Input((1024,), name="x2")
    x2 = fc_block(in2)

    x = concatenate([x1, x2], 1)
    x = fc_block(x)
    out = Dense(4716, activation="sigmoid", name="output")(x)

    model = Model([in1, in2], out)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model


def train():
    if not os.path.exists("weights"):
        os.mkdir("weights")
    batch = 10 * 1024
    n_itr = 10
    n_eph = 100

    _, x1_val, x2_val, y_val = next(tf_itr("val"))

    model = build_model()
    cnt = 0
    start = pendulum.now()
    fmt = start.format("Y-MM-DD hh:mm:ss")
    print(f"Started at {fmt}")

    for e in range(n_eph):
        for d in tf_itr("train", batch):
            _, x1_trn, x2_trn, y_trn = d
            model.train_on_batch({"x1": x1_trn, "x2": x2_trn}, {"output": y_trn})
            cnt += 1
            if cnt % n_itr == 0:
                y_prd = model.predict(
                    {"x1": x1_val, "x2": x2_val}, verbose=False, batch_size=100
                )
                g = mean_ap(y_prd, y_val)
                print("val mAP %0.5f; epoch: %d; iters: %d" % (g, e, cnt))
                now = pendulum.now()
                fmt = now.format("Y-MM-DD hh:mm:ss")
                print(fmt)
                model.save_weights("weights/%0.5f_%d_%d.h5" % (g, e, cnt))


def conv_pred(el):
    t = 20
    idx = np.argsort(el)[::-1]
    return " ".join(["{} {:0.5f}".format(i, el[i]) for i in idx[:t]])


def predict():
    model = build_model()

    wfn = sorted(glob.glob("../weights/*.h5"))[-1]
    model.load_weights(wfn)
    print("loaded weight file: %s" % wfn)
    idx, x1_val, x2_val, _ = next(tf_itr("test", 10 * 1024))

    ypd = model.predict({"x1": x1_val, "x2": x2_val}, verbose=1, batch_size=32)
    del x1_val, x2_val

    with Pool() as pool:
        out = pool.map(conv_pred, list(ypd))

    df = pd.DataFrame.from_dict({"VideoId": idx, "LabelConfidencePairs": out})
    df.to_csv(
        "subm1", header=True, index=False, columns=["VideoId", "LabelConfidencePairs"]
    )


if __name__ == "__main__":
    train()