import glob
import os
import re
from multiprocessing import Pool
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import pendulum
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    Layer,
    LeakyReLU,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorboard.plugins.hparams import api as hp

from .utils import create_logger

# Adapted from https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb/code

FOLDER = "/media/watemerald/Seagate/data/yt8m/video/"


log = create_logger(__name__, "file.log")


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
    tp: str = "test", batch: int = 1024, skip: int = 0
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

    log.info(f"total files in {tp} {len(tfiles)}")

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
                # Every time a batch is finished, skip it if required
                skip -= 1
                if skip < 0:
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


def train(load_model: bool = True):
    """
        Train the simple model

        Args:
            load_model: if true, load the latest weights file and train from it
    """

    # Create a weights directory to save the checkpoint files
    if not os.path.exists("weights"):
        os.mkdir("weights")

    # The number of records per batch
    batch = 10 * 1024

    # Save the weights each n_itr iterations
    n_itr = 10

    n_epochs = 100

    # Load the first validation TFRecord file to use in
    # periodically evaluating performance
    _, x1_val, x2_val, y_val = next(tf_itr("val"))

    model = build_model()

    # number of batches that have been processed
    n = 0
    # number of epochs that have passed
    e_passed = 0
    # Number of iterations since epoch
    ise = 0

    if load_model:
        # Load the best performing weights
        weight_pattern = os.path.join(os.path.dirname(__file__), "weights/*.h5")
        weights = glob.glob(weight_pattern)

        if len(weights) > 0:
            wfn = max(weights, key=os.path.getctime)
            model.load_weights(wfn)
            log.info(f"loaded weight file: {wfn}")

            # The weight file looks like this: weights/0.57366_0_20.h5
            # Parse it out to get the current epoch and iteration number
            match = re.match(
                r"weights/\d+\.\d+_(?P<epoch>\d+)_(?P<iter>\d+)_(?P<ise>\d+)\.h5", wfn
            )
            (e_passed, n, ise) = match.groups()

            # Convert the matched strings to ints
            e_passed = int(e_passed)
            n = int(n)
            ise = int(ise)

    start = pendulum.now()
    fmt = start.format("Y-MM-DD HH:mm:ss")
    log.info(f"Started at {fmt}")
    log.info(f"Starting at EPOCH {e_passed} iter {n}")

    # How many batches to skip on first processed epoch
    nskip = ise

    for e in range(e_passed, n_epochs):

        # Do batch training
        for d in tf_itr("train", batch=batch, skip=nskip):
            _, x1_trn, x2_trn, y_trn = d
            model.train_on_batch({"x1": x1_trn, "x2": x2_trn}, {"output": y_trn})

            # Keep track of total number of iterations and iterations since epoch start
            n += 1
            ise += 1

            # Every n_itr batches evaluate performance and save weight files
            if n % n_itr == 0:
                y_prd = model.predict(
                    {"x1": x1_val, "x2": x2_val}, verbose=False, batch_size=100
                )
                g = mean_ap(y_prd, y_val)
                log.info(f"val mAP {g:0.5f}; epoch: {e:d}; iters: {n:d}; ise: {ise:d}")
                now = pendulum.now()
                fmt = now.format("Y-MM-DD HH:mm:ss")
                log.info(fmt)
                model.save_weights(f"weights/{g:0.5f}_{e:d}_{n:d}_{ise:d}.h5")

        # Set to 0 to not skip any batches on further epocs
        nskip = 0
        ise = 0


def conv_pred(el, t: Optional[int] = None) -> str:
    """
        Convert a prediction to a formatted string
        
        Args:
            el: the predictions
            t: the number of top confidence labels to log (default: 20)
        
        Returns:
            str: the formatted string
    """
    if t is None:
        t = 20
    idx = np.argsort(el)[::-1]
    return " ".join([f"{i} {el[i]:0.5f}" for i in idx[:t]])


def predict():
    """
        Make a prediction using the latest trained weights
    """
    model = build_model()

    # Load the best newest weights
    weight_pattern = os.path.join(os.path.dirname(__file__), "weights/*.h5")
    weights = glob.glob(weight_pattern)

    if not weights:
        log.error(
            f"There are no weight files saved at {os.path.dirname(weight_pattern)}"
        )
        return

    wfn = max(weights, key=os.path.getctime)
    model.load_weights(wfn)
    log.info(f"loaded weight file: {wfn}")

    ids = []
    ypd = []

    # Create empty prediction csv file
    df = pd.DataFrame.from_dict({"VideoId": ids, "LabelConfidencePairs": ypd})
    df.to_csv(
        "subm1", header=True, index=False, columns=["VideoId", "LabelConfidencePairs"]
    )

    for d in tf_itr("test", 10 * 1024):
        idx, x1_val, x2_val, _ = d
        ypd = model.predict({"x1": x1_val, "x2": x2_val}, verbose=1, batch_size=32)

        with Pool() as pool:
            out = pool.map(conv_pred, list(ypd))

        # Append the results of the current batch to the output csv
        df = pd.DataFrame.from_dict({"VideoId": idx, "LabelConfidencePairs": out})
        df.to_csv(
            "subm1",
            header=False,
            index=False,
            columns=["VideoId", "LabelConfidencePairs"],
            mode="a",
        )


if __name__ == "__main__":
    train()
