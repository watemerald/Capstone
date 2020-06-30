import glob
import json
import os
import sys
from multiprocessing import Pool
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import pendulum
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
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

try:
    from .utils import create_logger
except Exception:
    import logging

    def create_logger(name: str, log_file: str) -> logging.Logger:
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)

        # create error file handler and set level to error
        handler = logging.FileHandler(log_file, "w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)

        return log


# Adapted from https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb/code

FOLDER = "/media/watemerald/Seagate/data/yt8m/video/"

BATCH_SIZE = 10 * 1024

# Number of epochs
N_EPOCHS = 100

# Save the weights every N iterations
N_ITR = 10


log = create_logger(__name__, "file.log")

TENSORBOARD_LOG_DIR = "logs/simple"
WEIGHTS_DIR = os.path.join(TENSORBOARD_LOG_DIR, "weights/")
DATA_FILE = os.path.join(TENSORBOARD_LOG_DIR, "data.json")

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=TENSORBOARD_LOG_DIR, histogram_freq=0, write_graph=True,
)


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
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # The number of records per batch
    batch = BATCH_SIZE

    # Save the weights each n_itr iterations
    n_itr = N_ITR

    n_epochs = N_EPOCHS

    # Load the first validation TFRecord file to use in
    # periodically evaluating performance
    _, x1_val, x2_val, y_val = next(tf_itr("val"))

    model = build_model()
    tensorboard.set_model(model)

    # number of batches that have been processed
    n = 0
    # number of epochs that have passed
    e_passed = 0
    # Number of iterations since epoch
    ise = 0

    data = None
    # Best mAP encoundered so far
    best_map = 0

    # Load JSON stats
    if os.path.isfile(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            best = data["best"]
            best_map = best["map"]

    else:
        # Create stats if they don't exist
        with open(DATA_FILE, "w") as f:
            data = {
                "best": {"iter": 0, "map": 0, "epoch": 0, "ise": 0, "file": ""},
                "runs": [],
            }
            json.dump({}, f)

    if load_model:
        if len(data["runs"]) == 0:
            log.info("No model to load, starting from 0")
        else:
            # Load the latest weight file
            latest = data["runs"][-1]
            wfn = latest["file"]
            model.load(wfn)
            log.info(f"Loaded weight file: {wfn}")
            e_passed = latest["epoch"]
            n = latest["iter"]
            ise = latest["ise"]

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
            loss = model.train_on_batch({"x1": x1_trn, "x2": x2_trn}, {"output": y_trn})

            # Keep track of total number of iterations and iterations since epoch start
            n += 1
            ise += 1

            # Every n_itr batches evaluate performance and save weight files
            if n % n_itr == 0:
                y_prd = model.predict(
                    {"x1": x1_val, "x2": x2_val}, verbose=False, batch_size=100
                )
                g = mean_ap(y_prd, y_val)
                now = pendulum.now()
                fmt = now.format("Y-MM-DD HH:mm:ss")
                log.info(fmt)
                log.info(f"val mAP {g:0.5f}; EPOCH: {e:d}; iters: {n:d}; ise: {ise:d}")

                # Weights file
                wfile = os.path.join(WEIGHTS_DIR, f"{g:0.5f}_{e:d}_{n:d}_{ise:d}.h5")
                model.save(wfile)

                # Save stats into data file
                data["runs"].append(
                    {"iter": n, "epoch": e, "map": g, "ise": ise, "file": wfile}
                )
                if g > best_map:
                    best_map = g
                    data["best"] = {
                        "iter": n,
                        "epoch": e,
                        "map": g,
                        "ise": ise,
                        "file": wfile,
                    }

                with open(DATA_FILE, "w") as f:
                    json.dump(data, f)

                tensorboard.on_epoch_end(n, {"loss": loss, "mAP": g})

        # Set to 0 to not skip any batches on further epocs
        nskip = 0
        ise = 0

    tensorboard.on_train_end(None)


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


def predict(file: str = "subm1"):
    """
        Make a prediction using the latest trained weights

        Args:
            file: the csv file that will hold the results
    """
    model = build_model()

    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            best = data["best"]
    except FileNotFoundError:
        log.error("No weight files saved. Can't make predictions")
        sys.exit(1)

    wfn = best["file"]

    model.load_weights(wfn)
    log.info(f"loaded weight file: {wfn}")

    ids = []
    ypd = []

    # Create empty prediction csv file
    df = pd.DataFrame.from_dict({"VideoId": ids, "LabelConfidencePairs": ypd})
    df.to_csv(
        file, header=True, index=False, columns=["VideoId", "LabelConfidencePairs"]
    )

    for d in tf_itr("test", 10 * 1024):
        idx, x1_val, x2_val, _ = d
        ypd = model.predict({"x1": x1_val, "x2": x2_val}, verbose=1, batch_size=32)

        with Pool() as pool:
            out = pool.map(conv_pred, list(ypd))

        # Append the results of the current batch to the output csv
        df = pd.DataFrame.from_dict({"VideoId": idx, "LabelConfidencePairs": out})
        df.to_csv(
            file,
            header=False,
            index=False,
            columns=["VideoId", "LabelConfidencePairs"],
            mode="a",
        )


if __name__ == "__main__":
    predict()
