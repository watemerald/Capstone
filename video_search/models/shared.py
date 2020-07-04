import glob
import json
from logging import Logger
import os
import sys
from multiprocessing import Pool
from typing import Iterator, List, Optional, Tuple


import numpy as np
import pandas as pd
import pendulum
import tensorflow as tf

from tensorflow.keras.models import Model


class NeuralNet:
    def __init__(
        self, TENSORBOARD_LOG_DIR: str, WEIGHTS_DIR: str, DATA_FILE: str, log: Logger
    ):
        self.TENSORBOARD_LOG_DIR = TENSORBOARD_LOG_DIR
        self.WEIGHTS_DIR = WEIGHTS_DIR
        self.DATA_FILE = DATA_FILE
        self.log = log
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=TENSORBOARD_LOG_DIR, histogram_freq=0, write_graph=True,
        )

    def build_model(self, *args, **kwargs) -> Model:
        raise NotImplementedError(
            "build_model should be implemented by inhereting classes"
        )

    def tf_itr(
        self,
        tp: str = "test",
        batch: int = 1024,
        skip: int = 0,
        *,
        media_folder: str,
        **kwargs,
    ) -> Iterator[Tuple[np.array, np.array, np.array, np.array]]:
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
        tfiles = sorted(glob.glob(os.path.join(media_folder, f"{tp}*tfrecord")))

        self.log.info(f"total files in {tp} {len(tfiles)}")

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

    def train(
        self,
        epochs: int,
        save_interval: int,
        batch: int,
        load_model: bool = True,
        **kwargs,
    ):
        """
            Train the simple model

            Args:
                load_model: if true, load the latest weights file and train from it
        """

        # Create a weights directory to save the checkpoint files
        if not os.path.exists(self.WEIGHTS_DIR):
            os.makedirs(self.WEIGHTS_DIR)

        # The number of records per batch
        batch = batch

        # Save the weights each n_itr iterations
        n_itr = save_interval

        n_epochs = epochs

        # Load the first validation TFRecord file to use in
        # periodically evaluating performance
        _, x1_val, x2_val, y_val = next(self.tf_itr("val", **kwargs))

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
        if os.path.isfile(self.DATA_FILE):
            with open(self.DATA_FILE, "r") as f:
                data = json.load(f)
                best = data["best"]
                best_map = best["map"]

        else:
            # Create stats if they don't exist
            with open(self.DATA_FILE, "w") as f:
                data = {
                    "best": {"iter": 0, "map": 0, "epoch": 0, "ise": 0, "file": ""},
                    "runs": [],
                }
                json.dump({}, f)

        if load_model:
            if len(data["runs"]) == 0:
                self.log.info("No model to load, starting from 0")
                model = self.build_model(**kwargs)
                self.tensorboard.set_model(model)
            else:
                # Load the latest weight file
                latest = data["runs"][-1]
                wfn = latest["file"]
                model = tf.keras.models.load_model(wfn)
                self.tensorboard.set_model(model)
                self.log.info(f"Loaded weight file: {wfn}")
                e_passed = latest["epoch"]
                n = latest["iter"]
                ise = latest["ise"]
        else:
            model = self.build_model(**kwargs)
            self.tensorboard.set_model(model)

        start = pendulum.now()
        fmt = start.format("Y-MM-DD HH:mm:ss")
        self.log.info(f"Started at {fmt}")
        self.log.info(f"Starting at EPOCH {e_passed} iter {n}")

        # How many batches to skip on first processed epoch
        nskip = ise

        for e in range(e_passed, n_epochs):

            # Do batch training
            for d in self.tf_itr("train", batch=batch, skip=nskip, **kwargs):
                _, x1_trn, x2_trn, y_trn = d
                loss = model.train_on_batch(
                    {"x1": x1_trn, "x2": x2_trn}, {"output": y_trn}
                )

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
                    self.log.info(fmt)
                    self.log.info(
                        f"val mAP {g:0.5f}; EPOCH: {e:d}; iters: {n:d}; ise: {ise:d}"
                    )

                    # Weights file
                    wfile = os.path.join(
                        self.WEIGHTS_DIR, f"{g:0.5f}_{e:d}_{n:d}_{ise:d}.h5"
                    )
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

                    with open(self.DATA_FILE, "w") as f:
                        json.dump(data, f)

                    self.tensorboard.on_epoch_end(n, {"loss": loss, "mAP": g})

            # Set to 0 to not skip any batches on further epocs
            nskip = 0
            ise = 0

        self.tensorboard.on_train_end(None)

    def predict(
        self,
        media_folder: str,
        batch: int,
        weights_file: Optional[str],
        outfile: Optional[str] = None,
        calculate_map: bool = False,
    ):
        """
            Make a prediction using the latest trained weights

            Args:
                weights_file: the weights file to use. If empty, use the best weights instead
                outfile: the csv file that will hold the results
                media_folder: the path where the YouTube-8M files are stored
                batch: number of records to load per batch
        """
        try:
            with open(self.DATA_FILE, "r") as f:
                data = json.load(f)
                best = data["best"]
        except FileNotFoundError:
            self.log.error("No weight files saved. Can't make predictions")
            sys.exit(1)

        if weights_file is None:
            wfn = best["file"]
        else:
            wfn = weights_file

        # model.load_weights(wfn)
        model = tf.keras.models.load_model(wfn)
        self.log.info(f"loaded weight file: {wfn}")

        if calculate_map:
            self.log.info("calculating mAP over the whole validation set")

            # The number of items per batch, along with the mean ap of each batch
            ap_per_batch: List[Tuple[int, float]] = []

            for d in self.tf_itr("val", batch, media_folder=media_folder):
                idx, x1_val, x2_val, lbs = d
                ypd = model.predict(
                    {"x1": x1_val, "x2": x2_val}, verbose=1, batch_size=32
                )

                m_ap = mean_ap(ypd, lbs)
                ap_per_batch.append((len(idx), m_ap))

                self.log.info(total_ap(ap_per_batch))

        else:
            self.log.info("createing a submission csv")
            ids = []
            ypd = []

            # Create empty prediction csv file
            df = pd.DataFrame.from_dict({"VideoId": ids, "LabelConfidencePairs": ypd})
            df.to_csv(
                outfile,
                header=True,
                index=False,
                columns=["VideoId", "LabelConfidencePairs"],
            )

            for d in self.tf_itr("test", batch, media_folder=media_folder):
                idx, x1_val, x2_val, _ = d
                ypd = model.predict(
                    {"x1": x1_val, "x2": x2_val}, verbose=1, batch_size=32
                )

                with Pool() as pool:
                    out = pool.map(conv_pred, list(ypd))

                # Append the results of the current batch to the output csv
                df = pd.DataFrame.from_dict(
                    {"VideoId": idx, "LabelConfidencePairs": out}
                )
                df.to_csv(
                    outfile,
                    header=False,
                    index=False,
                    columns=["VideoId", "LabelConfidencePairs"],
                    mode="a",
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


def total_ap(batches: List[Tuple[int, float]]) -> Tuple[int, float]:
    total = 0
    running_ap = 1.0
    for (n, ap) in batches:
        running_ap = (total * running_ap + n * ap) / (total + n)
        total += n

    return (total, running_ap)


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
