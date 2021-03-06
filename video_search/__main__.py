import os
from typing import Optional

import streamlit.cli
import typer
from typer import Argument, Option

from video_search.models.netvlad import NetVLADModel
from video_search.models.simple_model import SimpleModel
from video_search.utils import NeuralNetwork, create_logger, predict_url

app = typer.Typer()

log = create_logger(__name__, "file.log")

# Hyperparameters for the models
FOLDER = "/media/watemerald/Seagate/data/yt8m/video/"

BATCH_SIZE = 10 * 1024
# Netvlad is a more complex model, fat fewer tensors could be loaded at once
NETVLAD_BATCH_SIZE = 256

# Number of epochs
N_EPOCHS = 15

# Save the weights every N iterations
N_ITR = 500

# The number of latest models to keep to save disk space
# If None, then keep all
N_LATEST = 20

# Default output file for predictions
OUTFILE = "out1"

# Simple model parameters
SIMPLE_MODEL_DROPOUT_RATE = 0.25
SIMPLE_MODEL_BLOCK_NEURONS = 1024

# NetVLAD model parameters
NETVLAD_CLUSTER_SIZE = 128
NETVLAD_N_EXPERTS = 2


@app.command("train")
def train_model(
    media_folder: str = Option(
        FOLDER, help="The folder where the YouTube-8M files are stored"
    ),
    model: NeuralNetwork = Option(
        NeuralNetwork.simple, "--model", "-m", case_sensitive=False
    ),
    batch: int = Option(
        BATCH_SIZE, "--batch", "-b", help="Number of records to process per batch"
    ),
    epochs: int = Option(N_EPOCHS, help="Total number of epochs to train for"),
    save_interval: int = Option(N_ITR, help="How often to save intermediate results"),
    load_model: bool = Option(True, help="Load the latest model to train off of"),
    dropout_rate: float = Option(
        SIMPLE_MODEL_DROPOUT_RATE,
        help="The dropout rate for the simple model's fully connected block",
    ),
    hidden_neurons: int = Option(
        SIMPLE_MODEL_BLOCK_NEURONS,
        help="The number of neurons in the first layer in the simple model's fully connected block",
    ),
    cluster_size: int = Option(
        NETVLAD_CLUSTER_SIZE, help="The NetVLAD layer cluster size"
    ),
    n_experts: int = Option(
        NETVLAD_N_EXPERTS,
        "--experts",
        help="The number of experts in the Mixture-of-experts classifier",
    ),
    keep_n_latest: Optional[int] = Option(
        N_LATEST, "--latest", help="The number of latest model files to keep"
    ),
):
    kwargs = locals()
    log.info(f"Launching train function for model simple_model with arguments {kwargs}")

    if model == NeuralNetwork.netvlad:
        if batch == BATCH_SIZE:
            # Assume no batch size was given, set it to default
            batch = NETVLAD_BATCH_SIZE
            kwargs["batch"] = NETVLAD_BATCH_SIZE
        if batch > NETVLAD_BATCH_SIZE:
            raise ValueError(
                f"NetVLAD batch size must be <= {NETVLAD_BATCH_SIZE}, got {batch}"
            )

        m = NetVLADModel()
    else:
        m = SimpleModel()

    m.train(**kwargs)


@app.command("predict")
def predict_model(
    model: NeuralNetwork = Option(
        NeuralNetwork.simple, "--model", "-m", case_sensitive=False
    ),
    weights_file: Optional[str] = Argument(None),
    media_folder: str = Option(
        FOLDER, help="The folder where the YouTube-8M files are stored"
    ),
    batch: int = Option(BATCH_SIZE, help="Number of records to process per batch"),
    outfile: str = Option(OUTFILE, "-o", help="The output file"),
    calculate_map: bool = Option(
        False, "--map", help="Calculate average map of the test dataset instead"
    ),
):
    kwargs = locals()
    log.info(f"Launching train function for model simple_model with arguments {kwargs}")
    if model == NeuralNetwork.netvlad:
        if batch == BATCH_SIZE:
            # Assume no batch size was given, set it to default
            batch = NETVLAD_BATCH_SIZE
            kwargs["batch"] = NETVLAD_BATCH_SIZE
        if batch > NETVLAD_BATCH_SIZE:
            raise ValueError(
                f"NetVLAD batch size must be <= {NETVLAD_BATCH_SIZE}, got {batch}"
            )

        m = NetVLADModel()
    else:
        m = SimpleModel()

    m.predict(**kwargs)


@app.command("predict-url")
def predict_url_command(
    url: str,
    weights_file: Optional[str] = Argument(None),
    model: NeuralNetwork = Option(
        NeuralNetwork.simple, "--model", "-m", case_sensitive=False
    ),
):
    kwargs = locals()
    log.info(f"Launching train function for model simple_model with arguments {kwargs}")
    if model == NeuralNetwork.netvlad:
        m = NetVLADModel()
    else:
        m = SimpleModel()

    print(predict_url(url, m, weights_file=weights_file))


@app.command("streamlit")
def run_streamlit():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "streamlit.py")
    args = []
    streamlit.cli._main_run(filename, args)


if __name__ == "__main__":
    app()
