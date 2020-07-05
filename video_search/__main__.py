from enum import Enum
from typing import Optional

import typer
from typer import Argument, Option

from video_search.models.netvlad import NetVLADModel
from video_search.models.simple_model import SimpleModel
from video_search.utils import create_logger

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

# Default output file for predictions
OUTFILE = "out1"

# Simple model parameters
SIMPLE_MODEL_DROPOUT_RATE = 0.25
SIMPLE_MODEL_BLOCK_NEURONS = 1024

# NetVLAD model parameters
NETVLAD_CLUSTER_SIZE = 128
NETVLAD_N_EXPERTS = 2


class NeuralNetwork(str, Enum):
    simple = "simple"
    netvlad = "netvlad"


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


if __name__ == "__main__":
    app()
