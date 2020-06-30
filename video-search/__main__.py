import typer
from typer import Argument, Option
from pathlib import Path

from .simple_model import predict, train
from .utils import create_logger

app = typer.Typer()

log = create_logger(__name__, "file.log")

# Hyperparameters for the models
FOLDER = "/media/watemerald/Seagate/data/yt8m/video/"

BATCH_SIZE = 10 * 1024

# Number of epochs
N_EPOCHS = 100

# Save the weights every N iterations
N_ITR = 10


@app.command("train")
def train_model(
    media_folder: Path = Option(
        FOLDER, help="The folder where the YouTube-8M files are stored"
    ),
    batch: int = Option(BATCH_SIZE, help="Number of records to process per batch"),
    epochs: int = Option(N_EPOCHS, help="Total number of epochs to train for"),
    save_interval: int = Option(N_ITR, help="How often to save intermediate results"),
):
    typer.echo(f"media_folder: {media_folder}")
    typer.echo(f"batch: {batch}")
    typer.echo(f"epochs: {epochs}")
    typer.echo(f"save_interval: {save_interval}")

    typer.echo(f"locals: {locals()}")


@app.command("predict")
def predict_model():
    predict()


if __name__ == "__main__":
    app()
