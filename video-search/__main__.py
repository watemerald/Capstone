import typer

from .simple_model import predict

app = typer.Typer()


# Hyperparameters for the models


@app.command()
def train():
    print("Let's get training")


@app.command("predict")
def predict_model():
    predict()


if __name__ == "__main__":
    app()
