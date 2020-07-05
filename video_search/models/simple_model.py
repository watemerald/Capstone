import os

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

from ..utils import create_logger

from .shared import NeuralNet, AUDIO_DATA, VIDEO_DATA, OUTPUT_CLASSES

# Adapted from https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb/code


log = create_logger(__name__, "file.log")

TENSORBOARD_LOG_DIR = "logs/simple"
WEIGHTS_DIR = os.path.join(TENSORBOARD_LOG_DIR, "weights/")
DATA_FILE = os.path.join(TENSORBOARD_LOG_DIR, "data.json")


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


class SimpleModel(NeuralNet):
    def __init__(self):
        super().__init__(TENSORBOARD_LOG_DIR, WEIGHTS_DIR, DATA_FILE, log)

    def build_model(
        self, hidden_neurons: int = 1024, dropout_rate: float = 0.2, **kwargs
    ) -> Model:
        """
            Build a simple model
            Args:
                hidden_neurons: the number of neurons to be used in fc_block
                dropout_rate: the dropout rate to be used in fc_block

            Returns:
                Model: A simple YouTube-8M Video Level model
        """
        # Input 1 is the audio information
        in1 = Input((AUDIO_DATA,), name="x1")
        x1 = fc_block(in1, n=hidden_neurons, d=dropout_rate)

        # Input 2 is the video information
        in2 = Input((VIDEO_DATA,), name="x2")
        x2 = fc_block(in2, n=hidden_neurons, d=dropout_rate)

        x = concatenate([x1, x2], 1)
        x = fc_block(x, n=hidden_neurons, d=dropout_rate)
        out = Dense(OUTPUT_CLASSES, activation="sigmoid", name="output")(x)

        model = Model([in1, in2], out)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        return model
