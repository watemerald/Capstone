import os
import math

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

from ..utils import create_logger
from .shared import NeuralNet

log = create_logger(__name__, "file.log")

TENSORBOARD_LOG_DIR = "logs/netvlad"
WEIGHTS_DIR = os.path.join(TENSORBOARD_LOG_DIR, "weights/")
DATA_FILE = os.path.join(TENSORBOARD_LOG_DIR, "data.json")


# Adapted from https://github.com/antoine77340/LOUPE/blob/master/loupe.py
# Translated into custom keras layers


class ContextGating(Layer):
    """Creates a Context Gating layer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Create a trainable weight variable for this layer
        """
        self.gating_weights = self.add_weight(
            name="kernel_W1",
            shape=(input_shape[-1], input_shape[-1]),
            initializer=tf.random_normal_initializer(
                stddev=1 / math.sqrt(input_shape[-1])
            ),
            trainable=True,
        )
        self.gating_biases = self.add_weight(
            name="kernel_B1",
            shape=(input_shape[-1],),
            initializer=tf.random_normal_initializer(
                stddev=1 / math.sqrt(input_shape[-1])
            ),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        gates = K.dot(inputs, self.gating_weights)
        gates += self.gating_biases
        gates = tf.sigmoid(gates)

        activation = tf.multiply(inputs, gates)
        return activation

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)


class NetVLAD(Layer):
    """Creates a NetVLAD layer
    """

    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size

        super().__init__(**kwargs)

    def build(self, input_shape):
        """Create a trainable weight variable for this layer
        """
        self.cluster_weights = self.add_weight(
            name="kernel_W1",
            shape=(self.feature_size, self.cluster_size),
            initializer=tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.feature_size)
            ),
            trainable=True,
        )
        self.cluster_biases = self.add_weight(
            name="kernel_B1",
            shape=(self.cluster_size,),
            initializer=tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.feature_size)
            ),
            trainable=True,
        )
        self.cluster_weights2 = self.add_weight(
            name="kernel_W2",
            shape=(1, self.feature_size, self.cluster_size),
            initializer=tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.feature_size)
            ),
            trainable=True,
        )
        self.hidden1_weights = self.add_weight(
            name="kernel_H1",
            shape=(self.cluster_size * self.feature_size, self.output_dim),
            initializer=tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.cluster_size)
            ),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, reshaped_input):
        """Forward pass of a NetVLAD block
        """
        activation = K.dot(reshaped_input, self.cluster_weights)

        activation += self.cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        a = tf.multiply(a_sum, self.cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(
            reshaped_input, [-1, self.max_samples, self.feature_size]
        )

        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = K.dot(vlad, self.hidden1_weights)

        return vlad

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])


class NetVLADModel(NeuralNet):
    def __init__(self):
        super().__init__(TENSORBOARD_LOG_DIR, WEIGHTS_DIR, DATA_FILE, log)

    def build_model(self) -> Model:
        pass
