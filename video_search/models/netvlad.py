import os
import math

import tensorflow as tf
from tensorflow.keras.layers import Layer, ReLU, Softmax, Input, concatenate, Dense
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

import numpy as np

from tensorflow.keras.initializers import RandomUniform, Zeros


from video_search.utils import create_logger
from video_search.models.shared import NeuralNet, AUDIO_DATA, VIDEO_DATA, OUTPUT_CLASSES


TENSORBOARD_LOG_DIR = "logs/netvlad"
WEIGHTS_DIR = os.path.join(TENSORBOARD_LOG_DIR, "weights/")
DATA_FILE = os.path.join(TENSORBOARD_LOG_DIR, "data.json")

# Create a weights directory to save the checkpoint files
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

log = create_logger(
    __name__, ["file.log", os.path.join(TENSORBOARD_LOG_DIR, "netvlad.log")]
)

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

        a_sum = tf.reduce_sum(activation, -2, keepdims=True)

        a = tf.multiply(a_sum, self.cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(
            reshaped_input, [-1, self.max_samples, self.feature_size]
        )

        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = K.dot(vlad, self.hidden1_weights)

        return vlad

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])

    def get_config(self):
        config = {
            "feature_size": self.feature_size,
            "max_samples": self.max_samples,
            "cluster_size": self.cluster_size,
            "output_dim": self.output_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MoE(Layer):
    """Mixture-of-experts layer.
    Implements: y = sum_{k=1}^K g(v_k * x) f(W_k * x)

    Params:
    units: the number of hidden units
    n_experts: the number of experts
    expert_activation: ReLU
    gating_activation: Softmax
    """

    def __init__(self, units: int, n_experts: int, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.n_experts = n_experts

    def build(self, input_shape):
        input_dim = input_shape[-1]

        expert_init_lim = np.sqrt(3.0 / (max(1.0, float(input_dim + self.units) / 2)))
        gating_init_lim = np.sqrt(3.0 / (max(1.0, float(input_dim + 1) / 2)))

        self.expert_kernel = self.add_weight(
            shape=(input_dim, self.units, self.n_experts),
            initializer=RandomUniform(minval=-expert_init_lim, maxval=expert_init_lim),
            name="expert_kernel",
        )

        self.gating_kernel = self.add_weight(
            shape=(input_dim, self.n_experts),
            initializer=RandomUniform(minval=-gating_init_lim, maxval=gating_init_lim),
            name="gating_kernel",
        )

        self.expert_bias = self.add_weight(
            shape=(self.units, self.n_experts), initializer=Zeros, name="expert_bias"
        )

        self.gating_bias = self.add_weight(
            shape=(self.n_experts,), initializer=Zeros, name="gating_bias"
        )

        super().build(input_shape)

    def call(self, inputs):
        expert_outputs = tf.tensordot(inputs, self.expert_kernel, axes=1)
        expert_outputs = K.bias_add(expert_outputs, self.expert_bias)
        expert_outputs = ReLU()(expert_outputs)

        gating_outputs = K.dot(inputs, self.gating_kernel)
        gating_outputs = K.bias_add(gating_outputs, self.gating_bias)
        gating_outputs = Softmax()(gating_outputs)

        output = K.sum(
            expert_outputs
            * K.repeat_elements(
                K.expand_dims(gating_outputs, axis=1), self.units, axis=1
            ),
            axis=2,
        )

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {"units": self.units, "n_experts": self.n_experts}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NetVLADModel(NeuralNet):
    def __init__(self):
        super().__init__(TENSORBOARD_LOG_DIR, WEIGHTS_DIR, DATA_FILE, log)

    def build_model(
        self, netvlad_cluster_size: int = 256, n_experts: int = 2, **kwargs
    ) -> Model:
        """Builds a gated NetVLAD classification model

        Reference:
        Miech, Antoine, Ivan Laptev, and Josef Sivic.
        "Learnable pooling with context gating for video classification."
        arXiv preprint arXiv:1706.06905 (2017).
        """
        in1 = Input((AUDIO_DATA,), name="x1")
        x1 = NetVLAD(AUDIO_DATA, 1, netvlad_cluster_size, AUDIO_DATA)(in1)

        in2 = Input((VIDEO_DATA,), name="x2")
        x2 = NetVLAD(VIDEO_DATA, 1, netvlad_cluster_size, VIDEO_DATA)(in2)

        x = concatenate([x1, x2], 1)
        x = ContextGating()(x)

        x = MoE(OUTPUT_CLASSES, n_experts)(x)

        x = ContextGating()(x)

        out = Dense(OUTPUT_CLASSES, activation="sigmoid", name="output")(x)

        model = Model([in1, in2], out)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        return model
