from typing import List

import keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Model, Sequential, applications, initializers, utils
from keras.layers import Conv2D, Dropout, Input, Layer, LeakyReLU
from keras.losses import Hinge, Loss, MeanAbsoluteError

tf.get_logger().setLevel('ERROR')

class Noise(Layer):
    def __init__(self, size: List[int]):
        super().__init__()
        self.size = size

    def call(self, inputs, *args, **kwargs):
        return tf.random.normal(shape=(tf.shape(inputs)[0], *self.size))

    def get_config(self):
        return {"size": self.size}


class SPADE(Layer):
    def __init__(self, filters: int, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = Conv2D(128, 3, padding="same", activation="relu")
        self.conv_gamma = Conv2D(filters, 3, padding="same")
        self.conv_beta = Conv2D(filters, 3, padding="same")

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        output = gamma * normalized + beta
        return output

    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "conv": self.conv,
            "conv_gamma": self.conv_gamma,
            "conv_beta": self.conv_beta
        }


class ResBlock(Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.spade_1 = SPADE(input_filter)
        self.spade_2 = SPADE(self.filters)
        self.conv_1 = Conv2D(self.filters, 3, padding="same")
        self.conv_2 = Conv2D(self.filters, 3, padding="same")
        self.leaky_relu = LeakyReLU(0.2)
        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.spade_3 = SPADE(input_filter)
            self.conv_3 = Conv2D(self.filters, 3, padding="same")

    def call(self, input_tensor, mask):
        x = self.spade_1(input_tensor, mask)
        x = self.conv_1(self.leaky_relu(x))
        x = self.spade_2(x, mask)
        x = self.conv_2(self.leaky_relu(x))
        skip = (
            self.conv_3(self.leaky_relu(self.spade_3(input_tensor, mask)))
            if self.learned_skip
            else input_tensor
        )
        output = skip + x
        return output

    def get_config(self):
        return {"filters": self.filters}


class GaussianSampler(Layer):
    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, inputs):
        means, variance = inputs
        epsilon = tf.random.normal(
            shape=(tf.shape(means)[0], self.latent_dim), mean=0.0, stddev=1.0
        )
        samples = means + tf.exp(0.5 * variance) * epsilon
        return samples

    def get_config(self):
        return {"latent_dim": self.latent_dim}


class Downsample(Layer):
    def __init__(self,
                 channels: int,
                 kernels: int,
                 strides: int = 2,
                 apply_norm=True,
                 apply_activation=True,
                 apply_dropout=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.apply_norm = apply_norm
        self.apply_activation = apply_activation
        self.apply_dropout = apply_dropout

    def build(self, input_shape):
        self.block = Sequential([
            Conv2D(
                self.channels,
                self.kernels,
                strides=self.strides,
                padding="same",
                use_bias=False,
                kernel_initializer=initializers.initializers_v2.GlorotNormal(),
            )])
        if self.apply_norm:
            self.block.add(tfa.layers.InstanceNormalization())
        if self.apply_activation:
            self.block.add(LeakyReLU(0.2))
        if self.apply_dropout:
            self.block.add(Dropout(0.5))

    def call(self, inputs):
        return self.block(inputs)

    def get_config(self):
        return {
            "channels": self.channels,
            "kernels": self.kernels,
            "strides": self.strides,
            "apply_norm": self.apply_norm,
            "apply_activation": self.apply_activation,
            "apply_dropout": self.apply_dropout,
        }


class FeatureMatchingLoss(Loss):
    def __init__(self, coef: float = 10,  **kwargs):
        super().__init__(**kwargs)
        self.mae = MeanAbsoluteError()
        self.coef = coef

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(len(y_true) - 1):
            loss += self.mae(y_true[i], y_pred[i])
        return loss * self.coef


class VGGFeatureMatchingLoss(Loss):
    def __init__(self, coef: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = Model(vgg.input, layer_outputs, name="VGG")
        self.mae = MeanAbsoluteError()
        self.coef = coef

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true = applications.vgg19.preprocess_input(
            127.5 * (y_true + 1))
        y_pred = applications.vgg19.preprocess_input(
            127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * \
                self.mae(real_features[i], fake_features[i])
        return loss * self.coef


class DiscriminatorLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hinge_loss = Hinge()

    def call(self, y: tf.Tensor, is_real: bool):
        label = 1.0 if is_real else -1.0
        return self.hinge_loss(label, y)
    
class GeneratorLoss():
    def __init__(self):
        pass
    def __call__(self, y: tf.Tensor):
        return -tf.reduce_mean(y)
    
class KLDivergenceLoss(Loss):
    def __init__(self, coef: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        
    def call(self, mean: tf.Tensor, variance: tf.Tensor):
        return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance)) * self.coef

class Predictor():
    def __init__(self, model_g_path: str) -> None:
        print(self.__class__)
        custom_objects = {
            'ResBlock': ResBlock,
        }
        generator: Model = keras.models.load_model(
            model_g_path, custom_objects=custom_objects)
        inp = Input(generator.input_shape[1][1:])
        z = Noise(generator.input_shape[0][1:])(inp)
        out = generator([z, inp])
        self.model = Model(inp, out)
        self.num_classes = generator.input_shape[1][-1]

    def __call__(self, im: np.ndarray) -> np.ndarray:
        x = utils.to_categorical(im, self.num_classes)
        if len(x.shape) == 3:
            x = x[np.newaxis]
        x = np.array((self.model(x) + 1) * 127.5, np.uint8)
        return x[0] if x.shape[0] == 1 else x