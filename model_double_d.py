"""
Double Discriminator GAN architecture
@author: Adrian Kucharski
"""
import datetime
import math
import os
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union
import inspect
import numpy as np
from pyparsing import str_type
import tensorflow as tf
from skimage import io
from keras import Model, Sequential, initializers
from keras.layers import (
    BatchNormalization,
    GaussianDropout,
    Concatenate,
    Conv2D,
    UpSampling2D,
    Dropout,
    Input,
    LeakyReLU,
    ReLU,
    MaxPooling2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Activation,
    Reshape,
    Layer
)
from keras.losses import BinaryCrossentropy, MeanAbsoluteError, Hinge
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
from tqdm import tqdm
from dataset import DataIterator
import keras.backend as K
import pytictoc

tic_timer = pytictoc.TicToc()
np.set_printoptions(suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Noise(Layer):
    def __init__(self, size: List[int]):
        super().__init__()
        self.size = size

    def call(self, inputs, *args, **kwargs):
        return tf.random.normal(shape=(tf.shape(inputs)[0], *self.size))

    def get_config(self):
        return {"size": self.size}


class GAN_Model:
    def __init__(
        self,
        input_size=(256, 256, 25),
        output_size=(256, 256, 3),
        d_p_lr=1e-5,
        d_g_lr=1e-5,
        gan_lr=1e-5,
        gan_loss_weights: Tuple[float, float, float] = [1, 1, 100],
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.ini = initializers.initializers_v2.GlorotNormal()
        self.ini2 = initializers.initializers_v2.GlorotUniform()
        self.g_model = self._generator_model()
        self.d_model_patch = self._discriminator_patch_model()
        self.d_model_global = self._discriminator_global_model()
        self.gan_loss_weights = K.variable(gan_loss_weights)

        d_loss = BinaryCrossentropy()
        g_loss = MeanAbsoluteError()
        
        # Generator won't be trained directly
        # Just compile it
        self.g_model.compile()
        self.d_model_patch.compile(
            optimizer=Adam(d_p_lr, beta_1=0.0),
            loss=d_loss,
            metrics=["accuracy"]
        )
        self.d_model_global.compile(
            optimizer=Adam(d_g_lr, beta_1=0.0),
            loss=d_loss,
            metrics=["accuracy"]
        )

        # Disable a discriminator training during gan training
        self.d_model_patch.trainable = False
        self.d_model_global.trainable = False

        # Define GAN model
        self.gan = self._gan_model()
        self.gan.compile(
            optimizer=Adam(gan_lr, beta_1=0.0),
            loss=[d_loss, d_loss, g_loss],
            loss_weights=self.gan_loss_weights
        )

    def _gan_model(self):
        x = Input(self.input_size, name="mask")
        g_out = self.g_model(x)
        d_patch_out = self.d_model_patch([x, g_out])
        d_global_out = self.d_model_global([x, g_out])
        return Model(inputs=x, outputs=[d_patch_out, d_global_out, g_out], name="GAN")

    def _discriminator_patch_model(self):
        h = Input(self.input_size, name="mask")
        t = Input(self.output_size, name="image")
        kernels = 4
        rate = 0.2

        inputs = Concatenate()([h, t])
        x = Conv2D(64, kernels, padding="valid", use_bias=False,
                   strides=2, kernel_initializer=self.ini)(inputs)
        x = LeakyReLU(rate)(x)
        x = Dropout(0.5)(x)

        x = Conv2D(128, kernels, padding="valid", use_bias=False,
                   strides=2, kernel_initializer=self.ini)(x)
        x = LeakyReLU(rate)(x)
        x = Dropout(0.5)(x)

        x = Conv2D(256, kernels, padding="valid", use_bias=False,
                   strides=2, kernel_initializer=self.ini)(x)
        x = LeakyReLU(rate)(x)
        x = Dropout(0.5)(x)

        # x = MaxPooling2D((2, 2))(x)
        # x = Dropout(0.5)(x)
        # x = Conv2D(512, kernels, padding="valid", use_bias=False, kernel_initializer=self.ini)(x)
        # x = LeakyReLU(0.3)(x)

        x = Conv2D(1, kernels, activation='sigmoid', padding="valid", use_bias=False)(x)
        return Model(inputs=[h, t], outputs=x, name="discriminator_patch")

    def _discriminator_global_model(self):
        h = Input(self.input_size, name="mask")
        t = Input(self.output_size, name="image")
        kernels = 4
        rate = 0.2

        inputs = Concatenate()([h, t])
        x = Conv2D(64, kernels, padding="valid", use_bias=False,
                   strides=2, kernel_initializer=self.ini)(inputs)
        x = LeakyReLU(rate)(x)
        x = Dropout(0.5)(x)

        x = Conv2D(128, kernels, padding="valid", use_bias=False,
                   strides=2, kernel_initializer=self.ini)(x)
        x = LeakyReLU(rate)(x)
        x = Dropout(0.5)(x)

        
        x = Conv2D(256, kernels, padding="valid", use_bias=False,
                   strides=2, kernel_initializer=self.ini)(x)
        x = LeakyReLU(rate)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(128, activation=LeakyReLU(rate),
                  use_bias=False, kernel_initializer=self.ini)(x)
        x = Dense(1, activation='sigmoid', use_bias=False)(x)
        return Model(inputs=[h, t], outputs=x, name="discriminator_global")

    def _generator_model(self):
        kernels = 4
        depth = 6
        nc = 256
        fl = 64
        sd = self.input_size[0] // 2 ** depth
        
        H = h = Input(self.input_size, name="mask")
        z = Noise((256,))(h)
        n = Dense(sd * sd * nc)(z)
        x = Reshape((sd, sd, nc))(n)

        for p in reversed(range(1, depth + 1)):
            m = tf.image.resize(h, x.shape[1:3], method='nearest')
            x = Concatenate()([x, m])
            x = Conv2D(fl * p, kernels, padding='same', activation=LeakyReLU(0.2))(x)
            x = UpSampling2D()(x)
            x = Conv2D(fl * p, kernels, padding='same', activation='relu')(x)
            
        x = Conv2D(3, 3, padding='same', activation='tanh')(x)
        return Model(inputs=H, outputs=x, name="generator")

    def set_loss_weights(self, new_value: Tuple[float, float, float]):
        K.set_value(self.gan_loss_weights, new_value)


class GAN_Training(GAN_Model):
    def __init__(
        self,
        # GAN Model args
        input_size=(256, 256, 25),
        output_size=(256, 256, 3),
        d_p_lr=4e-4,
        d_g_lr=4e-4,
        gan_lr=1e-4,
        gan_loss_weights: Tuple[float, float, float] = [1, 1, 10],

        # GAN_Training args
        main_log_path: str = "logs",
        g_path_save: str = "generators",
        d_path_save: str = "discriminators",
        evaluate_path_save: str = "images",
        model_code_save: str = 'code',
        save_with_optimizer: bool = False,
        logging: bool = True,
        evaluate_per_step: int = None,
    ):
        super().__init__(input_size, output_size, d_p_lr, d_g_lr, gan_lr, gan_loss_weights)
        self.args_to_save = inspect.getargvalues(inspect.currentframe())

        timer = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.main_log_path = os.path.join(main_log_path, timer)
        self.g_path_save = g_path_save
        self.d_path_save = d_path_save
        self.evaluate_path_save = evaluate_path_save
        self.model_code_save = model_code_save
        self.save_with_optimizer = save_with_optimizer
        self.logging = logging
        self.evaluate_per_step = evaluate_per_step

        self._create_dirs()
        if self.logging:
            self.writer = tf.summary.create_file_writer(self.main_log_path)
            self._log_code()

    def _log_code(self):
        gt = inspect.getsource(GAN_Training)
        gm = inspect.getsource(GAN_Model)
        with open(os.path.join(self.model_code_save, 'gan.txt'), 'w') as file:
            file.write(gt)
            file.write('\n')
            file.write(gm)
            file.write('\n')
            file.write(str(self.args_to_save))

    def _evaluate(
        self,
        epoch: int,
        data: Tuple[np.ndarray, np.ndarray] = None,
        step: int = None
    ) -> Union[None, Tuple[np.ndarray]]:
        if data is not None:
            images = []
            if step is not None:
                path = os.path.join(self.evaluate_path_save, f'{epoch}_{step}')
            else:
                path = os.path.join(self.evaluate_path_save, str(epoch))

            Path(path).mkdir(parents=True, exist_ok=True)

            xdata, ydata = data
            pred = (self.g_model.predict_on_batch(xdata) + 1) / 2.0

            for i in range(len(xdata)):
                x, y = xdata[i], ydata[i]
                impath = os.path.join(path, f"org_{i}.png")
                if x.shape[-1] > 3:
                    x = np.argmax(x, axis=-1, keepdims=True)
                    x = np.concatenate([x, x, x], axis=-1)
                if x.max() > 2.0:
                    x = x / 255.0

                image = np.array(
                    np.concatenate([x, pred[i], (y + 1) / 2.0],
                                   axis=1) * 255, "uint8"
                )
                io.imsave(impath, image)
                images.append(image)
            return np.array(images, dtype="uint8")
        return None

    def load_models(self, g_path: str = None, d_path: str = None):
        if g_path:
            self.g_model.load_weights(g_path)
        if d_path:
            self.d_model_patch.load_weights(d_path)
        return self

    def _save_models(self, g_path: str = None, d_path: str = None):
        if self.logging:
            if g_path and self.g_path_save:
                path = os.path.join(self.g_path_save, g_path)
                self.g_model.save(
                    path, include_optimizer=self.save_with_optimizer)
            if d_path and self.d_path_save:
                path = os.path.join(self.d_path_save, d_path)
                self.d_model_patch.save(
                    path, include_optimizer=self.save_with_optimizer)

    def _write_log(self, names, metrics):
        if self.logging:
            with self.writer.as_default():
                for name, value in zip(names, metrics):
                    tf.summary.scalar(name, value)
                self.writer.flush()

    def _write_images(self, epoch: int, images: np.ndarray):
        if self.logging:
            with self.writer.as_default():
                tf.summary.image(
                    "Validation data",
                    images,
                    step=epoch,
                    max_outputs=len(images),
                    description="Mask|Generated|Orginal",
                )
            self.writer.flush()

    def _create_dirs(self):
        if self.logging:
            if self.g_path_save:
                self.g_path_save = os.path.join(
                    self.main_log_path, self.g_path_save)
            if self.d_path_save:
                self.d_path_save = os.path.join(
                    self.main_log_path, self.d_path_save)
            if self.model_code_save:
                self.model_code_save = os.path.join(
                    self.main_log_path, self.model_code_save)
            if self.evaluate_path_save:
                self.evaluate_path_save = os.path.join(
                    self.main_log_path, self.evaluate_path_save)

            for path in [
                self.g_path_save,
                self.d_path_save,
                self.evaluate_path_save,
                self.model_code_save,
            ]:
                if path:
                    Path(path).mkdir(parents=True, exist_ok=True)

    def _prepare_discriminator_labels(self, batch_size: int):
        # Prepare label arrays for D and GAN training
        patch_shape = (batch_size, *self.d_model_patch.output_shape[1:])
        global_shape = (batch_size, *self.d_model_global.output_shape[1:])

        real_labels_patch = tf.ones(patch_shape, dtype=tf.float32)
        real_labels_global = tf.ones(global_shape, dtype=tf.float32)
        fake_labels_patch = tf.zeros(patch_shape, dtype=tf.float32)
        fake_labels_global = tf.zeros(global_shape, dtype=tf.float32)

        labels_join_patch = tf.concat(
            [real_labels_patch, fake_labels_patch], axis=0)
        labels_join_global = tf.concat(
            [real_labels_global, fake_labels_global], axis=0)

        return (
            real_labels_patch, real_labels_global,
            fake_labels_patch, fake_labels_global,
            labels_join_patch, labels_join_global
        )

    def train(
        self,
        epochs: int,
        dataset: Tuple[np.ndarray],
        batch_size=16,
        save_per_epochs=5,
        log_per_steps=5,
        categorical_input: bool = True,
        random_rot90: bool = False
    ):
        # Prepare label arrays for D and GAN training
        (
            real_labels_patch, real_labels_global,
            _, _,
            labels_join_patch, labels_join_global
        ) = self._prepare_discriminator_labels(batch_size)

        # Init iterator
        data_it = DataIterator(dataset, batch_size,
                               as_categorical=categorical_input,
                               random_rot90=random_rot90)
        steps = len(data_it)
        assert steps > log_per_steps
        step_number = 0

        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1}/{epochs}')
            mdp, mdg, mg = [], [], []
            # Training discriminator loop
            for step, (gts, images_real), in enumerate(tqdm(data_it)):
                # Concatenate fake with true
                image_fake = self.g_model(gts)
                gts_join = tf.concat([gts, gts], axis=0)
                images_join = tf.concat([images_real, image_fake], axis=0)

                # Train discriminators on predicted and real and fake data
                metrics_d_patch = self.d_model_patch.train_on_batch(
                    [gts_join, images_join], labels_join_patch
                )
                metrics_d_global = self.d_model_global.train_on_batch(
                    [gts_join, images_join], labels_join_global
                )

                # Train generator via discriminator
                metrics_gan = self.gan.train_on_batch(
                    gts, [real_labels_patch, real_labels_global, images_real])

                # Store metrics in array
                mdp.append(metrics_d_patch)
                mdg.append(metrics_d_global)
                mg.append(metrics_gan)

                # Save evaluation
                if self.evaluate_per_step is not None and step_number % self.evaluate_per_step == 0:
                    images = self._evaluate(epoch, data_it[0], step_number)
                step_number += 1

                # Store generator and discriminator metrics
                if step % log_per_steps == log_per_steps - 1:
                    tf.summary.experimental.set_step(epoch * steps + step)
                    gan_mn = [m + '_gan' for m in self.gan.metrics_names]
                    dp_mn = [
                        m + '_d_patch' for m in self.d_model_patch.metrics_names]
                    dg_mn = [
                        m + '_d_global' for m in self.d_model_global.metrics_names]
                    self._write_log(gan_mn, metrics_gan)
                    self._write_log(dp_mn, metrics_d_patch)
                    self._write_log(dg_mn, metrics_d_global)

            # Call on epoch end on the dataset
            data_it.on_epoch_end()

            if self.evaluate_per_step is None:
                images = self._evaluate(epoch, data_it[0])
                self._write_images(epoch, images)

            if (epoch + 1) % save_per_epochs == 0:
                self._save_models(f"model_{epoch}.h5", f"model_{epoch}.h5")

            print('Discriminator patch: ', np.mean(
                mdp, axis=0), self.d_model_patch.metrics_names)
            print('Discriminator global: ', np.mean(
                mdg, axis=0), self.d_model_global.metrics_names)
            print('GAN: ', np.mean(mg, axis=0), self.gan.metrics_names)

    def summary(self):
        self.d_model_patch.summary()
        self.g_model.summary()
        self.gan.summary()
