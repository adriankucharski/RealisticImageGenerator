import datetime
import inspect
import os

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import (Model, Sequential, applications, initializers, optimizers)
from keras.layers import (Concatenate, Conv2D, Dense, Dropout, Flatten, Input,
                          Layer, LeakyReLU, Reshape, UpSampling2D, GaussianDropout, GaussianNoise)
from keras.losses import Loss, MeanAbsoluteError, Hinge
from skimage import io
from tensorflow import keras
from tqdm import tqdm
from keras import mixed_precision
from dataset import DataIterator

tf.get_logger().setLevel('ERROR')

BATCH_SIZE = 3
NUM_CLASSES = 25
IMG_HEIGHT = IMG_WIDTH = 256


def generator_loss(y: tf.Tensor):
    return -tf.reduce_mean(y)


def kl_divergence_loss(mean: tf.Tensor, variance: tf.Tensor):
    return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))


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
    def __init__(self, latent_dim, **kwargs):
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


class FeatureMatchingLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = MeanAbsoluteError()
        
    def call(self, y_true, y_pred):
        loss = 0
        for i in range(len(y_true) - 1):
            loss += self.mae(y_true[i], y_pred[i])
        return loss


class VGGFeatureMatchingLoss(Loss):
    def __init__(self, **kwargs):
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
        return loss


class DiscriminatorLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hinge_loss = Hinge()

    def call(self, y: tf.Tensor, is_real: bool):
        label = 1.0 if is_real else -1.0
        return self.hinge_loss(label, y)


class DDGauGAN:
    def __init__(self,
                 image_shape=(256, 256, 3),
                 num_classes: int = 25,
                 latent_dim: int = 256,
                 gen_lr=1e-4,
                 disc_lr=4e-4
                 ) -> None:
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.mask_shape = (*image_shape[:2], num_classes)

        self.discriminator_p = self._build_patch_discriminator()
        self.discriminator_g = self._build_global_discriminator()
        self.generator = self._build_generator()
        self.encoder = self._build_encoder()
        self.gan = self._build_combined_generator()
        self.sampler = GaussianSampler(latent_dim)

        self.generator_optimizer = optimizers.Adam(
            gen_lr, beta_1=0.0, beta_2=0.999
        )
        self.discriminator_p_optimizer = optimizers.Adam(
            disc_lr, beta_1=0.0, beta_2=0.999
        )
        self.discriminator_g_optimizer = optimizers.Adam(
            disc_lr, beta_1=0.0, beta_2=0.999
        )
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()

    def _downsample(
        self,
        channels,
        kernels,
        strides=2,
        apply_norm=True,
        apply_activation=True,
        apply_dropout=False,
    ):
        block = Sequential([
            Conv2D(
                channels,
                kernels,
                strides=strides,
                padding="same",
                use_bias=False,
                kernel_initializer=initializers.initializers_v2.GlorotNormal(),
            )])
        if apply_norm:
            block.add(tfa.layers.InstanceNormalization())
        if apply_activation:
            block.add(LeakyReLU(0.2))
        if apply_dropout:
            block.add(Dropout(0.5))
        return block

    def _build_encoder(self, encoder_downsample_factor=64):
        input_image = Input(shape=self.image_shape)
        x = self._downsample(encoder_downsample_factor, 3,
                             apply_norm=False)(input_image)
        x = self._downsample(2 * encoder_downsample_factor, 3)(x)
        x = self._downsample(4 * encoder_downsample_factor, 3)(x)
        x = self._downsample(8 * encoder_downsample_factor, 3)(x)
        x = self._downsample(8 * encoder_downsample_factor, 3)(x)
        x = Flatten()(x)
        mean = Dense(self.latent_dim, name="mean")(x)
        variance = Dense(self.latent_dim, name="variance")(x)
        return Model(input_image, [mean, variance], name="encoder")

    def _build_generator(self):
        dim = 1024
        latent = Input(shape=self.latent_dim)
        mask = Input(shape=self.mask_shape)
        x = Dense(4 * 4 * dim)(latent)
        x = Reshape((4, 4, dim))(x)
        x = ResBlock(filters=dim)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim // 2)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim // 2)(x, mask)
        x = UpSampling2D((2, 2))(x)
        # x = GaussianDropout(0.15)(x)
        x = ResBlock(filters=dim // 4)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim // 8)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim // 32)(x, mask)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(self.image_shape[-1], 4, padding="same", activation='tanh')(x)
        return Model([latent, mask], x, name="generator")

    def _build_patch_discriminator(self):
        downsample_factor = 64
        filters_size = 4
        input_image_A = Input(
            shape=self.image_shape, name="discriminator_image_A")
        input_image_B = Input(
            shape=self.image_shape, name="discriminator_image_B")
        x = Concatenate()([input_image_A, input_image_B])
        x1 = self._downsample(downsample_factor, filters_size, apply_norm=False)(x)
        x2 = self._downsample(2 * downsample_factor, filters_size)(x1)
        x3 = self._downsample(4 * downsample_factor, filters_size)(x2)
        x4 = self._downsample(8 * downsample_factor, filters_size)(x3)
        x5 = Conv2D(1, filters_size)(x4)
        outputs = [x1, x2, x3, x4, x5]
        return Model([input_image_A, input_image_B], outputs, name='patch_discriminator')

    def _build_global_discriminator(self):
        downsample_factor = 64
        filters_size = 4
        input_image_A = Input(
            shape=self.image_shape, name="discriminator_image_A")
        input_image_B = Input(
            shape=self.image_shape, name="discriminator_image_B")
        x = Concatenate()([input_image_A, input_image_B])
        x1 = self._downsample(downsample_factor, filters_size, apply_norm=False)(x)
        x2 = self._downsample(2 * downsample_factor, filters_size)(x1)
        x3 = self._downsample(4 * downsample_factor, filters_size)(x2)
        x4 = self._downsample(8 * downsample_factor, filters_size)(x3)
        x5 = Flatten()(x4)
        x5 = Dense(1)(x5)
        outputs = [x1, x2, x3, x4, x5]
        return Model([input_image_A, input_image_B], outputs, name='global_discriminator')

    def _build_combined_generator(self):
        self.discriminator_p.trainable = False
        self.discriminator_g.trainable = False
        mask_input = Input(shape=self.mask_shape, name="mask")
        image_input = Input(shape=self.image_shape, name="image")
        latent_input = Input(shape=(self.latent_dim), name="latent")
        g_out = self.generator([latent_input, mask_input])
        d_out_p = self.discriminator_p([image_input, g_out])
        d_out_g = self.discriminator_g([image_input, g_out])
        return Model(
            [latent_input, mask_input, image_input],
            [d_out_p, d_out_g, g_out],
            name="GAN"
        )


class Trainer(DDGauGAN):
    def __init__(self,
                 image_shape=(256, 256, 3),
                 num_classes: int = 25,
                 latent_dim: int = 256,
                 feature_loss_coeff=10,
                 vgg_feature_loss_coeff=0.1,
                 kl_divergence_loss_coeff=0.1,
                 double_disc = True,

                 # Logs args
                 main_log_path: str = "logs",
                 g_path_save: str = "generators",
                 d_path_save: str = "discriminators",
                 evaluate_path_save: str = "images",
                 model_code_save: str = 'code',
                 save_with_optimizer: bool = False,
                 logging: bool = True,
                 evaluate_per_step: int = None,
                 ) -> None:
        super().__init__(image_shape, num_classes, latent_dim)
        # Loss coeffs
        self.double_disc = double_disc
        self.feature_loss_coeff = feature_loss_coeff
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.kl_divergence_loss_coeff = kl_divergence_loss_coeff

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
        gt = inspect.getsource(DDGauGAN)
        gm = inspect.getsource(Trainer)
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

            segmentation_maps, imagesd, labels = data
            mean, variance = self.encoder(imagesd)
            latent_vector = self.sampler([mean, variance])
            pred = (self.generator([latent_vector, labels]) + 1) / 2.0

            for i in range(len(segmentation_maps)):
                x, y = segmentation_maps[i], imagesd[i]
                impath = os.path.join(path, f"org_{i}.png")
                if x.shape[-1] > 3:
                    x = np.argmax(x, axis=-1, keepdims=True)
                    x = np.concatenate([x, x, x], axis=-1)
                if x.max() > 2.0:
                    x = x / 255.0
                else:
                    x = (x + 1) / 2

                image = np.array(
                    np.concatenate([x, pred[i], (y + 1) / 2.0],
                                   axis=1) * 255, "uint8"
                )
                io.imsave(impath, image)
                images.append(image)
            return np.array(images, dtype="uint8")
        return None

    def _save_models(self, g_path: str = None):
        if self.logging:
            if g_path and self.g_path_save:
                path = os.path.join(self.g_path_save, g_path)
                self.generator.save(
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

    def _train_discriminators(self, latent_vector, segmentation_map, real_image, labels):
        total_loss_p = total_loss_g = 0
        fake_images = self.generator([latent_vector, labels])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator_p(
                [segmentation_map, fake_images])[-1]
            pred_real = self.discriminator_p(
                [segmentation_map, real_image])[-1]
            loss_fake = self.discriminator_loss(pred_fake, False)
            loss_real = self.discriminator_loss(pred_real, True)
            total_loss_p = 0.5 * (loss_fake + loss_real)

        self.discriminator_p.trainable = True
        gradients = gradient_tape.gradient(
            total_loss_p, self.discriminator_p.trainable_variables
        )
        self.discriminator_p_optimizer.apply_gradients(
            zip(gradients, self.discriminator_p.trainable_variables)
        )

        if self.double_disc:
            with tf.GradientTape() as gradient_tape:
                pred_fake = self.discriminator_g(
                    [segmentation_map, fake_images])[-1]
                pred_real = self.discriminator_g(
                    [segmentation_map, real_image])[-1]
                loss_fake = self.discriminator_loss(pred_fake, False)
                loss_real = self.discriminator_loss(pred_real, True)
                total_loss_g = 0.5 * (loss_fake + loss_real)

            self.discriminator_g.trainable = True
            gradients = gradient_tape.gradient(
                total_loss_g, self.discriminator_g.trainable_variables
            )
            self.discriminator_g_optimizer.apply_gradients(
                zip(gradients, self.discriminator_g.trainable_variables)
            )

        return total_loss_p, total_loss_g

    def _train_generator(
        self, latent_vector, segmentation_map, labels, image, mean, variance
    ):
        self.discriminator_p.trainable = False
        self.discriminator_g.trainable = False
        with tf.GradientTape() as tape:
            # Get networks outputs
            real_d_p_output = self.discriminator_p([segmentation_map, image])
            real_d_g_output = self.discriminator_g([segmentation_map, image])
            fake_d_p_output, fake_d_g_output, fake_image = self.gan(
                [latent_vector, labels, segmentation_map]
            )

            # Compute Patch D losses
            g_loss_d_p = generator_loss(fake_d_p_output[-1])
            feature_loss_p = self.feature_loss_coeff * self.feature_matching_loss(
                real_d_p_output, fake_d_p_output
            )

            # Compute Global D losses conditionaly
            g_loss_d_g = feature_loss_g = 0
            if self.double_disc:
                g_loss_d_g = generator_loss(fake_d_g_output[-1])
                feature_loss_g = self.feature_loss_coeff * self.feature_matching_loss(
                    real_d_g_output, fake_d_g_output
                )

            # Compute G losses
            kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image, fake_image)

            # Sum all losses into one
            total_loss = g_loss_d_g + g_loss_d_p + kl_loss + vgg_loss + feature_loss_p + feature_loss_g

        all_trainable_variables = (
            self.gan.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables)
        )
        return total_loss, feature_loss_p, feature_loss_g, vgg_loss, kl_loss

    def train_step(self, data):
        segmentation_map, image, labels = data
        mean, variance = self.encoder(image)
        latent_vector = self.sampler([mean, variance])
        d_p_loss, d_g_loss = self._train_discriminators(
            latent_vector, segmentation_map, image, labels
        )
        (generator_loss, feature_loss_p, feature_loss_g, vgg_loss, kl_loss) = self._train_generator(
            latent_vector, segmentation_map, labels, image, mean, variance
        )
        return generator_loss, feature_loss_p, feature_loss_g, vgg_loss, kl_loss, d_p_loss, d_g_loss

    def train(
        self,
        epochs: int,
        dataset: List[np.ndarray],
        batch_size=4,
        save_per_epochs=1,
        log_per_steps=5,
        random_rot90: bool = False
    ):
        # Init iterator
        data_it = DataIterator(
            dataset,
            batch_size,
            random_rot90=False, 
        )
        steps = len(data_it)
        assert steps > log_per_steps
        step_number = 0

        names = ["generator_loss", "feature_loss_p", "feature_loss_g", "vgg_loss", "kl_loss", "d_p_loss", "d_g_loss"]

        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1}/{epochs}')
            losses = []
            # Training GAN loop
            for step, (maps, images, labels), in enumerate(tqdm(data_it)):
                all_losses = self.train_step((maps, images, labels))

                # Store losses
                losses.append([loss.numpy() for loss in all_losses])

                # Save evaluation
                if self.evaluate_per_step is not None and step_number % self.evaluate_per_step == 0:
                    images = self._evaluate(epoch, data_it[0], step_number)
                step_number += 1

                # Store generator and discriminator metrics
                if step % log_per_steps == log_per_steps - 1:
                    tf.summary.experimental.set_step(epoch * steps + step)
                    self._write_log(names, all_losses)

            # Call on epoch end on the dataset
            data_it.on_epoch_end()

            if self.evaluate_per_step is None:
                images = self._evaluate(epoch, data_it[0])
                self._write_images(epoch, images)

            if (epoch + 1) % save_per_epochs == 0:
                self._save_models(f"model_{epoch}.h5")
            for name, loss in zip(names, np.mean(losses, axis=0)):
                print(f'Loss: {name}\t =', loss)


if __name__ == '__main__':
    # dataset = load_dataset('data/ADE20K/24_classes.npy', 'data/ADE20K/images.npy')
    # dataset = load_dataset('data/lhq_256/24_classes_rgb.npy', 'data/lhq_256/images.npy')

    dataset_part = 20_000 / 90_000
    maps = np.load('data/lhq_256/24_classes_rgb.npy')
    maps1 = maps[:int(len(maps) * dataset_part)]
    maps = None
    del maps
    
    imgs = np.load('data/lhq_256/images.npy')
    imgs1 = imgs[:int(len(imgs) * dataset_part)]
    imgs = None
    del imgs
    
    labels = np.load('data/lhq_256/24_classes.npy')
    labels1 = labels[:int(len(labels) * dataset_part)]
    labels = None
    del labels
    
    dataset = (
        maps1,
        imgs1,
        labels1,
    )

    args = {
        "image_shape": (256, 256, 3),
        "num_classes": 25,
        "latent_dim": 256,
        "feature_loss_coeff": 10,
        "vgg_feature_loss_coeff": 0.1,
        "kl_divergence_loss_coeff": 0.1,

        'main_log_path': "logs",
        'g_path_save': "generators",
        'd_path_save': None,  # "discriminators",
        'evaluate_path_save': "images",
        'model_code_save': 'code',
        'save_with_optimizer': False,
        'logging': True,
        'evaluate_per_step': 100,
    }

    print(dataset[0].dtype, dataset[1].dtype,
          dataset[1].max(), dataset[1].min())
    print(dataset[0].shape, dataset[1].shape, dataset[2].shape)


    gan = Trainer(**args)
    gan.train(100, dataset, save_per_epochs=1, batch_size=2, random_rot90=True)
