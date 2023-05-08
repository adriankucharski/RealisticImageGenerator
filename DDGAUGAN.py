import datetime
from glob import glob
import inspect
import os

from pathlib import Path
from typing import List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from keras import (Model, optimizers)
from keras.layers import (Concatenate, Conv2D, Dense, Flatten, Input,
                          LeakyReLU, Reshape, UpSampling2D)
from skimage import io
from tqdm import tqdm
from keras import utils
from dataset import DataIterator, load_dataset
import re
from GAUGAN_utils import *

tf.get_logger().setLevel('ERROR')


class DDGauGAN:
    def __init__(self,
                 image_shape=(256, 256, 3),
                 num_classes: int = 25,
                 latent_dim: int = 256,
                 gen_lr=1e-4,
                 disc_lr=4e-4,
                 feature_loss_coeff=10,
                 vgg_feature_loss_coeff=0.1,
                 kl_divergence_loss_coeff=0.1,
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

        self.generator_loss = GeneratorLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss(feature_loss_coeff)
        self.vgg_loss = VGGFeatureMatchingLoss(vgg_feature_loss_coeff)
        self.kl_divergence_loss = KLDivergenceLoss(kl_divergence_loss_coeff)

    def _build_encoder(self):
        encoder_downsample_factor = 64
        input_image = Input(shape=self.image_shape)
        x = Downsample(encoder_downsample_factor, 3,
                       apply_norm=False)(input_image)
        x = Downsample(2 * encoder_downsample_factor, 3)(x)
        x = Downsample(4 * encoder_downsample_factor, 3)(x)
        x = Downsample(8 * encoder_downsample_factor, 3)(x)
        x = Downsample(8 * encoder_downsample_factor, 3)(x)
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
        x = ResBlock(filters=dim // 4)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim // 8)(x, mask)
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(filters=dim // 32)(x, mask)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(self.image_shape[-1], 4,
                   padding="same", activation='tanh')(x)
        return Model([latent, mask], x, name="generator")

    def _build_patch_discriminator(self):
        downsample_factor = 64
        filters_size = 4
        input_image_A = Input(
            shape=self.image_shape, name="discriminator_image_A")
        input_image_B = Input(
            shape=self.image_shape, name="discriminator_image_B")
        x = Concatenate()([input_image_A, input_image_B])
        x1 = Downsample(
            downsample_factor, filters_size, apply_norm=False)(x)
        x2 = Downsample(2 * downsample_factor, filters_size)(x1)
        x3 = Downsample(4 * downsample_factor, filters_size)(x2)
        x4 = Downsample(8 * downsample_factor, filters_size)(x3)
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
        x1 = Downsample(
            downsample_factor, filters_size, apply_norm=False)(x)
        x2 = Downsample(2 * downsample_factor, filters_size)(x1)
        x3 = Downsample(4 * downsample_factor, filters_size)(x2)
        x4 = Downsample(8 * downsample_factor, filters_size)(x3)
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
                 gen_lr: float = 1e-4,
                 disc_lr: float = 4e-4,
                 feature_loss_coeff=10,
                 vgg_feature_loss_coeff=0.1,
                 kl_divergence_loss_coeff=0.1,
                 double_disc=True,

                 # Logs args
                 main_log_path: str = "logs",
                 g_path_save: str = "generators",
                 d_path_save: str = "discriminators",
                 e_path_save: str = "encoders",
                 evaluate_path_save: str = "images",
                 model_code_save: str = 'code',
                 save_with_optimizer: bool = False,
                 logging: bool = True,
                 evaluate_per_step: int = None,
                 ) -> None:
        super().__init__(image_shape, num_classes, latent_dim, gen_lr, disc_lr,
                         feature_loss_coeff, vgg_feature_loss_coeff, kl_divergence_loss_coeff)
        self.double_disc = double_disc
        self.args_to_save = inspect.getargvalues(inspect.currentframe())

        timer = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.main_log_path = os.path.join(main_log_path, timer)
        self.g_path_save = g_path_save
        self.d_path_save = d_path_save
        self.e_path_save = e_path_save
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

    def _save_models(self, epoch: int):
        if self.logging:
            if self.g_path_save:
                path = os.path.join(self.g_path_save, f"model_{epoch}.h5")
                self.generator.save(
                    path, include_optimizer=self.save_with_optimizer)
            if self.e_path_save:
                path = os.path.join(self.e_path_save, f"model_{epoch}.h5")
                self.encoder.save(
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
            if self.e_path_save:
                self.e_path_save = os.path.join(
                    self.main_log_path, self.e_path_save)
            if self.model_code_save:
                self.model_code_save = os.path.join(
                    self.main_log_path, self.model_code_save)
            if self.evaluate_path_save:
                self.evaluate_path_save = os.path.join(
                    self.main_log_path, self.evaluate_path_save)

            for path in [
                self.g_path_save,
                self.d_path_save,
                self.e_path_save,
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
            g_loss_d_p = self.generator_loss(fake_d_p_output[-1])
            feature_loss_p = self.feature_matching_loss(
                real_d_p_output, fake_d_p_output
            )

            # Compute Global D losses conditionaly
            g_loss_d_g = feature_loss_g = 0
            if self.double_disc:
                g_loss_d_g = self.generator_loss(fake_d_g_output[-1])
                feature_loss_g = self.feature_matching_loss(
                    real_d_g_output, fake_d_g_output
                )

            # Compute G losses
            kl_loss = self.kl_divergence_loss(mean, variance)
            vgg_loss = self.vgg_loss(image, fake_image)

            # Sum all losses into one
            total_loss = g_loss_d_g + g_loss_d_p + kl_loss + \
                vgg_loss + feature_loss_p + feature_loss_g

        all_trainable_variables = (
            self.gan.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables)
        )
        return total_loss, feature_loss_p, feature_loss_g, vgg_loss, kl_loss

    @tf.function
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
        log_per_steps=5
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

        names = ["generator_loss", "feature_loss_p", "feature_loss_g",
                 "vgg_loss", "kl_loss", "d_p_loss", "d_g_loss"]

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
                self._save_models(epoch)
            for name, loss in zip(names, np.mean(losses, axis=0)):
                print(f'Loss: {name}\t =', loss)


if __name__ == '__main__':

    if True:
        # dataset = load_dataset('data/ADE20K/24_classes.npy', 'data/ADE20K/images.npy')
        # dataset = load_dataset('data/lhq_256/24_classes_rgb.npy', 'data/lhq_256/images.npy')

        dataset_part = 45_000 / 90_000
        maps = np.load('data/lhq_256/24_classes_rgb_median.npy')
        maps1 = maps[:int(len(maps) * dataset_part)]
        maps = None
        del maps

        imgs = np.load('data/lhq_256/images.npy')
        imgs1 = imgs[:int(len(imgs) * dataset_part)]
        imgs = None
        del imgs

        labels = np.load('data/lhq_256/24_classes_median.npy')
        labels1 = labels[:int(len(labels) * dataset_part)]
        labels = None
        del labels

        dataset = (
            maps1,
            imgs1,
            labels1,
        )

        print(dataset[0].dtype, dataset[1].dtype,
              dataset[1].max(), dataset[1].min())
        print(dataset[0].shape, dataset[1].shape, dataset[2].shape)


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
            'e_path_save': "encoders",
            'evaluate_path_save': "images",
            'model_code_save': 'code',
            'save_with_optimizer': False,
            'logging': True,
            'evaluate_per_step': 100,
        }


        gan = Trainer(**args)
        gan.train(60, dataset, save_per_epochs=1, batch_size=2)

    if False:
        batch_size = 4
        maps = np.load('data/lhq_256/24_classes_rgb.npy')[20000:21000]
        imgs = np.load('data/lhq_256/images.npy')[20000:21000]
        labels = np.load('data/lhq_256/24_classes.npy')[20000:21000]

        rpath = Path(f'R:/real')
        rpath.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(imgs))):
            io.imsave(str(rpath / f'{i}.png'), imgs[i])

        for gen_path in glob('logs/20230424-2358/generators/*'):
            P = Predictor(gen_path)
            epoch = re.search(r'_(.*?).h5', gen_path).group(1)
            dir_path = Path(f'R:/{epoch}')
            dir_path.mkdir(parents=True, exist_ok=True)
            for k in tqdm(range(0, len(labels) - batch_size + 1, batch_size)):
                im = P(labels[k:k+batch_size])
                for i, j in zip(range(len(im)), range(k, k + batch_size)):
                    io.imsave(str(dir_path / f'{j}.png'), im[i])

    if False:
        import pytorch_fid
        true_path = 'R:/real/'
        all_paths = [f'R:/{i}/' for i in range(30)]

        fids = pytorch_fid.fid_score.calculate_fid_multiple_paths(
            [true_path, *all_paths], 8, 'cuda', 2048, 0)
        print(fids)

    if False:
        maps = np.load('data/lhq_256/24_classes_rgb.npy')[20000:21000]
        imgs = np.load('data/lhq_256/images.npy')[20000:21000]
        labels = np.load('data/lhq_256/24_classes.npy')[20000:21000]

        indexes = [68, 105, 113, 153, 156, 249, 415, 423]

        for i in indexes:
            io.imsave(f'{i}.png', imgs[i])
        