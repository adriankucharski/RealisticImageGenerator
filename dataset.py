import multiprocessing
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
from typing import Tuple, List
import keras
from keras.utils import to_categorical
from skimage import transform, color, io
import matplotlib.pyplot as plt
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import requests
import cv2
import collections

class FixSizeOrderedDict(collections.OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)

def generate_colors(num_of_colors: int, seed: int = 0) -> np.ndarray:
    colors = np.random.Generator(np.random.PCG64(seed)).integers(32, 255, size=(num_of_colors, 3))
    return colors.astype(np.uint8)

def image_to_mask(images, model, feature_extractor) -> List[np.ndarray]:
    inputs = feature_extractor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    data: np.ndarray = np.array(
        logits.detach().numpy()).transpose((0, 2, 3, 1))
    image_mask = np.argmax(data, axis=-1, keepdims=True)
    return image_mask.astype(np.uint8)


def parse_csv(path: str = 'data/included_classes.csv', sep: str = ';'):
    csv = pd.read_csv(path, sep=sep)
    classes = {}
    for index, (name, inc) in enumerate(zip(csv['Name'], csv['Include'])):
        if inc:
            classes[name] = index
    return classes


def processing_images(paths: List[str], classes: dict, cval_add: int = 0, image_size: Tuple[int, int] = None) -> List[np.ndarray]:
    images = []
    for p in paths:
        im = io.imread(p)
        if image_size is not None:
            im = transform.resize(im, image_size, preserve_range=True)
        if len(im.shape) == 3:
            im = im[..., 0:1]
        else:
            im = im[..., np.newaxis]
        x = np.zeros_like(im, dtype=np.uint8)
        for i, val in enumerate(classes.values()):
            x[im == val + cval_add] = i + 1
        images.append(x)
    return images


def transform_masks(classes_path: str = 'data/lhq_256/included_classes.csv',
                    masks_path: str = 'data/lhq_256/annotations/',
                    batch_size: int = 64,
                    cval_add: int = 0,
                    cpu_limit: int = None,
                    image_size: Tuple[int, int] = None
                    ) -> np.ndarray:

    classes = parse_csv(classes_path)
    if not masks_path.endswith('*.png'):
        masks_path = os.path.join(masks_path, '*.png')
    files = glob(masks_path)
    files_batched = [files[i:i+batch_size]
                     for i in range(0, len(files), batch_size)]

    args = zip(
        files_batched,
        itertools.repeat(classes),
        itertools.repeat(cval_add),
        itertools.repeat(image_size),
    )
    if cpu_limit is None:
        cpu_limit = multiprocessing.cpu_count()
    with multiprocessing.Pool(min(multiprocessing.cpu_count(), cpu_limit)) as pool:
        data = pool.starmap(processing_images, tqdm(
            args, total=len(files_batched)))
        try:
            return np.concatenate(data, axis=0)
        except:
            _data = []
            for d in data:
                _data.extend(d)
            return _data


def load_dataset(masks_path: str, images_path: str, classes_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.isfile(masks_path) and masks_path.endswith('.npy'):
        x = np.load(masks_path)
    else:
        assert classes_path is not None
        x = transform_masks(classes_path, masks_path)

    if os.path.isfile(images_path) and images_path.endswith('.npy'):
        y = np.load(images_path)
    else:
        if not images_path.endswith('*.png'):
            images_path = os.path.join(images_path, '*.png')
        y = np.asarray([io.imread(p) for p in glob(images_path)])

    return x, y


class DataIterator(keras.utils.Sequence):
    def __init__(
        self,
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray],
        batch_size=32,
        shuffle: bool = True,
        random_rot90: bool = True,
        mirroring: bool = True,
        cache_size: int = 128
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.segmentation_maps, self.real_image, self.labels = dataset
        self.classes = self.labels.max() + 1
        self.shuffle = shuffle
        self.random_rot90 = random_rot90
        self.mirroring = mirroring
        self.cache = {}
        self.cache_size = cache_size
        self.on_epoch_end()

    def __len__(self) -> int:
        "Denotes the number of batches per epoch"
        return len(self.real_image) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generate one batch of data and returns: (masks, images, labels)"
        # Get batch from cache
        if index in self.cache.keys():
            return self.cache[index]

        # Clear cache
        self.cache = {}

        # Fill cache
        for _index in range(index, min(index + self.cache_size, self.__len__())):
            idx = np.s_[_index * self.batch_size: (_index + 1) * self.batch_size]
            x = (self.segmentation_maps[idx].astype(np.float32) - 127.5) / 127.5
            y = (self.real_image[idx].astype(np.float32) - 127.5) / 127.5
            z = to_categorical(self.labels[idx], self.classes)
            self.cache[_index] = (x, y, z)
            
        return self.cache[index]

    def on_epoch_end(self):
        if self.shuffle:
            seed = np.random.randint(100000)
            np.random.seed(seed)
            np.random.shuffle(self.segmentation_maps)
            np.random.seed(seed)
            np.random.shuffle(self.real_image)
            np.random.seed(seed)
            np.random.shuffle(self.labels)
            np.random.seed(None)
            
        if self.random_rot90:
            for i in range(0, len(self.segmentation_maps), self.batch_size):
                k = np.random.randint(0, 5)
                self.segmentation_maps[i:i+self.batch_size] = np.rot90(self.segmentation_maps[i:i+self.batch_size], k = k, axes=(1, 2))
                self.real_image[i:i+self.batch_size] = np.rot90(self.real_image[i:i+self.batch_size], k = k, axes=(1, 2))
                self.labels[i:i+self.batch_size] = np.rot90(self.labels[i:i+self.batch_size], k = k, axes=(1, 2))

        if self.mirroring:
            for i in range(0, len(self.segmentation_maps), self.batch_size):
                self.segmentation_maps[i:i+self.batch_size] = np.flip(self.segmentation_maps[i:i+self.batch_size], axis=2)
                self.real_image[i:i+self.batch_size] = np.flip(self.real_image[i:i+self.batch_size], axis=2)
                self.labels[i:i+self.batch_size] = np.flip(self.labels[i:i+self.batch_size], axis=2)

    def get_number_of_classes(self):
        return self.classes


def load_and_resize(path: str, size: Tuple[int, int]) -> np.ndarray:
    im = io.imread(path)
    im = cv2.resize(im, size).astype(np.uint8)
    return im


if __name__ == '__main__':
    if False:
        for i, path in enumerate(glob('data/ADE20K/annotations/*.png')):
            p = path.replace('annotations', 'annotations_256')
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            mim = io.imread(path)
            mim = transform.resize(mim, (256, 256), preserve_range=True)
            # io.imsave(p, mim.astype(np.uint8), check_contrast=False)

            q = path.replace('annotations', 'images')
            r = path.replace('annotations', 'images_256')
            Path(r).parent.mkdir(parents=True, exist_ok=True)
            im = io.imread(q.replace('png', 'jpg'))
            if len(im.shape) != 3:
                im = color.gray2rgb(im)
            im = transform.resize(im, (256, 256), preserve_range=True)
            io.imsave(r, im.astype(np.uint8), check_contrast=False)

    if False:
        data = transform_masks(
            classes_path='data/included_classes.csv',
            masks_path='data/lhq_256/annotations',
            batch_size=128
        )
        np.save('data/lhq_256/24_classes.npy', data.astype(np.uint8))
        
    if False:
        data = transform_masks(
            classes_path='data/included_classes.csv',
            masks_path='data/lhq_256/annotations',
            batch_size=128,
            image_size=(128, 128)
        )
        np.save('data/lhq_256/24_classes_128.npy', data.astype(np.uint8))

    if False:
        data = transform_masks(
            classes_path='data/included_classes.csv',
            masks_path='data/ADE20K/annotations_256',
            batch_size=128,
            cval_add=1
        )
        np.save('data/ADE20K/24_classes.npy', data.astype(np.uint8))

    if False:
        images = np.asarray([np.array(io.imread(p), dtype=np.uint8)
                            for p in glob('data/ADE20K/images_256/*.png')])
        np.save('data/ADE20K/images.npy', images)

    if False:
        images = np.asarray([np.array(io.imread(p), dtype=np.uint8)
                            for p in glob('data/lhq_256/images/*.png')])
        np.save('data/lhq_256/images.npy', images)
        
    if False:
        with multiprocessing.Pool(12) as pool:
            paths = glob('data/lhq_256/images/*.png')
            args = zip(
                paths,
                itertools.repeat([128, 128])
            )
            images = pool.starmap(load_and_resize, tqdm(args, total=len(paths)))
            images = np.concatenate(images, axis=0)
        np.save('data/lhq_256/images_128.npy', images)


    if False:
        d256 = np.load('data/lhq_256/images.npy')
        data= []
        for i in tqdm(range(len(d256))):
            im128from256 = cv2.resize(d256[i], (128, 128), interpolation = cv2.INTER_NEAREST)
            data.append(im128from256)
        np.save('data/lhq_256/images_128.npy', data)
        
    if True:
        classes = 24
        d256 = np.load('data/ADE20K/24_classes.npy')
        data = np.zeros((*d256.shape[:3], 3), dtype=np.uint8)
        vals = generate_colors(classes)
        for i in tqdm(range(len(d256))):
            for c in range(1, classes + 1):
                data[i][d256[i, ..., 0] == c] = vals[c - 1]
            
        np.save('data/ADE20K/24_classes_rbg.npy', data)

