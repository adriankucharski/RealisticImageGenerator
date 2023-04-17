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

def image_to_mask(images, model, feature_extractor) -> List[np.ndarray]:
    inputs = feature_extractor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    data: np.ndarray = np.array(logits.detach().numpy()).transpose((0, 2, 3, 1))
    image_mask = np.argmax(data, axis=-1, keepdims=True)
    return image_mask.astype(np.uint8)

def parse_csv(path: str = 'data/included_classes.csv', sep: str = ';'):
    csv = pd.read_csv(path, sep=sep)
    classes = {}
    for index, (name, inc) in enumerate(zip(csv['Name'], csv['Include'])):
        if inc:
            classes[name] = index
    return classes

def processing_images(paths: List[str], classes: dict, cval_add: int = 0) -> List[np.ndarray]:
    images = []
    for p in paths:
        im = io.imread(p)
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
                      cpu_limit: int = None
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
        dataset: Tuple[np.ndarray, np.ndarray],
        batch_size=32,
        as_categorical: bool = True,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.segmentation_mask, self.real_image = dataset
        self.classes = self.segmentation_mask.max() + 1
        self.as_categorical = as_categorical
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        "Denotes the number of batches per epoch"
        return len(self.real_image) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generate one batch of data and returns: (mask, image)"
        # Generate indexes of the batch
        idx = np.s_[index * self.batch_size: (index + 1) * self.batch_size]
        x = self.segmentation_mask[idx]
        y = self.real_image[idx]
        if self.as_categorical:
            x = to_categorical(x, self.classes)
        return x, (y - 127.5) / 127.5

    def on_epoch_end(self):
        if self.shuffle:
            seed = np.random.randint(100000)
            np.random.seed(seed)
            np.random.shuffle(self.segmentation_mask)
            np.random.seed(seed)
            np.random.shuffle(self.real_image)
            np.random.seed(None)

    def get_number_of_classes(self):
        return self.classes



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
            
    if True:
        data = transform_masks(
            classes_path = 'data/included_classes.csv',
            masks_path = 'data/lhq_256/annotations',
            batch_size = 128
        )
        np.save('data/lhq_256/24_classes.npy', data.astype(np.uint8))

    if False:
        data = transform_masks(
            classes_path = 'data/included_classes.csv',
            masks_path = 'data/ADE20K/annotations_256',
            batch_size = 128,
            cval_add = 1
        )
        np.save('data/ADE20K/24_classes.npy', data.astype(np.uint8))
        

    if False:
        images = np.asarray([np.array(io.imread(p), dtype=np.uint8) for p in glob('data/ADE20K/images_256/*.png')])
        np.save('data/ADE20K/images.npy', images)

    if False:
        images = np.asarray([np.array(io.imread(p), dtype=np.uint8) for p in glob('data/lhq_256/images/*.png')])
        np.save('data/lhq_256/images.npy', images)
        

    if False:
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        dataset = load_dataset('data/lhq_256/24_classes.npy', 'data/lhq_256/images.npy')

        images = dataset[1][:10]
        masks128x128 = image_to_mask(images, model, feature_extractor)

        masks = []
        for k in range(len(masks128x128)):
            mask = transform.resize(masks128x128[k], images[k].shape, preserve_range=True).astype(np.uint8)
            masks.append(mask)

        print(parse_csv())
        for i in range(10):
            plt.imshow(dataset[0][i])
            plt.show()
            plt.imshow(masks[i])
            plt.show()
            plt.imshow(dataset[1][i])
            plt.show()
            
