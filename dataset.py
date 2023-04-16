import multiprocessing
import itertools
from typing import List
import numpy as np
import pandas as pd
from glob import glob
from skimage import io
from tqdm import tqdm
import os


def parse_csv(path: str = 'data/ADE20K/included_classes.csv', sep: str = ';'):
    csv = pd.read_csv(path, sep=sep)
    classes = {}
    for index, (name, inc) in enumerate(zip(csv['Name'], csv['Include'])):
        if inc:
            classes[name] = index
    return classes


def processing_images(paths: List[str], classes: dict) -> List[np.ndarray]:
    images = []
    for p in paths:
        im = io.imread(p)[..., 0:1]
        x = np.zeros_like(im)
        for i, val in enumerate(classes.values()):
            x[im == val] = i + 1
        images.append(x)
    return images


def transform_dataset(classes_path: str = 'data/lhq_256/included_classes.csv',
                      dataset_path: str = 'data/lhq_256/annotations/',
                      batch_size: int = 64,
                      cpu_limit: int = None
                      ) -> np.ndarray:

    classes = parse_csv(classes_path)
    files = glob(os.path.join(dataset_path, '*.png'))
    files_batched = [files[i:i+batch_size]
                     for i in range(0, len(files), batch_size)]

    args = zip(
        files_batched,
        itertools.repeat(classes),
    )
    if cpu_limit is None:
        cpu_limit = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_limit) as pool:
        data = pool.starmap(processing_images, tqdm(
            args, total=len(files_batched)))
        return np.concatenate(data, axis=0)


if __name__ == '__main__':
    data = transform_dataset(
        classes_path='data/lhq_256/included_classes.csv',
        dataset_path='data/lhq_256/annotations/',
        batch_size=64
    )
    np.save('data/lhq_256/24_classes.npy', data)
