import itertools
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from skimage import io
import numpy as np
from glob import glob
from typing import List
from tqdm import tqdm
import multiprocessing

import numpy as np
import cv2
from tqdm import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter


def mask_preprocess(im: np.ndarray, ksize: int = 13, max_f_size: int = 5, min_size: int = 1000) -> np.ndarray:
    fmaps = cv2.medianBlur(im, ksize)[..., np.newaxis]
    unique, counts = np.unique(fmaps, return_counts=True)
    for u, c in zip(unique, counts):
        if c < min_size:
            fmaps[fmaps == u] = 0
    fmaps = maximum_filter(fmaps, max_f_size)
    return fmaps


def image_to_mask(images, model, feature_extractor) -> List[np.ndarray]:
    inputs = feature_extractor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    data: np.ndarray = np.array(
        logits.detach().numpy()).transpose((0, 2, 3, 1))
    image_mask = np.argmax(data, axis=-1, keepdims=True)
    return image_mask.astype(np.uint8)


def transform_images_to_seg(paths, model, feature_extractor):
    images = [io.imread(p) for p in paths]
    masks = image_to_mask(images, model, feature_extractor)
    for k in range(len(masks)):
        name = paths[k].replace('images', 'annotations_128')
        io.imsave(name, masks[k], check_contrast=False)


if __name__ == '__main__':
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512")

    files = list(glob('./data/lhq_256/images/*.png'))
    batch_size = 16
    files_batched = [files[i:i+batch_size]
                     for i in range(0, len(files), batch_size)]

    with multiprocessing.Pool(6) as pool:
        args = zip(
            files_batched,
            itertools.repeat(model),
            itertools.repeat(feature_extractor)
        )
        pool.starmap(transform_images_to_seg, tqdm(
            args, total=len(files_batched)))
