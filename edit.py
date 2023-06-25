import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import tensorflow as tf
from keras import utils
from keras.losses import MeanSquaredError
from tqdm import tqdm

from dataset import parse_csv, image_to_mask
    
    
def get_labels(image, nvidia_feature_extractor, nvidia_model):
    labels = image_to_mask(image, nvidia_model, nvidia_feature_extractor)
    labels = np.repeat(np.repeat(labels, 2, axis=1), 2, axis=2)
    labels = _filter_with_median_blur(labels[0])
    labels = _transform_labels(labels)
    labels = utils.to_categorical(labels, 25)
    labels = labels[np.newaxis, ...].astype(np.float32)
    return labels


def get_noise(predictor, labels, target, mask, train_steps, 
              initial_noise, st_progress):
    target = cv2.resize(target, dsize=(256, 256))
    
    if initial_noise is None:
        initial_noise = tf.random.normal(shape=(1, 256))
    noise = tf.Variable(initial_value=initial_noise, trainable=True)
    
    generator = predictor.gen
    generator.trainable = False
    
    target = np.array(target) / 127.5 - 1
    target = target[np.newaxis, ...]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    iterator = tqdm(range(train_steps))
    for step in iterator:
        with tf.GradientTape() as tape:
            prediction = generator([noise, labels])
            loss_value = tf.math.reduce_sum((prediction - target)**2 * mask)
            
        gradients = tape.gradient(loss_value, [noise])
        optimizer.apply_gradients(zip(gradients, [noise]))
        
        iterator.set_postfix_str(f'Loss: {loss_value.numpy()}')
        
        if st_progress:
            st_progress.progress((step+1) / train_steps, text='Training')
        
    return noise.numpy()


def _filter_with_median_blur(map):
    fmaps = cv2.medianBlur(map, 13)[..., np.newaxis]
    unique, counts = np.unique(fmaps, return_counts=True)
    pmax_val = unique[np.argmax(counts)]
    for u, c in zip(unique, counts):
        if c < 1000:
            fmaps[fmaps == u] = 0
    fmaps = maximum_filter(fmaps, 5)
    return fmaps


def _transform_labels(map):
    classes = parse_csv()
    map = np.array(map)
    if len(map.shape) == 3:
        map = map[..., 0:1]
    else:
        map = map[..., np.newaxis]
    x = np.zeros_like(map, dtype=np.uint8)
    for i, val in enumerate(classes.values()):
        x[map == val + 0] = i + 1
    return x
