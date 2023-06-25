import tempfile
from pathlib import Path
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from DDGAUGAN import Predictor
from dataset import parse_csv, generate_colors
from skimage import transform, filters
import tensorflow as tf
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from model import resolve_single
from model.edsr import edsr
from model.srgan import generator
from keras import utils
import edit

tf.random.set_seed(0)


# -------------------utility-functions-----------------
def get_swap_dict(d):
    return {v: k for k, v in d.items()}

def image_to_class(im: np.ndarray, classes: dict) -> np.ndarray:
    im = filters.median(im, np.ones((7,7,1)))
    _im = im.reshape((-1, 3))
    data = [np.sum(_im, axis=-1, keepdims=True)]
    for val in classes.values():
        data.append(np.sum(_im - val, axis=-1, keepdims=True))
    data = np.concatenate(data, axis=-1)
    data = np.argmin(data, axis=-1, keepdims=True)
    return data.reshape((*im.shape[:2], 1))

def class_to_image(data, classes):
    data = data.argmax(axis=-1)
    colors = np.array(list(classes.values()))
    image = colors[data-1]
    return image[0]

def get_uploaded_image(file):
    temp_dir_path = Path(tempfile.mkdtemp())
    path = str(temp_dir_path / file.name)
    with open(path, 'wb') as f:
        f.write(file.read())
    image = cv2.imread(path)
    return image[..., ::-1]


def merge_labels(bg_labels, fg_labels):
    if bg_labels is None:
        return fg_labels
    if fg_labels is None:
        return bg_labels
    
    bg_labels = bg_labels.argmax(axis=-1)
    fg_labels = fg_labels.argmax(axis=-1)
    labels = np.where(fg_labels==0, bg_labels, fg_labels)
    labels = utils.to_categorical(labels, 25)
    return labels


# -------------------persisted-variables--------------------
if 'predictor' not in st.session_state:
    st.session_state.predictor = Predictor('trained_models/image_generator_model.h5', 'trained_models/image_encoder_model.h5')
if 'model_sr' not in st.session_state:
    st.session_state.model_sr = generator()
    st.session_state.model_sr.load_weights('trained_models/gan_generator.h5')
if 'nvidia_feature_extractor' not in st.session_state:
    st.session_state.nvidia_feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
if 'nvidia_model' not in st.session_state:
    st.session_state.nvidia_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
if 'noise' not in st.session_state:
    st.session_state.noise = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'added_labels' not in st.session_state:
    st.session_state.added_labels = None
    
    
# -------------------standard-variables--------------------
rgb2hex = lambda r,g,b: '#%02x%02x%02x' %(r,g,b)
classes = parse_csv()
colors = {cl: rgb2hex(*co) for (cl, co) in zip(classes.keys(), generate_colors(len(classes)))}
colors_rgb = {cl: co for (cl, co) in zip(classes.keys(), generate_colors(len(classes)))}
swaped_colors = get_swap_dict(colors)


# ----------------------editing-panel-----------------------
with st.expander('Editing', expanded=True):
    edit_image = st.file_uploader('Image to edit')

    if edit_image:
        edit_image = get_uploaded_image(edit_image)
        st.image(edit_image)
        
        st.session_state.labels = edit.get_labels(
            edit_image,
            st.session_state.nvidia_feature_extractor,
            st.session_state.nvidia_model
        )
        
        training_steps = st.number_input('Training steps', value=100)
        if st.button('Start training'):
            progress = st.progress(0)
            merged_labels = merge_labels(st.session_state.labels, st.session_state.added_labels)
            mask = st.session_state.added_labels.argmax(axis=-1)[..., np.newaxis] > 0
            st.session_state.noise = edit.get_noise(
                predictor=st.session_state.predictor,
                labels=merged_labels,
                target=edit_image,  # ok
                mask=mask,
                train_steps=training_steps,
                initial_noise=st.session_state.noise,
                st_progress=progress
            )
    else:
        st.session_state.labels = None
        st.session_state.noise = None


# ------------------------sidebar-stuff---------------------------
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3)
stroke_color = st.sidebar.selectbox("Stroke class hex: ", options=colors.values(), format_func=lambda x: swaped_colors[x])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
super_resolution = st.sidebar.checkbox("Enable Super-Resolution", False)
tf_seed = st.sidebar.number_input('Seed', value=0, disabled=edit_image is not None)

# --------------------------canvas------------------------------
background_image = Image.fromarray((class_to_image(st.session_state.labels, colors_rgb)*0.8).astype(np.uint8)) \
    if st.session_state.labels is not None else None
canvas_result = st_canvas(
    fill_color="#000",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#000",
    background_image=background_image,
    update_streamlit=realtime_update,
    height=256,
    width=256,
    drawing_mode=drawing_mode,
    point_display_radius= 0,
    key="canvas",
)


# --------------------------generating------------------------------
if canvas_result.image_data is not None:
    
    added_labels = np.array(canvas_result.image_data)[..., :3]
    added_labels = image_to_class(added_labels, colors_rgb)
    added_labels = utils.to_categorical(added_labels, 25)
    added_labels = added_labels[np.newaxis, ...]
    st.session_state.added_labels = added_labels
    
    if edit_image is not None:
        merged_labels = merge_labels(st.session_state.labels, st.session_state.added_labels)
        noise = st.session_state.noise if st.session_state.noise is not None else tf.random.normal((1, 256))
        image = (st.session_state.predictor.gen([noise, merged_labels]) + 1) * 127.5
        image = np.array(image, np.uint8)
        
    else:
        tf.random.set_seed(tf_seed.real)
        image = st.session_state.predictor(st.session_state.added_labels)[0]
        
    if super_resolution:
        image = resolve_single(st.session_state.model_sr, image[0])
        image = filters.median(image)

    st.image(np.array(image))
