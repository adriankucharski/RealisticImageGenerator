import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from DDGAUGAN import Predictor
from dataset import parse_csv, generate_colors
from skimage import transform, filters
import tensorflow as tf
from model import resolve_single
from model.edsr import edsr
from model.srgan import generator

tf.random.set_seed(0)

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

if 'predictor' not in st.session_state:
    st.session_state.predictor = Predictor(r'logs\20230504-2303\generators\model_36.h5')
if 'model_sr' not in st.session_state:
    st.session_state.model_sr = generator()
    st.session_state.model_sr.load_weights('srgan_model/gan_generator.h5')

rgb2hex = lambda r,g,b: '#%02x%02x%02x' %(r,g,b)
classes = parse_csv()
colors = {cl: rgb2hex(*co) for (cl, co) in zip(classes.keys(), generate_colors(len(classes)))}
colors_rgb = {cl: co for (cl, co) in zip(classes.keys(), generate_colors(len(classes)))}
swaped_colors = get_swap_dict(colors)
# print(colors)

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3)
stroke_color = st.sidebar.selectbox("Stroke class hex: ", options=colors.values(), format_func=lambda x: swaped_colors[x])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
super_resolution = st.sidebar.checkbox("Enable Super-Resolution", False)
tf_seed = st.sidebar.number_input('Seed', value=0)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#000",
    background_image=None,
    update_streamlit=realtime_update,
    height=256,
    width=256,
    drawing_mode=drawing_mode,
    point_display_radius= 0,
    key="canvas",
)

if canvas_result.image_data is not None:
    tf.random.set_seed(tf_seed.real)
    im = np.array(canvas_result.image_data)[..., :3]
    im = st.session_state.predictor(image_to_class(im, colors_rgb))
    if super_resolution:
        im = resolve_single(st.session_state.model_sr, im)
        im = filters.median(im)
    st.image(np.array(im))
    