# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Neuro Disease Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Neurodegenerative Disease Detector using YOLOv8")

# Sidebar
st.sidebar.header("YOLOv8 Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Pre-Trained', 'Exclusive'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100


if model_type == 'Pre-Trained':
    model_path = Path(settings.PT_MODEL)
elif model_type == 'Exclusive':
    model_path = Path(settings.EX_MODEL)
# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
# Selecting Detection Model
pretrained_model_path = Path(settings.PT_MODEL)
exclusive_model_path = Path(settings.EX_MODEL)

# Load Models
try:
    pretrained_model = helper.load_model(pretrained_model_path)
    exclusive_model = helper.load_model(exclusive_model_path)
except Exception as ex:
    st.error(f"Unable to load models. Check the specified paths.")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):

                res = model.predict(uploaded_image,
                                               conf=confidence)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)

    # Compare Models Button
    st.sidebar.markdown("---")
    if st.sidebar.button('Compare Models'):
        st.subheader("Model Comparison Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Pre-Trained Model Output")
            res_pretrained = pretrained_model.predict(uploaded_image, conf=confidence)
            res_pretrained_img = res_pretrained[0].plot()[:, :, ::-1]
            st.image(res_pretrained_img, caption="Pre-Trained Model Detection", use_column_width=True)

        with col2:
            st.markdown("### Exclusive Model Output")
            res_exclusive = exclusive_model.predict(uploaded_image, conf=confidence)
            res_exclusive_img = res_exclusive[0].plot()[:, :, ::-1]
            st.image(res_exclusive_img, caption="Exclusive Model Detection", use_column_width=True)

        st.markdown("### 🧐 Observation:")
        st.write("The **Exclusive Model** is designed to offer enhanced precision and improved detection capabilities, minimizing false positives and ensuring better diagnosis.")

