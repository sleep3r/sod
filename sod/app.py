import cv2
import streamlit as st

# ML
import torch
import torch.nn.functional as F

# SOD
from config import get_config
from load import load_model
from model import model_info
from transform import get_test_augmentation
from utils.image import imread


def predict(img, model, batch_t):
    with torch.no_grad():
        outputs, edge_mask, ds_map = model(batch_t)

    h, w = img.shape[:2]
    output = (
        F.interpolate(outputs[0].unsqueeze(0), size=(h, w), mode="bilinear")[0][0]
        .cpu()
        .numpy()
    )
    return output


def app():
    st.title("REMOVE BACKGROUND")
    st.sidebar.title("Model scale selection")
    arch = st.sidebar.selectbox(
        "Choose model architecture",
        ("0", "1", "2", "3", "4", "5", "6", "7"),
        index=5,
    )
    st.sidebar.table(model_info)

    cfg = get_config(int(arch))
    transform = get_test_augmentation(cfg.img_size)

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        with st.spinner("Loading model..."):
            model = load_model(cfg, device="cpu")

        img_bytes = uploaded_file.read()
        st.image(img_bytes, caption="Uploaded image")

        with st.spinner("Predicting..."):
            img = imread(img_bytes)
            img_t = transform(image=img)["image"]
            batch_t = torch.unsqueeze(img_t, 0)

            output = predict(img, model, batch_t)

            rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = output * 255
            st.image(rgba, caption="Removed background image")

    with st.expander("Based on the current SoTA paper:"):
        st.markdown(
            """
            ### TRACER: Extreme Attention Guided Salient Object Tracing Network
    
            This paper was accepted at AAAI 2022 SA poster session. [[pdf]](https://arxiv.org/abs/2112.07380)    
            
            [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-duts-te)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te?p=tracer-extreme-attention-guided-salient)  
            [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-dut-omron)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron?p=tracer-extreme-attention-guided-salient)  
            [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-hku-is)](https://paperswithcode.com/sota/salient-object-detection-on-hku-is?p=tracer-extreme-attention-guided-salient)  
            [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-ecssd)](https://paperswithcode.com/sota/salient-object-detection-on-ecssd?p=tracer-extreme-attention-guided-salient)  
            [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-pascal-s)](https://paperswithcode.com/sota/salient-object-detection-on-pascal-s?p=tracer-extreme-attention-guided-salient)                 
            """
        )
        st.image(
            "https://github.com/Karel911/TRACER/raw/main/img/Poster.png",
            caption="Poster",
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Background removal app")  # noqa
    app()
