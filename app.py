import cv2
import streamlit as st
import torch
import torch.nn.functional as F

from config import get_config
from load import load_model
from model import model_info
from transform import get_test_augmentation
from utils.image import imread


def predict(img, model, batch_t):
    with torch.no_grad():
        outputs, edge_mask, ds_map = model(batch_t)

    h, w = img.shape[:2]
    output = F.interpolate(outputs[0].unsqueeze(0), size=(h, w), mode='bilinear')[0][0].cpu().numpy()
    return output


def app():
    st.title('REMOVE BACKGROUND')
    st.sidebar.title('Model scale selection')
    arch = st.sidebar.selectbox(
        'Choose model architecture',
        ('0', '1', '2', '3', '4', '5', '6', '7')
    )
    st.sidebar.table(model_info)

    cfg = get_config(int(arch))

    model = load_model(cfg, device="cpu")
    transform = get_test_augmentation(cfg.img_size)

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        st.image(img_bytes, caption='Uploaded image')

        with st.spinner('Predicting...'):
            img = imread(img_bytes)
            img_t = transform(image=img)["image"]
            batch_t = torch.unsqueeze(img_t, 0)

            output = predict(img, model, batch_t)

            rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = output * 255
            st.image(rgba, caption='Removed background image')


if __name__ == "__main__":
    st.set_page_config(page_title="Background removal app")  # noqa
    app()
