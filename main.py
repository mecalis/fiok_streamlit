import cv2
import torch
torch.classes.__path__ = []

from ultralytics import YOLO
import base64
import os

import numpy as np


import requests
import streamlit as st
from PIL import Image
#from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates

def clahe(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) alkalmazása.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

# Load a model
model = YOLO("model.pt")

# Alkalmazás cím
st.set_page_config(page_title="YOLO Object Detection", page_icon="🔥")
st.title("YOLO Object Detection")
st.write("A modell most a CLAHE (Contrast Limited Adaptive Histogram Equalization) előfeldolgozást használja.")
st.write("A modell jelenleg az 'S' méretű, azaz kicsi modellre lett betanítva. Gyors, de van tőle pontosabb.")
# Feltöltött kép tárolása
uploaded_image = st.file_uploader("Tölts fel egy képet", type=["jpg", "png", "jpeg"])

def clear_images():
    if "image" in st.session_state:
        del st.session_state["image"]
    if "detected_image" in st.session_state:
        del st.session_state["detected_image"]
    uploaded_image_slot.empty()
    detected_image_slot.empty()
    detected_data_slot.empty()
    detected_speed_slot.empty()

st.button("Clear", on_click=clear_images)

# Helyek létrehozása képek megjelenítéséhez
uploaded_image_slot = st.empty()
detected_image_slot = st.empty()
detected_data_slot = st.empty()
detected_speed_slot = st.empty()

if uploaded_image is not None:
    # Eredeti kép megjelenítése
    image = Image.open(uploaded_image)
    uploaded_image_slot.image(image, caption="Feltöltött kép", use_container_width=True)

    # Kép mentése az állapotba
    st.session_state["image"] = image

    if st.button("Detect"):
        # YOLO detekció futtatása
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        clahe_img = clahe(image_bgr)
        pre_image = Image.fromarray(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))

        results = model.predict(source = pre_image, conf = 0.8, iou = 0.8)
        print("Predikciók")
        #print(results)
        lines = []
        lines.append("Adatok: \n")
        text = ""
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box koordináták
                if y1 > 200:
                    continue
                conf = box.conf[0].item()  # Konfidencia érték
                cls = int(box.cls[0])  # Osztály index
                label = result.names[cls]  # Osztály neve
                print(label, cls, "conf:", conf, "x1, y1, x2, y2", x1, y1, x2, y2)
                lines.append(f"Label: {label}, cls: {cls}, conf:{conf:.2f}, x1, y1, x2, y2:{x1}, {y1}, {x2}, {y2} \n")
                # Téglalap rajzolása
                cv2.rectangle(clahe_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Felirat szövege
                text = f"{label} {conf:.2f}"

                # Szöveg mérete és háttér
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(clahe_img, (x1, y1 - h + 25), (x1 + w, y1+25), (0, 255, 0), -1)
                cv2.putText(clahe_img, text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        text = "\n".join(lines)

        result_image = results[0].orig_img
        detected_image = Image.fromarray(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))

        #detected_image_path = os.path.join("output", uploaded_image.name)
        #detected_image = Image.open(detected_image_path)

        # Detektált kép megjelenítése
        st.session_state["detected_image"] = detected_image
        detected_image_slot.image(detected_image, caption="Detektált kép", use_container_width =True)

        #Adatok kiírása
        #st.write("Adatok a detekció után:")

        detected_data_slot.write(f"{text}")
        detected_speed_slot.write(f"Sebességek [ms]: {results[0].speed}. Megjegyzés: online szerveren kb ~1 sec, local szerveren ~300 ms. ")



