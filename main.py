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

files = {
    "Kép 1": os.path.join("img3", "1.jpg"),
    "Kép 2": os.path.join("img3", "2.jpg"),
    "Kép 3": os.path.join("img3", "3.jpg")
}

mosaic_path = os.path.join("img3", "mosaic.jpg")

# Alkalmazás cím
st.set_page_config(page_title="YOLO Object Detection", page_icon="🔥")
st.title("🤖YOLO Object Detection")
with st.expander("📌 Projekt célja"):
    st.markdown(
        """
        Ez a projekt célja, hogy bemutassa a **YOLO Object Detection** modell működését 
        különböző tesztképeken. A modell a Logisztikai Központban lévő automata
        raktárrobotok fejlesztéséhez készült.
        Az automata rendszer összes hibáinak 20%-a a kamerás fiókelőlap keresésnél, 10%-a az
        akasztókeresésnél jelentkezik. Ez a projekt a jelenlegi, pixel számolós algoritmus
        kiváltására készült.

        - Előfeldolgozás: CLAHE (`clipLimit=3.0`, `tileGridSize=(8, 8)`)  
        - Modell változat: **XS** (`confidence=0.78`, `iou=0.2`) 
        - Tesztképek letöltésére és detektálás kipróbálására készült demo oldal.  

        A cél, hogy egy egyszerű, webes felületen lehessen kipróbálni a
        detektálási folyamatot és értékelni a modell pontosságát.
        """
    )
with st.expander("🔍Korábbi algoritmussal történő összehasonlítás"):
    st.markdown(
        """
        A tesztelést ~1200 darab képen végeztem. Ebből a YOLO modell mindösszesen 1 darab
        horizontális koordináta tengelyt nem talált meg. Az eredeti megoldás 26 képen ért el
        bármely irányban 10 mm-nél nagyobb eltérést az OD eredményéhez képest. Minden esetben
        az eredti script tévedett.
        A mozaik kép ezekből a képekből lett véletlenszerűen összeállítva.
        A megjelenített koordináta rendszerek:
        - Zöld: a kép középvonalai (sokszor kitakarja a piros vonal)
        - Kék YOLO modell eredménye
        - Piros: eredeti script eredménye
        """
    )
    st.image(mosaic_path, caption="Összehasonlító mozaik", use_container_width=True)
st.write("Letölthető képek teszteléshez:")
cols = st.columns([1, 1, 1], gap="small")

for col, (label, path) in zip(cols, files.items()):
    with col:
        with open(path, "rb") as f:
            st.download_button(
                label=f"📥 {label}",
                data=f,
                file_name=os.path.basename(path),
                mime="image/png",
                use_container_width=False  # fontos, hogy ne nyújtsa szét
            )

github_url = "https://github.com/mecalis/fiok_streamlit/tree/main/img3"
st.markdown(f"[👉 Nyisd meg az img3 mappát a GitHubon az összes többi képért!]({github_url})")
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

        results = model.predict(source = pre_image, conf = 0.7, iou = 0.85)
        #print("Predikciók")
        #print(results)
        lines = []
        lines.append("Adatok: \n")
        text = ""
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box koordináták
                if y1 > 250:
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
        detected_speed_slot.write(f"Sebességek [ms]: {results[0].speed}. Általában <= ~100 ms. ")

















