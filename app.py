import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

model = load_model("monkeypox_model.h5")
labels = [ "Monkey Pox","Non Monkey Pox"]

st.title("Monkey Pox Detection")
st.write("Upload an image to see if it contains Monkey Pox")

file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if file is not None:
   
   
   
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    img = cv2.resize(img, (256,256))
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)[0][0]
   
    label = labels[int(round(pred))]
    st.write("Prediction: " + label)



    st.write("Original Image:")
    st.image(img_disp, channels="BGR")
