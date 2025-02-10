import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Loading Trained Model
model = tf.keras.models.load_model("resnet_fish_classification.keras")

#Defining Class Labels 
class_labels = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 
                'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 
                'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']

#Streamlit App
st.title("Fish Classifier")
st.write("Upload an image of a fish, and the model will predict its category.")

#Uploading Image
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    #Loading and Displaying Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True) 

    #Preprocessing Image
    img = image.resize((224, 224))  #Resizing to match model input shape
    img_array = np.array(img) / 255.0  #Normalizing
    img_array = np.expand_dims(img_array, axis=0)  #Adding batch dimension

    #Making Prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence_score = np.max(predictions) * 100

    #Displaying Prediction
    st.success(f"Predicted Class: **{predicted_class}**")
    st.info(f"Confidence Score: **{confidence_score:.2f}%**")
