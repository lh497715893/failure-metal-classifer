import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('my_model.h5')



st.write("""
         # Fatigue-Tensile Fracture Metal Classifier
         """
         )
st.write("This is a image classification web app to predict types of fracture metal")
st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



def import_and_predict(image_data, model):
        size = (800,450)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(800, 450), interpolation=cv2.INTER_CUBIC))/255.     
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)     
        return prediction



if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    print("np.argmax(prediction):"+str(np.argmax(prediction)))
    #if np.argmax(prediction) == 0:
    print("prediction:"+str(prediction[0]))
    if np.round(prediction[0]) == 0:
        st.write("It is a fatigue fracture metal!")
    #elif np.argmax(prediction) == 1:
    elif np.round(prediction[0]) == 1:
        st.write("It is a tensile fracture metal!")
    else:
        st.write("Wrong!")
        
    st.text("Probability (0: Fatigue Fracture, 1: Tensile Fracture)")
    st.write(prediction)
