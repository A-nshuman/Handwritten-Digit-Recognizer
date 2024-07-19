import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from annotated_text import annotated_text

model = tf.keras.models.load_model('handwritten_best.model.keras')

st.markdown("# Handwritten Number Recognizer")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
	# Read image as bytes
	file_bytes = uploaded_file.read()
	
	# Converting file_data to np array
	nparr = np.frombuffer(file_bytes, np.uint8)
	
	# Decoding image
	img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
	
	# Resize image to 28x28
	img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
	
	# Invert image colors
	img_inverted = np.invert(img_resized)
	
	# Normalize the image
	img_normalized = img_inverted / 255.0
	img_batch = np.expand_dims(img_normalized, axis=0)
	img_final = np.expand_dims(img_batch, axis=-1)
	
	prediction = model.predict(img_final)

	st.markdown(f"## The number in the image is :blue-background[{np.argmax(prediction)}]")