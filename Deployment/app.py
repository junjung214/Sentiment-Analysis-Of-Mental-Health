import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.header('GC 7 Model Deployment')
st.write("""
This website was created to find out about your mental health with the results of modeling using NLP with the LSTM method based on dataset processing from Kaggle.
""")
st.write('Please enter text that describes your feelings.')

data = st.text_input('Must Be Containt English Language', value= '')

data_list = data.split(',') if data else []

#convert dari tuple menajadi list
st.write(f'Text data : \n {data_list}')

#load model
model = tf.keras.models.load_model('lstm_bidirectional.tf')

if st.button('predict'):

    prediciton = model.predict(data_list)
    predicted_classes = np.argmax(prediciton, axis=1)
    # Map predicted classes to labels
    target_names = ['Normal', 'Depression', 'Suicidal', 'Anxiety', 'Bipolar', 'Stress', 'Personality disorder']
    predicted_labels = [target_names[i] for i in predicted_classes]
    st.write('Result Model is :')
    st.write(predicted_labels)