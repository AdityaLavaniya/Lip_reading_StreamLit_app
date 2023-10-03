import streamlit as st
import os
import imageio
import tensorflow as tf
from util import load_data, num_to_char
from model import load_model

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://img.freepik.com/premium-vector/brain-logo-template_15146-27.jpg?w=740')
    st.title('lipBuddy')
    st.info('lipBuddy is an innovative web application designed to assist you in the fascinating world of lip reading.With the power of cutting-edge machine learning technology, this tool is here to help you understand spoken words by analyzing the movements of the lips.')
 
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('choose video', options)

col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        #Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)   
    

   
    