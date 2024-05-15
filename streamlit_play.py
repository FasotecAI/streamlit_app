# -*- coding: utf-8 -*-

import streamlit as st
import time
import datetime
import os
import glob
import cv2
from PIL import Image
from datetime import datetime, date, time

PATH='C:/Users/Jeong/Desktop/2024_AD_AI_program/video_images'

def disp(device):
    cap = cv2.VideoCapture(device)
    image_loc = st.empty()
    while cap.isOpened:
        ret, img = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image_loc.image(img)
        else:
            break

    cap.release()
    st.button('Replay')

def main():
    st.header("AD/AIデモビデオ")
    #date=st.date_input('Select date')
    #path=PATH+date.strftime("%Y%m%d")
    path =PATH+"/robot"
    #st.write(path)
    if os.path.exists(path):
        files=glob.glob(path+'/*mp4')
        option = st.selectbox('Select file:',files)
        disp(option)
    else:
        st.write('No data exists!')

if __name__ == '__main__':
    main()