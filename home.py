# IMPORTS 
#images
import glob
from PIL import Image
import cv2
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from retinaface import RetinaFace
#paths and dataframes / arrays
import os
import pandas as pd
import numpy as np
# Mood and age recognition
from deepface import DeepFace
#pickle
import pickle
#Streamlit
import streamlit as st

def render():
  #page 
  distance_slider = st.sidebar.slider("distance threshold", min_value=0.0, max_value=1.0, value=0.6)
  with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

  # Import Database pickle 
  path_others ='data/other_member.pkl'
  path_same_person = "data/same_member.pkl"
  path_embeddeed_pics = "data/embedded_img.pkl"

  with open(path_same_person, "rb") as f:
      same_member_pickle = pickle.load(f)
  with open(path_others, "rb") as f:
      other_member_pickle = pickle.load(f)
  with open(path_embeddeed_pics, "rb") as f:
      embedded_pics_pickle = pickle.load(f)

  # FONCTIONS  

  #Preprocess to get img actors list, boxes and min dist
  def preprocess_image(image_path):
    # Get the image
    img = plt.imread(image_path)
    # Encode the faces
    faces_to_find = face_recognition.face_encodings(img)
    # Get the faces boxes for viz
    boxes = face_recognition.face_locations(img)#, model = "cnn")
    actors_list = []
    dist_list = []
    # Loop over each encoded face
    for test_encoding in faces_to_find:
      min_dist = distance_slider
      min_key = "unknown"
      for key in embedded_pics_pickle:
          for encoding in embedded_pics_pickle[key]:
              dist = np.linalg.norm(encoding-test_encoding)
              if dist < min_dist: #erreur dans la correction ici ^^
                  min_dist = dist
                  min_key = key
      actors_list.append(min_key)
      dist_list.append(min_dist)
    return boxes, actors_list, dist_list, img 

  # convert image from opened file to encoded (embedded) image and face recognition
  def display_image_recognition(image_path):
    boxes, actors_list, dist_list, img = preprocess_image(image_path)
    for (top, right, bottom, left), member, min_dist  in zip(boxes, actors_list, dist_list):
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale  = 0.75
        fontColor = (255,0,0)
        lineType = 2
        # get actor's name above the boxe
        cv2.putText(img, f'{member}', (left, top-10), font, fontScale, fontColor, lineType)
    return img
    # Show Image    
    # plt.figure(figsize=(16,10))
    # plt.axis('off')
    # plt.imshow(img)

  def get_emotion(image_path):
    boxes, actors_list, _, img = preprocess_image(image_path)
    for (top, right, bottom, left), member  in zip(boxes, actors_list):
      cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
      font = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = (10,500)
      fontScale  = 0.75
      fontColor = (255,0,0)
      lineType = 2
      # get actor's emotion under the boxe
      obj = DeepFace.analyze(img_path = img, actions = ['emotion'], enforce_detection =False) #'age', 'gender', 'race', 
      cv2.putText(img, f"{obj['dominant_emotion']}", (left, bottom+30), font, fontScale, fontColor, lineType)
    return img
    # Show Image    
    # plt.figure(figsize=(16,10))
    # plt.axis('off')
    # plt.imshow(img)

  def get_age(image_path):
    boxes, actors_list, _, img = preprocess_image(image_path)
    for (top, right, bottom, left), member  in zip(boxes, actors_list):
      cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
      font = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = (10,500)
      fontScale  = 0.75
      fontColor = (255,0,0)
      lineType = 2
      # get actor's emotion under the boxe
      obj = DeepFace.analyze(img_path = img, actions = ['age'], enforce_detection =False) # 'gender', 'race', 'emotion'
      cv2.putText(img, f"{obj['dominant_age']}", (left, bottom+30), font, fontScale, fontColor, lineType)
    return img
    # Show Image    
    # plt.figure(figsize=(16,10))
    # plt.axis('off')
    # plt.imshow(img)

  


  def preprocess_df_filter(image_path):
    # Get the image
    img = plt.imread(image_path)
    # Encode the faces
    faces_to_find = face_recognition.face_encodings(img)
    # Get the faces boxes for viz
    boxes = face_recognition.face_locations(img)#, model = "cnn")
    actors_list = []
    dist_list = []
    # Loop over each encoded face
    for test_encoding in faces_to_find:
      min_dist = distance_slider
      min_key = "unknown"
      for key in embedded_pics_pickle:
          for encoding in embedded_pics_pickle[key]:
              dist = np.linalg.norm(encoding-test_encoding)
              if dist < min_dist: #erreur dans la correction ici ^^
                  min_dist = dist
                  min_key = key
      actors_list.append(min_key)
      dist_list.append(min_dist)
    return actors_list, dist_list


  # title area
  st.markdown("""
  # SCREENEMA
  > Who the *** is this actor playing this movie? Scan/browse it to find it out !! 
  """)

  # displays a file uploader widget and return to BytesIO
  screenshot = st.file_uploader(
      label="Select a screenshot from the movie:", type=['jpg']
  )


  if screenshot:
    col1, col2, col3 = st.columns([6,1,6])
    with col1:
      if st.button("recognise"):
        image = display_image_recognition(screenshot)
        st.image(image)
        # display df with recognised actors preprocess_image
        actors_filter = preprocess_df_filter(screenshot)
        df = pd.read_csv('data/TBBT_actors.csv', sep=',')
        mask = df['actor'].isin(actors_filter[0])
        df_filtered = df[mask].drop(columns = "Unnamed: 0")
        df_filtered["distance"] = df_filtered["actor"].apply(lambda x: actors_filter[1][actors_filter[0].index(x)])
        df_filtered = df_filtered.to_html(escape=False)
        st.write(df_filtered, unsafe_allow_html=True)
    with col2:
      ticket = Image.open('assets/LOGO1.png')
      st.image(ticket)
    with col3:
      if st.button("emotion"):
        image = get_emotion(screenshot)
        st.image(image)

      

      
    



    


