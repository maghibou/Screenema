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
      min_dist = 0.5
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
        fontScale  = 2
        fontColor = (255,0,0)
        lineType = 4
        # get actor's name above the boxe
        cv2.putText(img, f'{member}, dist: ({round(min_dist, 2)})', (left, top-10), font, fontScale, fontColor, lineType)
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
      fontScale  = 2
      fontColor = (255,0,0)
      lineType = 4
      # get actor's emotion under the boxe
      obj = DeepFace.analyze(img_path = img, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection =False)
      cv2.putText(img, f"{obj['dominant_emotion']}", (left, bottom+30), font, fontScale, fontColor, lineType)
    return img
    # Show Image    
    # plt.figure(figsize=(16,10))
    # plt.axis('off')
    # plt.imshow(img)


  # convert face distance to similirity likelyhood
  def face_distance_to_conf(face_distance, face_match_threshold=0.6):
      if face_distance > face_match_threshold:
          range = (1.0 - face_match_threshold)
          linear_val = (1.0 - face_distance) / (range * 2.0)
          return linear_val
      else:
          range = face_match_threshold
          linear_val = 1.0 - (face_distance / (range * 2.0))
          return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))


  # convert opencv BRG to regular RGB mode
  def BGR_to_RGB(image_in_array):
      return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)



  # # disable warning signs:
  # # https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
  # st.set_option("deprecation.showfileUploaderEncoding", False)

  # # set page config
  # page_icon =Image.open("/content/drive/MyDrive/Screenema_Streamlit/assets/popcorn.svg")
  # st.set_page_config(
  #   page_title="SCREENEMA"
  #   page_icon=page_icon
  # )

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
    if st.button("recognise"):
      image = display_image_recognition(screenshot)
      st.image(image)
    if st.button("emotion"):
      image = get_emotion(screenshot)
      st.image(image)
    


    


