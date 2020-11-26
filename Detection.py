# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# import cv2
# import matplotlib.pyplot as plt
# import dlib
# from imutils import face_utils
# font = cv2.FONT_HERSHEY_SIMPLEX


# # In[2]:


# cascPath = "/Users/abdulrehman/opt/anaconda3/envs/Face-Detection/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"
# eyePath = "/Users/abdulrehman/opt/anaconda3/envs/Face-Detection/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml"
# smilePath = "/Users/abdulrehman/opt/anaconda3/envs/Face-Detection/lib/python3.6/site-packages/cv2/data/haarcascade_smile.xml"
# data_folder = '/Users/abdulrehman/Desktop/SML Project/Faces'


# # In[3]:


# # gray = cv2.imread(data_folder+'/images/test/10.jpg', 0)


# # # In[4]:


# # plt.figure(figsize=(12,8))
# # plt.imshow(gray, cmap='gray')
# # plt.show()


# # In[5]:


# faceCascade = cv2.CascadeClassifier(cascPath)
# eyeCascade = cv2.CascadeClassifier(eyePath)
# smileCascade = cv2.CascadeClassifier(smilePath)


# # In[6]:


# # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
# # for (x, y, w, h) in faces:
# #     cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)


# # # In[7]:


# # plt.figure(figsize=(12,8))
# # plt.imshow(gray, cmap='gray')
# # plt.show()


# # In[ ]:


# video_capture = cv2.VideoCapture(0)
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
#         minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
#     for (x, y, w, h) in faces:
#         if w > 250 :
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]
        
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()


# # In[ ]:




import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    flags, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', gray)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows() 



