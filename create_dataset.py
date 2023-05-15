import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import os

for t in range(1,6):
    path = 'data/'+str(t)+'/'

    images = os.listdir(path)
    for i in images:
        image = cv2.imread(path+i)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.8,min_tracking_confidence=0.8)
        mp_draw = mp.solutions.drawing_utils
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results = hands.process(image)
        image.flags.writeable=True 

        if results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(image = image, landmark_list = hand_landmarks,
                                connections = mp_hands.HAND_CONNECTIONS)
        a = dict()
        a['label'] = t
        for i in range(21):
            s = ('x','y')
            k = (hand_landmarks.landmark[i].x,hand_landmarks.landmark[i].y)
            for j in range(len(k)):
                a[str(mp_hands.HandLandmark(i).name)+'_'+str(s[j])] = k[j]
        df = df.append(a,ignore_index=True)
