import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

DATA_DIR = './isl_dataset'

mp_hands = mp.solutions.hands
# static_image_mode=True is accurate but slow; let's keep it robust
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

data = []
labels = []

print("Starting optimized feature extraction...")

folders = os.listdir(DATA_DIR)
for dir_path in folders:
    class_dir = os.path.join(DATA_DIR, dir_path)
    if not os.path.isdir(class_dir):
        continue
        
    print(f"Processing letter folder: {dir_path}...")
    img_list = os.listdir(class_dir)
    
    for img_name in img_list:
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
            
        # FAST OPTIMIZATION: Resize image to speed up MediaPipe processing
        image = cv2.resize(image, (640, 480))
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = []
                for landmark in hand_landmarks.landmark:
                    row.append(landmark.x)
                    row.append(landmark.y)
                
                data.append(row)
                labels.append(dir_path)

columns = []
for i in range(21):
    columns.append(f'x_{i}')
    columns.append(f'y_{i}')

df = pd.DataFrame(data, columns=columns)
df['label'] = labels

cols = ['label'] + [col for col in df.columns if col != 'label']
df = df[cols]

df.to_csv('isl_landmarks.csv', index=False)
print("Feature extraction complete! Saved to 'isl_landmarks.csv'.")