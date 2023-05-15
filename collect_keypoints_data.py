import os
import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define directories and actions
directory = "Image/"
actions = ["peace", "thumbs up", "stop"]

# Initialize empty lists for keypoints and labels
keypoints = []
labels = []
wait_time = 5
# Loop over actions and images
for action in actions:
    action_dir = os.path.join(directory, action)
    image_files = sorted(os.listdir(action_dir))

    for image_file in image_files:
        image_path = os.path.join(action_dir, image_file)

        # Read and preprocess the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # cv2.imshow("hands", image)

        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)

        # Check if hand landmarks were detected
        if results.multi_hand_landmarks:
            # Extract the keypoints for the first detected hand
            annotated_image = image.copy()
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            cv2.imshow("hands", image)
            
            wait_time = 1

            # Store the keypoints in a list
            landmark_list = np.array([(lm.x, lm.y) for lm in hand_landmarks]).flatten()
            keypoints.append(landmark_list)
            labels.append(action)

        if cv2.waitKey(wait_time) == ord('q'):
            break

# Convert the keypoints and labels to NumPy arrays
keypoints_array = np.array(keypoints)
labels_array = np.array(labels)

# Print the shape of the keypoints and labels arrays
print("Keypoints array shape:", type(keypoints_array))
print("Labels array shape:", type(labels_array))

#save keypoints in a file
try:
        os.makedirs("keypoints_data")
except:
        pass
filename = "keypoints_data/"
np.save(filename+"keypoints_data.npy", keypoints_array)
np.save(filename+"labels.npy", labels_array)

key = np.load(filename+"keypoints_data.npy")
print(key.shape)
labels = np.load(filename+"labels.npy")
print(labels.shape)

# Release resources
hands.close()
