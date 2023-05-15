import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
actions = ["peace", "thumbs up", "stop"]
cap = cv2.VideoCapture(0)
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Unable to open the webcam")
    exit()

model = tf.keras.models.load_model(model_save_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def draw_bounding_box(image, hand_landmarks):
    # Get the image dimensions
    height, width, _ = image.shape

    # Get the x, y coordinates of the hand landmarks
    x_list = [landmark.x * width for landmark in hand_landmarks]
    y_list = [landmark.y * height for landmark in hand_landmarks]

    # Calculate the bounding box coordinates
    x_min = int(min(x_list))
    y_min = int(min(y_list))
    x_max = int(max(x_list))
    y_max = int(max(y_list))

    # Draw the bounding box rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Return the image with the bounding box drawn
    return image

# Keep reading frames from the webcam until the user quits the program
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Unable to read a frame from the webcam")
        break
    x , y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB color space
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)


    
    # Post-process the result if hand landmarks are detected
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmark in result.multi_hand_landmarks:
            draw_bounding_box(frame, hand_landmark.landmark)

            #extract keypoints data as a numpy array
            landmark_list = np.array([(lm.x, lm.y) for lm in hand_landmark.landmark]).flatten()
            landmark_list = np.expand_dims(landmark_list, axis=0)
            
            #predict hand gestures
            predict = model.predict(landmark_list)
            print(np.squeeze(predict))
            action = actions[np.argmax(np.squeeze(predict))]
            color = (0, 255, 0)  # White color
            thickness = 2
            print(actions[np.argmax(np.squeeze(predict))])

            cv2.putText(frame, action, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
            # Draw the landmarks on the frame
            mpDraw.draw_landmarks(frame, hand_landmark, mpHands.HAND_CONNECTIONS)

    # Display the frame in a window named "Webcam"
    cv2.imshow("Webcam", frame)

    # Wait for 1 millisecond for the user to press any key
    # If the user presses the "q" key, break the loop and quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()

