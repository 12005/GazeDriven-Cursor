import cv2 as cv
import numpy as np
import tensorflow as tf
import face_recognition
import time
import pyautogui

# Load the models (1 for open/closed eye detection, 5 for direction detection)
eye_state_model = tf.keras.models.load_model("modelsB/eye_open_closed_cnn_model_0.h5")
direction_models = [
    tf.keras.models.load_model(f"modelsD/eye_gazeD_8directions_cnn_model_{i}.h5")
    for i in range(5)
]

# Define the label dictionary for eye direction
label_dict = {
    0: 'left', 1: 'right', 2: 'up', 3: 'down',
    4: 'top-left', 5: 'top-right', 6: 'bottom-left', 7: 'bottom-right'
}

# Variables to manage toggling and timing
eye_closed_start_time = None
eye_direction_active = False
predictions_list = []
eye_closed_duration = 0
predicted_label = "Detecting"
last_prediction = "None"

# Function to capture eye region using face landmarks
def capture_eye(frame, frameShrink=0.15):
    smallframe = cv.resize(frame, (0, 0), fx=frameShrink, fy=frameShrink)
    smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

    try:
        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
            left_eye = cv.resize(left_eye, (100, 50))
            return left_eye
    except Exception as e:
        print(f"Error capturing eye: {e}")
        return None
    return None

# Function to find min and max of eye region
def maxAndMin(featCoords, mult=1):
    adj = 10 / mult
    listX = [tup[0] for tup in featCoords]
    listY = [tup[1] for tup in featCoords]
    maxminList = np.array([min(listX) - adj, min(listY) - adj, max(listX) + adj, max(listY) + adj])
    return (maxminList * mult).astype(int), (np.array([sum(listX) / len(listX) - maxminList[0], sum(listY) / len(listY) - maxminList[1]]) * mult).astype(int)

# Initialize webcam feed
webcam = cv.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Capture the eye region
    left_eye = capture_eye(frame)
    if left_eye is not None:
        # Preprocess the captured eye for prediction
        eye_input = np.expand_dims(left_eye, axis=-1)
        eye_input = np.expand_dims(eye_input, axis=0)
        eye_input = eye_input / 255.0

        # Detect if the eye is open or closed
        prediction = eye_state_model.predict(eye_input)
        eye_closed = (prediction > 0.5).astype(int)[0][0]  # 1 means closed, 0 means open

        # Handle eye closure detection timing
        if eye_closed == 1:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            eye_closed_duration = time.time() - eye_closed_start_time
            if eye_closed_duration >= 1.0:
                # Toggle eye direction detection
                eye_direction_active = not eye_direction_active
                eye_closed_start_time = None
                eye_closed_duration = 0
        else:
            eye_closed_start_time = None
            eye_closed_duration = 0

        if eye_direction_active:
            # Detect eye direction
            predictions = np.zeros((1, 8))
            for model in direction_models:
                predictions += model.predict(eye_input)

            predictions /= len(direction_models)
            predicted_class = np.argmax(predictions)
            predicted_label = label_dict[predicted_class]

            # Store the prediction and update last prediction
            if predicted_label != last_prediction:
                predictions_list.append(predicted_label)
                last_prediction = predicted_label

            # Move the mouse in the detected direction
            if predicted_label == 'left':
                pyautogui.moveRel(-10, 0, duration=0.1)
            elif predicted_label == 'right':
                pyautogui.moveRel(10, 0, duration=0.1)
            elif predicted_label == 'up':
                pyautogui.moveRel(0, -10, duration=0.1)
            elif predicted_label == 'down':
                pyautogui.moveRel(0, 10, duration=0.1)
            elif predicted_label == 'top-left':
                pyautogui.moveRel(-10, -10, duration=0.1)
            elif predicted_label == 'top-right':
                pyautogui.moveRel(10, -10, duration=0.1)
            elif predicted_label == 'bottom-left':
                pyautogui.moveRel(-10, 10, duration=0.1)
            elif predicted_label == 'bottom-right':
                pyautogui.moveRel(10, 10, duration=0.1)
        else:
            predicted_label = "Inactive"

        # Display the frame with the detected state
        font = cv.FONT_HERSHEY_SIMPLEX
        mode_text = "Eye Direction Detection: " + ("Active" if eye_direction_active else "Inactive")
        current_prediction_text = f"Current Prediction: {predicted_label}"
        last_prediction_text = f"Last Prediction: {last_prediction}"

        cv.putText(frame, mode_text, (50, 50), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, current_prediction_text, (50, 100), font, 0.7, (255, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, last_prediction_text, (50, 150), font, 0.7, (255, 0, 0), 2, cv.LINE_AA)

        # Display eye closed duration
        closed_duration_text = f"Eye Closed Duration: {int(eye_closed_duration)}s"
        cv.putText(frame, closed_duration_text, (50, 200), font, 0.7, (255, 0, 255), 2, cv.LINE_AA)

    # Show the video feed
    cv.imshow("Webcam - Eye Control", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv.destroyAllWindows()