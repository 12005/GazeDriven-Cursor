import cv2 as cv
import numpy as np
import tensorflow as tf
import face_recognition

# Load the ensemble models (assuming you have 3 models saved for open/closed eyes)
models = [
    tf.keras.models.load_model("modelsB/eye_open_closed_cnn_model_0.h5")
]

# Define the label dictionary for eye state (open/closed)
label_dict = {
    0: 'open',
    1: 'closed'
}

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
        eye_input = np.expand_dims(left_eye, axis=-1)  # Add channel dimension
        eye_input = np.expand_dims(eye_input, axis=0)  # Add batch dimension
        eye_input = eye_input / 255.0  # Normalize to [0, 1]

        # Make predictions using all models and average the results
        predictions = np.zeros((1, 1))  # Initialize an array for the predictions (binary classification)
        for model in models:
            predictions += model.predict(eye_input)

        # Average the predictions across all models
        predictions /= len(models)

        # Get the predicted eye state (open/closed)
        predicted_class = (predictions > 0.5).astype(int)[0][0]
        predicted_label = label_dict[predicted_class]

        # Overlay the predicted eye state as text on the frame
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, f"Eye State: {predicted_label}", (50, 50), font, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Display the frame with the predicted eye state
    cv.imshow("Webcam - Eye State", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv.destroyAllWindows()
