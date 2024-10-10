import face_recognition
import numpy as np
import cv2 as cv
import copy
import os

# Utility function for getting the bounding box and center of a facial feature
def maxAndMin(featCoords, mult=1):
    adj = 10 / mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX) - adj, min(listY) - adj, max(listX) + adj, max(listY) + adj])
    return (maxminList * mult).astype(int), (np.array([sum(listX) / len(listX) - maxminList[0], sum(listY) / len(listY) - maxminList[1]]) * mult).astype(int)

# Function to create new folder for each run
def createNewFolder(base_folder="runs"):
    os.makedirs(base_folder, exist_ok=True)  # Ensure base folder exists
    existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    folder_numbers = [int(f) for f in existing_folders if f.isdigit()]  # Find numeric folders
    next_folder_number = max(folder_numbers) + 1 if folder_numbers else 0  # Increment to the next folder number
    new_folder_path = os.path.join(base_folder, str(next_folder_number))
    os.makedirs(new_folder_path, exist_ok=True)  # Create new folder
    return new_folder_path

# Function to capture eye images and save them
def getEyeAndHead(direction, times=20, frameShrink=0.15, folder="runs"):
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return None

    counter = 0

    while counter < times:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        try:
            feats = face_recognition.face_landmarks(smallframe)
            if len(feats) > 0:
                # Get left eye
                leBds, _ = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
                left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
                left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
                left_eye = cv.resize(left_eye, dsize=(100, 50))

                # Save the eye image with the direction in the filename
                img_path = os.path.join(folder, f"{direction}_{counter}.jpg")
                cv.imwrite(img_path, left_eye)

                # Print success message
                print(f"Image saved successfully at {img_path}")

                counter += 1
        except Exception as e:
            print(f"Error: {e}")

    webcam.release()
    cv.destroyAllWindows()

# Main Execution Flow

# Create a new folder for this execution
run_folder = createNewFolder(base_folder="runsD")

# Directions for capturing eye movements in 8 directions
directions = ['left', 'right', 'up', 'down', 'top-left', 'top-right', 'bottom-left', 'bottom-right']

# Capture 20 images for each eye movement direction
for direction in directions:
    input(f"Please look {direction}. Press Enter when ready...")
    getEyeAndHead(direction=direction, times=20, folder=run_folder)
