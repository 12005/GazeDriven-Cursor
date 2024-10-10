import face_recognition
import numpy as np
import cv2 as cv
import copy
import pyautogui
import time
import os

def maxAndMin(featCoords, mult=1):
    adj = 10 / mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX) - adj, min(listY) - adj, max(listX) + adj, max(listY) + adj])
    return (maxminList * mult).astype(int), (np.array([sum(listX) / len(listX) - maxminList[0], sum(listY) / len(listY) - maxminList[1]]) * mult).astype(int)

def createNewFolder(base_folder="runs"):
    os.makedirs(base_folder, exist_ok=True)  # Ensure base folder exists
    existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    folder_numbers = [int(f) for f in existing_folders if f.isdigit()]  # Find numeric folders
    next_folder_number = max(folder_numbers) + 1 if folder_numbers else 0  # Increment to the next folder number
    new_folder_path = os.path.join(base_folder, str(next_folder_number))
    os.makedirs(new_folder_path, exist_ok=True)  # Create new folder
    return new_folder_path

# Inside getEyeAndHead function, modify the head position file naming
def getEyeAndHead(times=1, frameShrink=0.15, coords=(0, 0), counterStart=0, folder="runs"):
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return None

    counter = counterStart

    while counter < counterStart + times:
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
                leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
                left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
                left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
                left_eye = cv.resize(left_eye, dsize=(100, 50))

                # Capture head position (using nose bridge center for simplicity)
                nose_bridge = feats[0]['nose_bridge']
                head_position = np.mean(nose_bridge, axis=0) * (1/frameShrink)  # Use nose bridge as a representative of head position

                # Save the eye image
                img_path = os.path.join(folder, f"{coords[0]}.{coords[1]}.{counter}.jpg")
                cv.imwrite(img_path, left_eye)

                # Save the head position (modify filename to avoid overwriting)
                head_position_path = os.path.join(folder, f"head_position_{coords[0]}_{coords[1]}_{counter}.txt")
                with open(head_position_path, 'w') as f:
                    f.write(f"{coords[0]},{coords[1]},{int(head_position[0])},{int(head_position[1])}\n")

                # Print success message
                print(f"Image and head position saved successfully at {img_path} and {head_position_path}")

                counter += 1
        except Exception as e:
            print(f"Error: {e}")

    webcam.release()
    cv.destroyAllWindows()


def captureScreenPoints(num_points=32):
    screen_width, screen_height = pyautogui.size()
    grid_rows = 4
    grid_cols = 8
    step_x = screen_width // grid_cols
    step_y = screen_height // grid_rows

    coords = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * step_x + step_x // 2
            y = row * step_y + step_y // 2
            coords.append((x, y))

    return coords

# Main Execution Flow

# Create a new folder for this execution
run_folder = createNewFolder(base_folder="runs")

# Capture eye images and head positions at 32 points around the screen
coords = captureScreenPoints(32)

for (i, j) in coords:
    print(f"Capturing at {i}, {j}")
    pyautogui.moveTo(i, j)
    time.sleep(0.2)  # Add a delay to reduce lag and allow for smoother processing
    getEyeAndHead(times=5, coords=(i, j), counterStart=0, folder=run_folder)
