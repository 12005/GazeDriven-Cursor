import numpy as np
import cv2 as cv
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load the data from all folders inside the 'runs' folder
def load_data(root_folder):
    images = []
    head_positions = []
    labels = []

    # Iterate over all subfolders in the 'runs' folder
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            print(f"Loading data from folder: {folder_path}")  # Print the folder being processed
            image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

            for img_path in image_paths:
                # Read the image
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (100, 50))  # Ensure all images are resized to 100x50
                img = np.expand_dims(img, axis=-1)  # Add channel dimension

                # Extract coordinates from the filename (e.g., '404.383.0.jpg')
                filename = os.path.basename(img_path).split('.')[0:2]
                coords = tuple(map(int, filename))

                # Find the head position file
                head_position_file = None
                for file in os.listdir(folder_path):
                    if file.startswith(f"head_position_{coords[0]}_{coords[1]}") and file.endswith(".txt"):
                        head_position_file = os.path.join(folder_path, file)
                        break

                if head_position_file and os.path.exists(head_position_file):
                    with open(head_position_file, "r") as f:
                        head_position = list(map(float, f.readline().strip().split(',')))[2:]
                else:
                    # Default to some value if not available (you might want to update this based on your head tracking data)
                    head_position = [0.0, 0.0]

                # Append to the lists
                images.append(img)
                head_positions.append(head_position)
                labels.append(coords)

    images = np.array(images, dtype="float32") / 255.0  # Normalize the images
    head_positions = np.array(head_positions, dtype="float32")  # Head positions as float32
    labels = np.array(labels, dtype="float32")  # Keep labels as float32

    return images, head_positions, labels

# Load the dataset from 'runs' folder
images, head_positions, labels = load_data("runs")
print(images[0],head_positions[2],labels[0])

# Split into training and test sets
X_train_eyes, X_test_eyes, X_train_heads, X_test_heads, y_train, y_test = train_test_split(
    images, head_positions, labels, test_size=0.2, random_state=42)

# Define the CNN model with both eye image and head position as inputs
def build_model(input_shape_eye, input_shape_head):
    # Eye image input branch
    eye_input = layers.Input(shape=input_shape_eye)
    x = layers.Conv2D(32, (3, 3), activation='relu')(eye_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Head position input branch
    head_input = layers.Input(shape=input_shape_head)
    head_x = layers.Dense(64, activation='relu')(head_input)
    head_x = layers.Dense(32, activation='relu')(head_x)

    # Concatenate eye and head features
    combined = layers.Concatenate()([x, head_x])

    # Final layers
    combined = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(2)(combined)  # 2 output units for (x, y) coordinates

    model = models.Model(inputs=[eye_input, head_input], outputs=output)
    return model

# Build the model
input_shape_eye = (50, 100, 1)  # Image shape (height, width, channels)
input_shape_head = (2,)  # Head position shape (x, y)

model = build_model(input_shape_eye, input_shape_head)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model with both eye images and head positions
history = model.fit([X_train_eyes, X_train_heads], y_train, epochs=20, batch_size=32, validation_data=([X_test_eyes, X_test_heads], y_test))

# Save the model after training
model.save('models/eye_gaze_model_with_head.keras')

# Evaluate the model
test_loss, test_mae = model.evaluate([X_test_eyes, X_test_heads], y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# To predict using a new image and head position
def predict_gaze(image_path, head_position, model):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    img = cv.resize(img, (100, 50))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    head_position = np.expand_dims(head_position, axis=0)  # Add batch dimension
    prediction = model.predict([img, head_position])
    return prediction

# Example of prediction on a new image and head position
new_image_path = 'runs/0/360.135.0.jpg'  # Adjust path as needed
new_head_position = [373,303]  # Replace with actual head position values
gaze_coords = predict_gaze(new_image_path, new_head_position, model)
print(f"Predicted gaze coordinates: {gaze_coords}")
