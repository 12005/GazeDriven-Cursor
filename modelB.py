import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Load images and labels
def load_data(base_folder="runsB"):
    data = []
    labels = []
    
    # Go through the directories and load images and corresponding labels
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".jpg"):
                    # Read the image
                    image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (100, 50))  # Resize to (100x50)
                    
                    # Normalize the image
                    image = image / 255.0
                    
                    # Extract the label from the filename (open or closed)
                    label = file.split('_')[0]
                    
                    data.append(image)
                    labels.append(label)
    
    # Convert to numpy arrays
    data = np.array(data).reshape(-1, 50, 100, 1)  # Shape: (num_samples, height, width, channels)
    
    # Convert labels to numeric values (open: 0, closed: 1)
    labels = np.array([0 if label == 'open' else 1 for label in labels])
    
    return data, labels

# Prepare the CNN model
def create_cnn_model():
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output and add fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (open or closed)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train and save a single model
def train_and_save_model(model_num, X_train, y_train, X_val, y_val):
    model = create_cnn_model()
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the model with a unique name in the 'modelsB' folder
    model.save(f"modelsB/eye_open_closed_cnn_model_{model_num}.h5")

# Main function to train multiple models and create an ensemble
def main():
    # Load data
    data, labels = load_data(base_folder="runsB")

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Train and save multiple models
    num_models = 1  # Number of models in the ensemble
    for model_num in range(num_models):
        train_and_save_model(model_num, X_train, y_train, X_val, y_val)

    print(f"Trained and saved {num_models} models.")

    # Evaluate ensemble model on test data
    ensemble_evaluate(X_test, y_test, num_models)

# Function to evaluate an ensemble of models
def ensemble_evaluate(X_test, y_test, num_models):
    # Load the models from the 'modelsB' folder
    models = [tf.keras.models.load_model(f"modelsB/eye_open_closed_cnn_model_{i}.h5") for i in range(num_models)]
    
    # Get predictions from each model
    predictions = np.zeros(X_test.shape[0])  # Initialize an array for combined predictions
    for model in models:
        predictions += model.predict(X_test).reshape(-1)  # Sum up the predictions
    
    # Average the predictions
    predictions /= num_models
    
    # Convert predictions to class labels (0 for open, 1 for closed)
    predicted_classes = (predictions > 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.sum(predicted_classes == y_test) / len(y_test)
    print(f"Ensemble Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
