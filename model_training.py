import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_save_model(data, labels, model_path='model.pickle', le_path='label_encoder.pickle'):
    # Encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Check if we have at least two classes
    if len(np.unique(labels)) < 2:
        raise ValueError("The dataset must contain at least two classes.")

    # Split the data into training and testing sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Train the classifier
    print("[INFO] training classifier...")
    model = LinearSVC()
    model.fit(trainX, trainY)

    # Evaluate the classifier
    print("[INFO] evaluating classifier...")
    predictions = model.predict(testX)
    print(classification_report(testY, predictions, target_names=le.classes_))

    # Save the model and label encoder
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
