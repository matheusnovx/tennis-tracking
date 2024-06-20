import pickle
import cv2
from hog_quantification import quantify_image_HoG

def is_tennis_court(image, model_path='model.pickle', le_path='label_encoder.pickle', cell_size=12):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image_HoG(image, cell_size)

    prediction = model.predict([features])
    label = le.inverse_transform(prediction)[0]
    
    return label == "tennis"
