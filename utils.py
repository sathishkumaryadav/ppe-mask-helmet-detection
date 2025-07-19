import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model/ppe_detector_model.h5")

# Define the class names in same order as folders used for training
class_names = ['helmet', 'mask', 'nohelmet', 'nomask']

def predict_image(image: Image.Image):
    # Resize and convert to array
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize to [0,1]
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_label, confidence


