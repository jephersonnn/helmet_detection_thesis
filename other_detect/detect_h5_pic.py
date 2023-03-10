# Detect using original model
# Load Model
import tensorflow as tf
import numpy as np
import cv2

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/trial_model4.h5'
file_path = '//Users/jeph/Dev/Python/Helmet_Detection/Test-data/17.jpg'
model = tf.keras.models.load_model(model_path)
class_names = ['helmet-on', 'helmet-off']

image = cv2.imread(file_path)

# Preprocess the frame.
input_data = cv2.resize(image, (480, 480))
input_data = np.expand_dims(input_data, axis=0)
input_data = input_data.astype(np.float32) / 255.0

predictions = model.predict(input_data)
score = tf.nn.softmax(predictions[0])
helmet_status = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(
    "{} {:.2f}%"
        .format(helmet_status, confidence)
)