# Load Model
import tensorflow as tf
import numpy as np
import cv2

model_path = '//Users/jeph/Dev/Python/detect.py/trial_model.tflite'
file_path = '//Users/jeph/Dev/Python/detect.py/Test-data/1.jpg'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(output_details)

image = cv2.imread(file_path)

input_shape = interpreter.get_input_details()[0]['shape']
input_data = cv2.resize(image, (input_shape[1], input_shape[2]))
input_data = np.expand_dims(input_data, axis=0)
input_data = input_data.astype(np.float32) / 255.0

# Run inference on the TFLite model.
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# Postprocess the output.
class_labels = ["helmet-off", "helmet-on"]
predicted_class = class_labels[np.argmax(output_data)]

print(predicted_class)
print(str(np.max(output_data)))
print(output_data)

