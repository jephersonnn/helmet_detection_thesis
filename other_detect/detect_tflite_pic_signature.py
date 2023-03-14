# Load Model
import tensorflow as tf
import numpy as np
import cv2

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/good trial_model Mar-10-2023 13_35_47.tflite'
file_path = '//Users/jeph/Dev/Python/Helmet_Detection/Test-data/8.jpg'
interpreter = tf.lite.Interpreter(model_path=model_path)

image = cv2.imread(file_path)

input_frame = cv2.resize(image, (180, 180))
input_data = tf.keras.utils.img_to_array(input_frame)
input_data = tf.expand_dims(input_data, 0)

# Run inference on the TFLite model.
detect = interpreter.get_signature_runner('serving_default')

# Postprocess the output.
class_names = ["helmet-off", "helmet-on"]
prediction = detect(sequential_1_input=input_data)['outputs']
print(prediction)
score = tf.nn.softmax(prediction)
status = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

color = 255, 255, 255  # TODO modify and apply threshold
if status == "helmet-on":
    color = 0, 255, 0
else:
    color = 0, 0, 255

cv2.putText(image, status + " " + str(confidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
cv2.imshow("Frame", image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
