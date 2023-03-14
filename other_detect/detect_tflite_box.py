# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model

import tensorflow as tf
import numpy as np
import cv2

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/good trial_model1.tflite'
#model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/good trial_model Mar-10-2023 13_35_47.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

cap = cv2.VideoCapture(1)
rectColors = [(0, 255, 0), (255, 0, 0)]
class_names = ["helmet-off", "helmet-on"]

# Loop over frames.
while cap.isOpened():
    # Read a frame from the webcam.
    ret, frame = cap.read()

    # Preprocess the frame.
    input_frame = cv2.resize(frame, (180, 180))
    input_data = tf.keras.utils.img_to_array(input_frame)
    input_data = tf.expand_dims(input_data, 0)

    # Run inference on the TFLite model.
    detect =  interpreter.get_signature_runner('serving_default')

    # Postprocess the output.

    prediction = detect(sequential_1_input=input_data)['outputs']
    print(prediction)
    score = tf.nn.softmax(prediction)
    status = class_names[np.argmax(score)]
    status_id = np.argmax(score)
    confidence = 100 * np.max(score)
    box_color = rectColors[status_id]

    # Draw the bounding box
    height, width, _ = frame.shape
    left = int(width * 0.25)
    top = int(height * 0.25)
    right = int(width * 0.75)
    bottom = int(height * 0.75)
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness=2)

    color = 255,255,255 #TODO modify and apply threshold
    if status == "helmet-on": color = 0,255,0
    else: color = 0,0,255


    # Display the frame with the predicted class label.
    cv2.putText(frame, status + " " + str(confidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Helmet Detect", frame)

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
