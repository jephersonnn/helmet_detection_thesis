# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model
# To be deployed on raspberry pi

import tensorflow as tf
import numpy as np
import cv2
import time

# -----please uncomment on RPi-------
# from tensorflow.keras.preprocessing.image import img_to_array
# from tflite_runtime.interpreter import Interpreter

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/new model9.tflite'
# model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/good trial_model Mar-10-2023 13_35_47.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
#interpreter = Interpreter(model_path=model_path)

cap = cv2.VideoCapture(1)
neutral_time = time.time()
helmet_off_timeout = 5

# Loop over frames.
while cap.isOpened():
    elapsed_time = time.time() - neutral_time
    # Read a frame from the webcam.
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Preprocess the frame.
    input_frame = cv2.resize(frame, (161, 241))
    input_data = tf.keras.utils.img_to_array(input_frame)
    input_data = tf.expand_dims(input_data, 0)

    # Run inference on the TFLite model.
    detect = interpreter.get_signature_runner('serving_default')

    # Postprocess the output.
    class_names = ["helmet-off", "helmet-on"]
    prediction = detect(sequential_1_input=input_data)['outputs']
    score = tf.nn.softmax(prediction)
    status = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    color = 255, 255, 255  # TODO modify and apply threshold
    if status == "helmet-on":
        color = 0, 255, 0
        neutral_time = time.time()  # reset timeout when helmet is on
    else:
        color = 0, 0, 255

    #If n seconds has elapsed while helmet is off
    if elapsed_time >= helmet_off_timeout:
        cv2.putText(frame, "Please wear a helmet", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame with the predicted class label.
    cv2.putText(frame, status + " " + str(confidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, str(fps) + "fps", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
