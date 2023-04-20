# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model
# TODO: Make sure to copy to detect_notimeout.py
import tensorflow as tf
import numpy as np
import cv2
import time as t

print(tf.__version__)
print(cv2.__version__)

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/model9-2.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

cap = cv2.VideoCapture(1)
# Loop over frames.
while cap.isOpened():

    ret, frame = cap.read()
    #frame = cv2.convertScaleAbs(frame, alpha=(np.random.rand()), beta=(np.random.rand()))
    x, y, w, h = 250, 100, 1280, 720
    frame = frame[y:y+h, x:x+w]
    input_frame = cv2.resize(frame, (241, 161), fx=0.1, fy=0) #model9
    #input_frame = cv2.resize(frame, (181, 102), fx=1, fy=1)  #model11
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    input_data = tf.keras.utils.img_to_array(gray)
    input_data = tf.expand_dims(input_data, 0)

    # Run inference on the TFLite model.
    detect = interpreter.get_signature_runner('serving_default')


    # Postprocess the output.
    class_names = ['helmet-off', 'helmet-on']
    prediction = detect(sequential_1_input=input_data)['outputs']
    score = tf.nn.softmax(prediction)
    status = class_names[np.argmax(score)]

    print("Helmet Status....: " + status)
    cv2.imshow("Frame", gray)

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
