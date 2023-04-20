# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model
# TODO: Make sure to copy to detect_notimeout.py
import tensorflow as tf
import numpy as np
import cv2
import time

helmet_detector=cv2.CascadeClassifier("haar/cascade7 s400n5.xml")

cap = cv2.VideoCapture(1)
# Loop over frames.
while cap.isOpened():

    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Preprocess the frame.

    #frame = cv2.convertScaleAbs(frame, alpha=(np.random.rand()), beta=(np.random.rand()))
    input_frame = cv2.resize(frame, (161, 241))
    input_data = tf.keras.utils.img_to_array(input_frame)
    input_data = tf.expand_dims(input_data, 0)

    #Run Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results_helmet = helmet_detector.detectMultiScale(gray, 5.00, 20)


    for (x, y, w, h) in results_helmet:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
