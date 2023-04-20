# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model
# To be deployed on raspberry pi

from buzzer_module import double_beep, bz_warn, bz_off, goodbye
from led_module import h_off, h_on, h_neutral, led_start, goodbye_led

double_beep()
led_start()

import tensorflow as tf
import numpy as np
import cv2
import time
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tflite_runtime.interpreter import Interpreter

face_detector_alt=cv2.CascadeClassifier("haar/haarcascade_frontalface_alt_tree.xml")
face_detector_def=cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
eye_detector=cv2.CascadeClassifier("haar/haarcascade_eye.xml")
model_path = '/home/pi/helmet_detection_thesis/Models/model4-3.tflite'
interpreter = Interpreter(model_path=model_path)

cap = cv2.VideoCapture(0)
helmet_off_timeout = 2
helmet_on_timeout = 3
neutral_time_hOff = time.time()
neutral_time_hOn = time.time()
elapsed_time_hOn = time.time()
helmetIsOn = False
warning_trigger = False
bz_warn_trigger = False
bz_triggered = False
warn_message = " "
color = 255, 255, 255

h_neutral()
# Loop over frames.
while cap.isOpened():
    elapsed_time_hOff = time.time() - neutral_time_hOff
    # Read a frame from the webcam.
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Preprocess the frame.
    input_frame = cv2.resize(frame, (161, 241))
    input_data = img_to_array(input_frame)
    input_data = tf.expand_dims(input_data, 0)

    # Run inference on the TFLite model.
    detect = interpreter.get_signature_runner('serving_default')

    # Run Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results_def = face_detector_def.detectMultiScale(gray, 1.3, 5)
    results_alt = face_detector_alt.detectMultiScale(gray, 1.3, 5)
    results_eye = eye_detector.detectMultiScale(gray, 1.3, 5)
    try:
        if not results_def and not results_alt:
            print("No face")
    except:
        print("Face detected")

    # Postprocess the output.
    class_names = ["helmet-off", "helmet-on"]
    prediction = detect(sequential_1_input=input_data)['outputs']
    score = tf.nn.softmax(prediction)
    status = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # color = 255, 255, 255  # TODO modify and apply threshold
    if status == "helmet-on":
        try:
            # not checks if naay sulod ang results
            # so if walay sulod ang results_def and results_alt, meaning walay nawong
            if not results_def and not results_alt:
                color = 0, 255, 0  # green
                elapsed_time_hOn = time.time() - neutral_time_hOn
                neutral_time_hOff = time.time()

                if elapsed_time_hOn > helmet_on_timeout:
                    neutral_time_hOff = time.time()  # reset timeout when helmet is on
                    warning_trigger = False
                    bz_warn_trigger = False
                    bz_triggered = False
                    warn_message = "Helmet detected"
                    bz_off()
                    h_on()

        except:  # False positive, if helmet-on pero naay nawong
            warn_message = "False Positive"
            warning_trigger = True
            color = 0, 0, 255
            neutral_time_hOn = time.time()

            if helmetIsOn:
                h_neutral()
                helmetIsOn = False

    else:
        warning_trigger = True
        color = 0, 0, 255
        neutral_time_hOn = time.time()

        if helmetIsOn:
            h_neutral()
            helmetIsOn = False

    # If n seconds has elapsed while helmet is off
    if elapsed_time_hOff >= helmet_off_timeout and warning_trigger == True and bz_warn_trigger == False and bz_triggered == False:
        bz_warn_trigger = True

    # this is to make sure that bz_warn does not run repeatedly.
    if bz_warn_trigger == True and warning_trigger == True and bz_triggered == False:
        bz_warn()
        h_off()
        warn_message = "Please wear your helmet"
        warning_trigger = False
        bz_warn_trigger = False
        bz_triggered = True
        h_off()

    #Draw bounding boxes
    for (x, y, w, h) in results_def:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in results_alt:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in results_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the predicted class label.
    cv2.putText(frame, warn_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, status + " " + str(confidence), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, str(fps) + "fps", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)

    # print(warning_trigger)
    # print(bz_warn_trigger)
    # print(bz_triggered)
    # print("helmet on time: " + str(elapsed_time_hOn))
    # print("helmet off time: " + str(elapsed_time_hOff))

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        goodbye()
        goodbye_led()
        break

# Release the webcam and close all windows.


cap.release()
cv2.destroyAllWindows()


