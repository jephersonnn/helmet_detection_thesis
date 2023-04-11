# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model
# TODO: Make sure to copy to detect.py
import tensorflow as tf
import numpy as np
import cv2
import time


model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/model4-3.tflite'
# model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/good trial_model Mar-10-2023 13_35_47.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

cap = cv2.VideoCapture(1)
helmet_off_timeout = 2
helmet_on_timeout = 3
neutral_time_hOff = time.time()
neutral_time_hOn = time.time()
elapsed_time_hOn = time.time()
warning_trigger = False
bz_warn_trigger = False
bz_triggered = False
warn_message = " "
color = 255, 255, 255

# Loop over frames.
while cap.isOpened():
    elapsed_time_hOff = time.time() - neutral_time_hOff
    # Read a frame from the webcam.
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Preprocess the frame.

    #frame = cv2.convertScaleAbs(frame, alpha=(np.random.rand()), beta=(np.random.rand()))
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

    if status == "helmet-on":
        color = 0, 255, 0  # green
        elapsed_time_hOn = time.time() - neutral_time_hOn
        neutral_time_hOff = time.time()

        if elapsed_time_hOn > helmet_on_timeout:
            neutral_time_hOff = time.time()  # reset timeout when helmet is on
            warning_trigger = False
            bz_warn_trigger = False
            bz_triggered = False
            warn_message = "Helmet detected"

    else:
        warning_trigger = True
        color = 0, 0, 255
        neutral_time_hOn = time.time()

    # If n seconds has elapsed while helmet is off
    if elapsed_time_hOff >= helmet_off_timeout and warning_trigger == True and bz_warn_trigger == False and bz_triggered == False:
        bz_warn_trigger = True


    # this is to make sure that bz_warn does not run repeatedly.
    if bz_warn_trigger == True and warning_trigger == True and bz_triggered == False:
        warn_message = "Please wear your helmet"
        neutral_time_hOn = time.time()
        warning_trigger = False
        bz_warn_trigger = False
        bz_triggered = True


    # Display the frame with the predicted class label.
    cv2.putText(frame, warn_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, status + " " + str(confidence), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, str(fps) + "fps", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)

    print("helmet on time: " + str(elapsed_time_hOn))
    print("helmet off time: " + str(elapsed_time_hOff))

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
