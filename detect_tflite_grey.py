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

face_detector=cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/model9-1.tflite'
# model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/good trial_model Mar-10-2023 13_35_47.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

cap = cv2.VideoCapture(1)
helmet_off_timeout = 1
helmet_on_timeout = 1
face_timeout = 1
dface_timeout = 1

fps_to = 1
fps_n = t.time()
fps_t = 0
fps_ave = 0
fps = 0

min_frames = 10
min_hOff_frames = 20
face_frames = 0
hOff_frames = 0

ntime_hOff = t.time()
ntime_h0n = t.time()
ntime_dface = t.time()
ntimer = t.time()
nfalse = t.time()

etime_hOn = 0
etime_hOff = 0
etime_dface = 0
etime_false = 0
timer = t.time() - ntimer

face_detected = False
face_to_helmet = False
reset_count = False
warning_trigger = False
bz_warn_trigger = False
bz_triggered = False
startTimer = True
display_message = " "
color = 255, 255, 255

# Loop over frames.
while cap.isOpened():
    #etime_hOff = t.time() - ntime_hOff #actual time minus the last recorded time =

    # Read a frame from the webcam.
    ret, frame = cap.read()
    fps+=1
    fps_t = t.time() - fps_n

    if (fps_t > 1):
        fps_ave = fps
        fps = 0
        fps_t = 0
        fps_n = t.time()


    # Preprocess the frame.

    #frame = cv2.convertScaleAbs(frame, alpha=(np.random.rand()), beta=(np.random.rand()))
    input_frame = cv2.resize(frame, (161, 241))
    print(input_frame.shape)
    input_data = tf.keras.utils.img_to_array(input_frame)
    input_data = tf.expand_dims(input_data, 0)

    # Run inference on the TFLite model.
    detect = interpreter.get_signature_runner('serving_default')

    #Run Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results_def = face_detector.detectMultiScale(gray, 1.3, 1)

    # Postprocess the output.
    class_names = ['helmet-off', 'helmet-on']
    prediction = detect(sequential_15_input=input_data)['outputs']
    score = tf.nn.softmax(prediction)
    status = class_names[np.argmax(score)]

    print(status)
    #confidence = 100 * np.max(score)

    try: #face not detected
        etime_dface = t.time() - ntime_dface
        if not results_def:
            face_frames += 0

    except: #detected face
        face_frames += 1
        etime_dface = t.time() - ntime_dface

    if etime_dface > dface_timeout:

        if face_frames >= min_frames:
            face_detected = True

        else:
            face_detected = False

        face_frames = 0
        etime_dface = 0
        ntime_dface = t.time()

    if status != "helmet-off":

            # if face_detected and face_to_helmet == False:

            # if face_detected: #then ayha na dayon pwede mag helmet on
            color = 0, 255, 0  # green
            etime_hOn = t.time() - ntime_h0n
            ntime_hOff = t.time()
            #display_message = "Confirming helmet..."

            if etime_hOn > helmet_on_timeout: #Final Helmet On
                warning_trigger = False
                bz_warn_trigger = False
                bz_triggered = False
                hOff_frames = 0

                if face_detected:
                    display_message = "False Positive"

                else:
                    display_message = "Helmet On"
                # bz_off()
                # h_on()

            # else: #False positive, if helmet-on pero naay nawong
            #     display_message = "Show yourself in the screen"
            #     warning_trigger = True
            #     color = 0, 0, 255
            #     ntime_h0n = t.time()
            #     # if helmetIsOn:
            #     #     h_neutral()
            #     #     helmetIsOn = False

    else:
        hOff_frames += 1
        warning_trigger = True
        color = 0, 0, 255
        ntime_h0n = t.time()

        if hOff_frames > 10:
            etime_hOff = t.time() - ntime_hOff

        else:
            ntime_hOff = t.time()

    # If n seconds has elapsed while helmet is off
    if etime_hOff > helmet_off_timeout and warning_trigger == True and bz_warn_trigger == False and bz_triggered == False:

        if hOff_frames >= min_hOff_frames:
            bz_warn_trigger = True

        hOff_frames = 0
        etime_hOff = 0

    # if helmet_off_frames > min_frames and etime_hOff >= helmet_off_timeout and warning_trigger == True and bz_warn_trigger == False and bz_triggered == False:
    #     bz_warn_trigger = True

    # this is to make sure that bz_warn does not run repeatedly.
    if bz_warn_trigger == True and warning_trigger == True and bz_triggered == False:
        display_message = "Please wear your helmet"
        ntime_h0n = t.time()
        warning_trigger = False
        bz_warn_trigger = False
        bz_triggered = True
        #insert warnings
        print("Beep beep")

    # Display the frame with the predicted class label.
    cv2.putText(frame, display_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #cv2.putText(frame, status + " " + str(confidence), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, str(fps_ave) + "fps", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)

    print("---")
    print("Warning: ", bz_triggered)
    print("Detected: " + status)
    print("Face", face_detected)
    print("helmet on time: " + str(etime_hOn))
    print("helmet off time: " + str(etime_hOff))
    print("detected time: " + str(etime_dface))
    print("Face Frames: " + str(face_frames))
    print("Helmet Off frames: " + str(hOff_frames))

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
