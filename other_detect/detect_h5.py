# Detect using original model
# Load Model
import tensorflow as tf
import numpy as np
import cv2

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/trial_model Mar-10-2023 11_21_11.h5'
model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(1)

class_names = ['helmet-on', 'helmet-off']

# Loop over frames.
while cap.isOpened():
    # Read a frame from the webcam.
    ret, frame = cap.read()

    # Preprocess the frame.
    input_data = cv2.resize(frame, (180, 180))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float32) / 255.0

    predictions = model.predict(input_data)
    score = tf.nn.softmax(predictions[0])
    helmet_status = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)


    # Display the frame with the predicted class label.
    cv2.putText(frame, helmet_status + " " + str(confidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)


    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()