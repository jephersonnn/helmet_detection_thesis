# Detect using TFLite Model
# Load Model
import tensorflow as tf
import numpy as np
import cv2

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/trial_model1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)

cap = cv2.VideoCapture(1)

# Loop over frames.
while cap.isOpened():
    # Read a frame from the webcam.
    ret, frame = cap.read()

    # Preprocess the frame.
    input_shape = interpreter.get_input_details()[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float32) / 255.0

    # Run inference on the TFLite model.
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    # Postprocess the output.
    class_labels = ["helmet-on", "helmet-off"]
    predicted_class = class_labels[np.argmax(output_data)]
    print(output_data)

    # Display the frame with the predicted class label.
    cv2.putText(frame, predicted_class + " " + str(np.max(output_data)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()