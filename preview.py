# Detect using TFLite Model
# Modified with appropriate prediction algorithm
# Load Model
#TODO: fix get_signature_runner as it invokes an error on raspberry pi


import cv2

cap = cv2.VideoCapture(0)

# Loop over frames.
while cap.isOpened():
    # Read a frame from the webcam.
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)

    # Exit if the user presses the "q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
