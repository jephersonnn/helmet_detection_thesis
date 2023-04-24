import cv2
import os
import pathlib
import tensorflow as tf
import numpy as np

model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/model9-16.tflite' #sequential_1
#model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/model11-3.tflite' #sequential_1
interpreter = tf.lite.Interpreter(model_path=model_path)
face_detected = False
face_detected_alt = False

face_detector=cv2.CascadeClassifier('/Users/jeph/Dev/Python/Helmet_Detection/haar/haarcascade_frontalface_default.xml')
face_detector_alt=cv2.CascadeClassifier('/Users/jeph/Dev/Python/Helmet_Detection/haar/haarcascade_frontalface_alt_tree.xml')
# helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data"
#helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_grey"
# helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_main"
helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet-eval3"
# helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/large"
#helmet_data_directory = "/Users/jeph/Dev/Python/Helmet_Detection/test"
helmet_on_directory = helmet_data_directory + "/helmet-on/"
helmet_off_directory = helmet_data_directory + "/helmet-off/"
print(helmet_off_directory)
print("Evaluating...")

alpha=1.0 #contrast 1.0-3.0
beta=80 #brightness 0-100

def run_eval(helmet_dir):
    true_count = 0
    false_count = 0
    evaluated = 0

    helmet_dir = pathlib.Path(helmet_dir)
    helmet_data = helmet_dir.glob('*.jpg')

    for hOn in helmet_data:

        img = cv2.imread(str(hOn))
        img = cv2.resize(img, (241, 161)) #model9
        #img = cv2.resize(img, (181, 102)) #model11
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Normalize the histogram
        hist_norm = hist.ravel() / hist.max()

        # Calculate the cumulative distribution function (CDF)
        cdf = hist_norm.cumsum()

        # Find the cutoff value where 1% of the pixels are brighter
        cutoff = np.percentile(gray, 0.5)

        print(cdf[int(cutoff)])
        if cdf[int(cutoff)] > 0.1:
            print("The image is underexposed.")
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        # else:
            print("The image is properly exposed.")

        input_data = tf.keras.utils.img_to_array(gray)
        input_data = tf.expand_dims(input_data, 0)

        # Run inference on the TFLite model.
        detect = interpreter.get_signature_runner('serving_default')

        class_names = ['helmet-off' , 'helmet-on']

        prediction = detect(sequential_input=input_data)['outputs']
        score = tf.nn.softmax(prediction)
        status = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        if status == "helmet-on" and confidence > 95:
            true_count += 1


        else:
            false_count += 1



        evaluated += 1
        print("-------------------------")
        print(status + " " + str(confidence))
        # print("Face detected:", face_detected)
        print("Evaluated " + str(evaluated) + " images for " + str(helmet_dir))
        print("T: " + str(true_count) + " | F: " + str(false_count))
        cv2.imshow("Frame", gray)
        # cv2.imshow("Frame 2", gray2)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return true_count, false_count


hOn_true, hOn_false = run_eval(helmet_on_directory)
hOff_false, hOff_true = run_eval(helmet_off_directory)
print("Helmet-On  T F:  " + str(hOn_true) + " " + str(hOn_false))
print("Helmet-Off T F:  " + str(hOff_true) + " " + str(hOff_false))

tp = hOn_true
tn = hOff_true
fp = hOff_false
fn = hOn_false

total = tp + tn + fp + fn
accuracy = (tp + tn) / total
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print('Accuracy:', accuracy * 100)
print('Precision:', precision * 100)
print('Recall:', recall * 100)
print('F1-score:', f1_score * 100)

print("Evaluation complete for model " + model_path)
