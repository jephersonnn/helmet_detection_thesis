import cv2
import os
import pathlib
import tensorflow as tf
import numpy as np


model_path = '//Users/jeph/Dev/Python/Helmet_Detection/Models/model8-1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

#helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_main"
helmet_data_directory = "/Users/jeph/Dev/Python/Helmet_Detection/test"
helmet_on_directory = helmet_data_directory + "/helmet-on/"
helmet_off_directory = helmet_data_directory + "/helmet-off/"
print(helmet_off_directory)

print("Evaluating...")

def run_eval(helmet_dir):
    true_count = 0
    false_count = 0
    evaluated=0
    helmet_dir = pathlib.Path(helmet_dir)
    helmet_data = helmet_dir.glob('*.jpg')

    for hOn in helmet_data:
        img = cv2.imread(str(hOn))
        input_data = tf.keras.utils.img_to_array(img)
        input_data = tf.expand_dims(input_data, 0)

        # Run inference on the TFLite model.
        detect = interpreter.get_signature_runner('serving_default')

        class_names = ['helmet-off', 'helmet-on-blue', 'helmet-on-cloudy', 'helmet-on-indoor', 'helmet-on-running', 'helmet-on-tree', 'helmet-on-white']
        prediction = detect(sequential_2_input=input_data)['outputs']
        score = tf.nn.softmax(prediction)
        status = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)


        if status != "helmet-off":
            true_count += 1

        else:
            false_count += 1

        evaluated += 1
        print(status + " " + str(confidence))
        print("Evaluated " + str(evaluated) + " images for " + str(helmet_dir))
        os.system('clear')
    return true_count, false_count


hOn_true, hOn_false = run_eval(helmet_on_directory)
hOff_false, hOff_true = run_eval(helmet_off_directory)
print("Helmet-On T F:  " + str(hOn_true) + " " + str(hOn_false))
print("Helmet-Off T F: " + str(hOff_false) + " " + str(hOff_true))

tp = hOn_true
tn = hOff_true
fp = hOff_false
fn = hOn_false

total = tp + tn + fp + fn
accuracy = (tp + tn) / total
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)

print("Evaluation complete for model " + model_path)





