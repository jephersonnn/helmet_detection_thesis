import cv2
import pathlib

cascade = cv2.CascadeClassifier('../haar/cascade7 s400n5.xml')

#parameters
scale =  1.3
neighbors = 150
sample_size = (36, 24)

helmet_on_directory = "../test/helmet-on"
helmet_off_directory = "../test/helmet-off"

print("Evaluating...")

def run_eval(helmet_dir):
    true_count = 0
    false_count = 0
    helmet_dir = pathlib.Path(helmet_dir)
    helmet_data = helmet_dir.glob('*.jpg')
    print(len(list(helmet_data)))

    for hOn in helmet_data:
        img = cv2.imread(str(hOn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        helmet_result = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=sample_size)

        try:
            if not helmet_result:
                true_count += 1

            else:
                false_count += 1

        except:
            false_count += 1

    return true_count, false_count


hOn_true, hOn_false = run_eval(helmet_on_directory)
hOff_true, hOff_false = run_eval(helmet_off_directory)
print("Helmet-On T F:  " + str(hOn_true) + " " + str(hOn_false))
print("Helmet-Off T F: " + str(hOff_true) + " " + str(hOff_false))

tp = hOn_true
tn = hOff_false
fp = hOff_true
fn = hOn_false

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("\nParameters: S:" + str(scale) + " N:" + str(neighbors))
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)

print("Evaluation complete")





