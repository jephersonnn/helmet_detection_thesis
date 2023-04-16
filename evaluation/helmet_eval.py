import cv2
import pathlib
import seaborn

cascade = cv2.CascadeClassifier('../haar/haarcascade_frontalface_default.xml')

helmet_on_directory = "../Test-data"
helmet_off_directory = ''

def run_eval(helmet_dir):
    true_count = 0
    false_count = 0
    helmet_dir = pathlib.Path(helmet_dir)
    helmet_data = helmet_dir.glob('*.jpg')

    for hOn in helmet_data:
        img = cv2.imread(str(hOn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        helmet_result = cascade.detectMultiScale(gray, 1.35, 3)

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
print(str(hOn_true) + " " + str(hOn_false))

tp = hOn_true
tn = hOff_false
fp = hOff_true
fn = hOn_false

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)




