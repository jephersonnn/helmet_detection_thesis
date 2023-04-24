import cv2
import numpy as np
from os.path import join
import pathlib

# helmet_on_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_grey3/helmet-on"
# helmet_off_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_grey3/helmet-off"
# helmet_on_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_toAdd/helmet_on"
# helmet_off_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_toAdd/helmet_off"
# helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_main"
# helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_main"
# helmet_on_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data/helmet-on"
# helmet_off_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data/helmet-off"
# helmet_on_directory = "/Users/jeph/Downloads/Documents/helmet-data/validation/evaluation_ds_normal copy/helmet-on"
# helmet_off_directory = "/Users/jeph/Downloads/Documents/helmet-data/validation/evaluation_ds_normal copy/helmet-off"

helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet-eval"
# helmet_data_directory = "/Users/jeph/Downloads/Documents/helmet-data/large"
#helmet_data_directory = "/Users/jeph/Dev/Python/Helmet_Detection/test"
helmet_on_directory = helmet_data_directory + "/helmet-on/"
helmet_off_directory = helmet_data_directory + "/helmet-off/"
mainSaveDir = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_grey"

alpha=1.0 #contrast 1.0-3.0
beta=80 #brightness 0-100

def run_greyscale(helmet_dir, save_dir):
    helmet_dir = pathlib.Path(helmet_dir)
    helmet_data = helmet_dir.glob('*.jpg')
    processed = 0

    for hOn in helmet_data:
        img = cv2.imread(str(hOn))
        gray = gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate the histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.max()
        cdf = hist_norm.cumsum()
        cutoff = np.percentile(gray, 0.5)
        print(cdf[int(cutoff)])
        if cdf[int(cutoff)] > 0.1:
            print("The image is underexposed.")
            gray2 = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        else:
            print("The image is properly exposed.")

        processed += 1
        print("Processed " + str(processed) + " images")

        cv2.imshow("Raw", img)
        cv2.imshow("Gray", gray)
        cv2.imshow("Gray 2", gray2)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


run_greyscale(helmet_on_directory, mainSaveDir + "/helmet-on")
run_greyscale(helmet_off_directory, mainSaveDir + "/helmet-off")
