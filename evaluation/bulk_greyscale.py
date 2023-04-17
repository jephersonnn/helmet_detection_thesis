import cv2
from os.path import join
import pathlib

helmet_on_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data/helmet-on"
helmet_off_directory = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data/helmet-off"
# helmet_on_directory = "/Users/jeph/Downloads/Documents/helmet-data/validation/evaluation_ds_normal copy/helmet-on"
# helmet_off_directory = "/Users/jeph/Downloads/Documents/helmet-data/validation/evaluation_ds_normal copy/helmet-off"
mainSaveDir = "/Users/jeph/Downloads/Documents/helmet-data/helmet_data_grey"


def run_greyscale(helmet_dir, save_dir):
    helmet_dir = pathlib.Path(helmet_dir)
    helmet_data = helmet_dir.glob('*.jpg')
    processed = 0

    for hOn in helmet_data:
        img = cv2.imread(str(hOn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        save = join(save_dir,hOn)
        cv2.imwrite(save, gray)
        processed += 1
        print("Processed " + str(processed) + " images")


run_greyscale(helmet_on_directory, mainSaveDir + "/helmet-on")
run_greyscale(helmet_off_directory, mainSaveDir + "/helmet-off")
