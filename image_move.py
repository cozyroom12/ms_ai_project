import os
import shutil
import glob

import pandas as pd
from sklearn.model_selection import train_test_split


class ImageMove :
    def __init__(self, org_folder, csv_dir):
        self.org_folder = org_folder
        self.csv_data = pd.read_csv(csv_dir)
        self.label = self.csv_data['level'].to_list()
        self.path = self.csv_data['image'].to_list()
    def move_images(self):
        os.makedirs('data/No DR', exist_ok=True)
        os.makedirs('data/Mild', exist_ok=True)
        os.makedirs('data/Moderate', exist_ok=True)
        os.makedirs('data/Severe', exist_ok=True)
        os.makedirs('data/Proliferative DR', exist_ok=True)
        file_path_list = glob.glob(os.path.join(self.org_folder, "*.jpeg"))

        last_folder = []
        for file_path in file_path_list:
            folder_name = os.path.splitext(os.path.basename(file_path))[0]
            last_folder.append(folder_name)

        for label, value1 in zip(self.label, self.path):
            for value2 in last_folder:
                if value1 == value2:
                    label = int(label)
                    if label  == 0 :
                        shutil.move(f"./diabetic-retinopathy-detection/aug/{value2}.jpeg", "data/No DR")
                    elif label  == 1:
                        shutil.move(f"./diabetic-retinopathy-detection/aug/{value2}.jpeg", "data/Mild")
                    elif label  == 2 :
                        shutil.move(f"./diabetic-retinopathy-detection/aug/{value2}.jpeg","data/Moderate")
                    elif label  == 3 :
                        shutil.move(f"./diabetic-retinopathy-detection/aug/{value2}.jpeg", "data/Severe")
                    elif label  == 4 :
                        shutil.move(f"./diabetic-retinopathy-detection/aug/{value2}.jpeg", "data/Proliferative DR")

test = ImageMove("./diabetic-retinopathy-detection/train","./diabetic-retinopathy-detection/trainLabels.csv/trainLabels.csv")
test.move_images()

class ImageDataMove :
    def __init__(self, org_dir, train_dir, val_dir):
        self.org_dir = org_dir
        self.train_dir = train_dir
        self.val_dir = val_dir

    def move_images(self):

        # file path list
        file_path_list01 = glob.glob(os.path.join(self.org_dir, "No DR", "*.jpeg"))
        file_path_list02 = glob.glob(os.path.join(self.org_dir,  "Mild", "*.jpeg"))
        file_path_list03 = glob.glob(os.path.join(self.org_dir,  "Moderate", "*.jpeg"))
        file_path_list04 = glob.glob(os.path.join(self.org_dir,  "Severe", "*.jpeg"))
        file_path_list05 = glob.glob(os.path.join(self.org_dir,  "Proliferative DR", "*.jpeg"))

        # data split
        NoDR_train_data_list , NoDR_val_data_list = train_test_split(file_path_list01, test_size=0.2)
        Mild_train_data_list , Mild_val_data_list = train_test_split(file_path_list02, test_size=0.2)
        Moderate_train_data_list , Moderate_val_data_list = train_test_split(file_path_list03, test_size=0.2)
        Severe_train_data_list, Severe_val_data_list = train_test_split(file_path_list04, test_size=0.2)
        ProliferativeDR_train_data_list, ProliferativeDR_val_data_list = train_test_split(file_path_list05, test_size=0.2)

        # file move
        self.move_file(NoDR_train_data_list, os.path.join(self.train_dir, "No DR"))
        self.move_file(NoDR_val_data_list, os.path.join(self.val_dir, "No DR"))
        self.move_file(Mild_train_data_list, os.path.join(self.train_dir, "Mild"))
        self.move_file(Mild_val_data_list, os.path.join(self.val_dir, "Mild"))
        self.move_file(Moderate_train_data_list, os.path.join(self.train_dir, "Moderate"))
        self.move_file(Moderate_val_data_list, os.path.join(self.val_dir, "Moderate"))
        self.move_file(Severe_train_data_list, os.path.join(self.train_dir, "Severe"))
        self.move_file(Severe_val_data_list, os.path.join(self.val_dir, "Severe"))
        self.move_file(ProliferativeDR_train_data_list, os.path.join(self.train_dir, "Proliferative DR"))
        self.move_file(ProliferativeDR_val_data_list, os.path.join(self.val_dir, "Proliferative DR"))

    def move_file(self, file_list, mov_dir):
        os.makedirs(mov_dir, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, mov_dir)

org_dir = "./data"
train_dir = "./final/train"
val_dir = "./final/val"

move_temp = ImageDataMove(org_dir, train_dir, val_dir)
move_temp.move_images()