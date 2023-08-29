import glob, os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
from PIL import Image


class DataPrep:
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode

        txt_file = pd.read_csv('./class_list.txt', sep=' ', names=['label', 'name'], header=None)
        self.name_list = txt_file.name.to_list()
        self.label_list = txt_file.label.to_list()

        self.labels_dict = {}
        for i, name in enumerate(self.name_list):
            self.labels_dict[i] = name

        self.file_csv = pd.read_csv(f'./{mode}Labels_cropped.csv')

        for name in self.name_list:  # diagnosis
            train_name_path = os.path.join(self.root_dir, 'train', name)
            val_name_path = os.path.join(self.root_dir, 'val', name)
            os.makedirs(train_name_path, exist_ok=True)
            os.makedirs(val_name_path, exist_ok=True)

    # convert to png
    def convert_to_png(self, image_path):
        img = Image.open(image_path)
        png_path = os.path.splitext(image_path)[0] + '.png'
        img.save(png_path, 'PNG')

    def move_files(self):
        for label in self.label_list:
            mask = (self.file_csv.level == label)
            file_masked = self.file_csv.loc[mask].image.to_list()
            file_name = self.labels_dict[label]  # will be folder name

            file_masked_train, file_masked_test = train_test_split(file_masked,
                                                                   test_size=0.2, random_state=777)


            for image in tqdm(file_masked_train, desc=f'Moving {file_name} train images'):
                src_dir = os.path.join(f'./resized_train_cropped/resized_train_cropped', image+'.jpeg')
                dst_dir = os.path.join(self.root_dir, 'train', file_name, image+'.png')

                #self.convert_to_png(src_dir)
                shutil.copyfile(src_dir, dst_dir)

            for image in tqdm(file_masked_test, desc=f'Moving {file_name} val images'):
                src_dir = os.path.join(f'./resized_train_cropped/resized_train_cropped', image+'.jpeg')
                dst_dir = os.path.join(self.root_dir, 'val', file_name, image+'.png')

                #self.convert_to_png(src_dir)
                shutil.copyfile(src_dir, dst_dir)
        print('moving has finished.')