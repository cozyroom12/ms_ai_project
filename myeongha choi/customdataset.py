import os

import cv2
from torch.utils.data import Dataset
from PIL import Image
import glob

class CustomDataset(Dataset) :
    def __init__(self, data_dir, transform=None):
        # data_dir = ./data/train/
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpeg"))
        self.transform = transform
        self.label_dict = {"No DR" : 0 , "Mild" : 1, "Moderate" : 2,'Severe':3, 'Proliferative DR':4}

    def __getitem__(self, item):
        image_path = self.data_dir[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_name = image_path.split("\\")[1]
        label = self.label_dict[label_name]

        if self.transform is not None :
            image = self.transform(image=image)['image']

        return image ,label

    def __len__(self):
        return len(self.data_dir)
