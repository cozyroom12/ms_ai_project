import os, glob
from torch.utils.data import Dataset
import cv2 

class RetDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.img_path = glob.glob(os.path.join(root_dir,mode,'*','*.jpeg'))
        self.label_list = os.listdir(os.path.join(root_dir, mode))

        label_dict = {}
        for i, label in enumerate(self.label_list):
            # {'mild': 0, 'moderate': 1, 'no_dr': 2, 'proliferative_dr': 3, 'severe': 4}
            label_dict[label] = i
        self.label_dict = label_dict

        self.transform = transform



    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_name = os.path.basename(os.path.dirname(self.img_path[idx]))
        label = self.label_dict[class_name]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return len(self.img_path)