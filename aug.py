
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from PIL import Image

def aug(source_folder, target_folder):
    train_transform = A.Compose([
        A.Blur(),
        A.Flip(),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(),
        A.ElasticTransform(),
        A.Transpose(),
        A.GridDistortion(),
        A.HueSaturationValue(),
        A.CLAHE(),
        A.CoarseDropout(),
    ])

    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith(".jpeg"):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            image = Image.open(source_path)
            image_np = np.array(image)
            augmented_image = train_transform(image=image_np)['image']

            augmented_image_pil = Image.fromarray(augmented_image)
            augmented_image_pil.save(target_path)

aug("./final/train/Mild","./final/aug/Mild")
aug("./final/train/No DR","./final/aug/No DR")
aug("./final/train/Moderate","./final/aug/Moderate")
aug("./final/train/Severe","./final/aug/Severe")
aug("./final/train/Proliferative DR","./final/aug/Proliferative DR")

