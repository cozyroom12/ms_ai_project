import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import torch 

path = './dataset_resized_cropped/'
train_df = pd.read_csv('trainLabels_cropped.csv')

from sklearn.utils import class_weight 
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=train_df['level'].values)
class_weights = torch.tensor(class_weights,dtype=torch.float)