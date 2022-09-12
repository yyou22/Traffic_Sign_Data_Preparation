import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class GTSRB_Test(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.csv_file_name = 'GT-final_test.csv'

        csv_file_path = os.path.join(
            self.root_dir, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path, sep=';', usecols=["Filename", "ClassId"])

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId
