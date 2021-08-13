"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
from glob import glob

# 2018 path = 'MSO/2018_Case_Image_Dataset/*/*'
# 2019 path = 'MSO/2019_Case_Image_Dataset/*/*/*/*'


class MSODataset(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.ids = []
        self.path = '../MSO/2018_Case_Image_Dataset/*/*'
        self.globs = glob(self.path)

        for filepath in self.globs:
            if filepath.lower().endswith(('jpg', 'png', 'jpeg')):  # ignore Thumbs.db and .gif
                self.ids.append(filepath)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.ids[idx]
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, 0 # MSO dataset has no labels


