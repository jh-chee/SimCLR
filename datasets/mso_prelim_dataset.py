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


class MSOPrelimDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.labels = [i for i in sorted(os.listdir(self.root_dir)) if i not in ['.DS_Store', 'README.txt']]
        self.ids = []
        self._load_file()

    def _load_file(self):
        for label in self.labels:
            try:
                label_path = os.path.join(self.root_dir, label)
                label_imgs = sorted(os.listdir(label_path))
                for img in label_imgs:
                    full_path = os.path.join(label_path, img)
                    self.ids.append((full_path, label))
            except NotADirectoryError as e:
                print(e)
                continue

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, label = self.ids[idx]
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels.index(label)


