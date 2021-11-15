import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
import cv2

class PANDA_dataset(Dataset):
    def __init__(self, path: str, train: str, valid_num=200):
        super(PANDA_dataset, self).__init__()
        self.train = train
        self.path = path
        self.data = []
        if train == "train" or train == "valid":
            with open(os.path.join(path, "train.csv")) as f:
                csv_f = csv.reader(f)
                for row in csv_f:
                    if row[0] != "image_id":
                        self.data.append([row[0], row[1], int(row[2]), row[3]])
            random.seed(17373331)
            random.shuffle(self.data)
            self.data = self.data[:200] if train == "valid" else self.data[200:]
        else:
            assert 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        x = self.data[x]
        cv2.imread(os.path.join(self.path,"image",))