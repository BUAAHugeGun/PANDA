import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import csv
import random
import cv2
import numpy as np
import time


class PANDA_dataset(Dataset):
    def __init__(self, path: str, split: str, valid_num, valid_blocks, IMAGE_DIR="image_small"):
        super(PANDA_dataset, self).__init__()
        self.IMAGE_DIR = IMAGE_DIR
        self.split = split
        self.path = path
        self.data = []
        self.valid_blocks = valid_blocks
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(45, fill=255)
        ])
        if split == "train" or split == "valid":
            csv_name = "train.csv" if IMAGE_DIR == "image" else "train_small.csv"
            with open(os.path.join(path, csv_name), 'r') as f:
                csv_f = csv.DictReader(f)
                for row in csv_f:
                    self.data.append(
                        [row['image_id'], row['data_provider'], int(row['isup_grade']), row['gleason_score']])
            self.data_check()
            self.data.sort()
            self.data = self.data[:valid_num] if split == "valid" else self.data[valid_num:]
            print("Using split = {}, data = [{} ... {}].".format(split, self.data[0][0], self.data[-1][0]))
        else:
            assert 0

    def data_check(self):
        missing = []
        for x in self.data:
            name = x[0]
            if not os.path.exists(os.path.join(self.path, self.IMAGE_DIR, name + ".png")):
                missing.append(name)
        if len(missing):
            print("Missing images:", name)
            assert 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        x = self.data[x]
        img = cv2.imread(os.path.join(self.path, self.IMAGE_DIR, x[0] + ".png"))
        img = img.reshape(8, 256, 8, 256, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 256, 256, 3)
        weights = []
        for i in range(img.shape[0]):
            weights.append((img[i] < 240).sum() / 256 / 256 + 1e-5)  # [0.0 ~ 3.0]
            if self.split == "train":
                img[i] = np.array(self.transform(img[i]))
        weights = np.array(weights)
        weights /= weights.sum()
        idx = np.random.choice(np.arange(img.shape[0]), self.valid_blocks, p=weights, replace=False)
        img = img[idx]
        img = img.reshape(-1, 256, 3).transpose(1, 0, 2)
        img = self.toTensor(img)
        return img, x[2]


def build_data(path: str, batch_size, train: str, num_worker, valid_num=200, valid_block=25, IMAGE_DIR="image_small"):
    return DataLoader(PANDA_dataset(path, train, valid_num, valid_block, IMAGE_DIR), batch_size, shuffle=True,
                      num_workers=num_worker)
