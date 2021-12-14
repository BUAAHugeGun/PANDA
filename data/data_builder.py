import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import csv
import random
import cv2
import numpy as np
import time
import albumentations
import skimage

import skimage.io
from skimage import transform

sz = 256
N = 64
sN = 8
scale_factor = 1


def tile(img):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=255)
    #     mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
    #                 constant_values=0)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    #     mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    #     mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        #         mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    #     mask = mask[idxs]
    av_blk = 0
    for i in range(N):
        ss = (img[i] < 240).sum()
        if ss / sz / sz > 0.5:
            av_blk += 1
    img = np.array(img).reshape(sN, sN, sz, sz, 3).transpose(0, 2, 1, 3, 4).reshape(sN * sz, sN * sz, 3)
    #     mask = np.array(mask).reshape(sN,sN,sz,sz,3).transpose(0,2,1,3,4).reshape(sN*sz,sN*sz,3)
    #     for i in range(len(img)):
    #         imgs.append(img)
    #         result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return img, av_blk


class PANDA_dataset(Dataset):
    def __init__(self, data_path="./data", split="train", valid_blocks=25, kfold=0, nfold=5, image_dir="images_all",
                 scale_factor=1):
        super(PANDA_dataset, self).__init__()
        self.split = split
        self.path = data_path
        self.data = []
        self.valid_blocks = valid_blocks
        self.toTensor = transforms.ToTensor()
        self.IMAGE_DIR = image_dir
        self.scale_factor = scale_factor
        self.transform = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
        ])
        if split == "train" or split == "valid":
            with open(os.path.join(data_path, "train.csv"), 'r') as f:
                csv_f = csv.DictReader(f)
                for row in csv_f:
                    self.data.append(
                        [row['image_id'], row['data_provider'], int(row['isup_grade']), row['gleason_score']])
            self.data_check()
            self.data.sort()
            valid_begin = int(len(self.data) / nfold * kfold)
            valid_end = int(len(self.data) / nfold * (kfold + 1))
            if split == "valid":
                self.data = self.data[valid_begin:valid_end]
                print("Using split = {}, data = [{} ... {}], size = {}.".format(
                    split, self.data[0][0], self.data[-1][0], len(self.data)))
            else:
                self.data = self.data[:valid_begin] + self.data[valid_end:]
                print("Using split = {}, data = [{} ...(val)... {}], size = {}.".format(
                    split, self.data[0][0], self.data[-1][0], len(self.data)))
        else:
            print(os.path.join(self.path, self.IMAGE_DIR))
            for root, dirs, files in os.walk(os.path.join(self.path, self.IMAGE_DIR)):
                for file in files:
                    self.data.append(file.split('.')[0])

    def data_check(self):
        missing = []
        for x in self.data:
            name = x[0]
            if not os.path.exists(os.path.join(self.path, self.IMAGE_DIR, name + ".png")):
                missing.append(name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        if self.split == "test":
            x = self.data[x]
            img = skimage.io.MultiImage(os.path.join(self.path, self.IMAGE_DIR, x + ".tiff"))[1]
            img, av_blk = tile(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.reshape(8, 256, 8, 256, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 256, 256, 3)
            img = img[0:self.valid_blocks]
            img = img.reshape(-1, 256, 3).transpose(1, 0, 2)
            img = self.toTensor(255 - img)
            shape = img.shape
            img = functional.interpolate(img.unsqueeze(0),
                                         size=(shape[1] // self.scale_factor, shape[2] // self.scale_factor))
            return img.squeeze(0), x[2]
        else:
            x = self.data[x]
            img = cv2.imread(os.path.join(self.path, self.IMAGE_DIR, x[0] + ".png"))
            img = img.reshape(8, 256, 8, 256, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 256, 256, 3)
            weights = []
            for i in range(img.shape[0]):
                weights.append((img[i] < 240).sum() / 256 / 256 + 1e-5)  # [0.0 ~ 3.0]
                if self.split == "train":
                    img[i] = self.transform(image=img[i])["image"]
            weights = np.array(weights)
            weights /= weights.sum()
            idx = np.random.choice(np.arange(img.shape[0]), self.valid_blocks, p=weights, replace=False)
            img = img[idx]
            img = img.reshape(-1, 256, 3).transpose(1, 0, 2)
            img = self.toTensor(255 - img)
            shape = img.shape
            img = functional.interpolate(img.unsqueeze(0),
                                         size=(shape[1] // self.scale_factor, shape[2] // self.scale_factor))
            return img.squeeze(0), x[2]


def build_data(batch_size, num_worker, **kwargs):
    return DataLoader(PANDA_dataset(**kwargs), batch_size, shuffle=True,
                      num_workers=num_worker)
