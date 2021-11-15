import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
import cv2
import numpy as np
import time


class PANDA_dataset(Dataset):
    def __init__(self, path: str, train: str, valid_num=200, valid_blocks=64):
        super(PANDA_dataset, self).__init__()
        self.train = train
        self.path = path
        self.data = []
        self.valid_num = valid_num
        self.valid_blocks = valid_blocks
        if train == "train" or train == "valid":
            with open(os.path.join(path, "train.csv")) as f:
                csv_f = csv.reader(f)
                for row in csv_f:
                    if row[0] != "image_id":
                        self.data.append([row[0], row[1], int(row[2]), row[3]])
            # if not self.data_check():
            #     assert 0
            random.seed(17373331)
            random.shuffle(self.data)
            self.data = self.data[:valid_num] if train == "valid" else self.data[valid_num:]
        else:
            assert 0

    def data_check(self):
        for x in self.data:
            name = x[0]
            if not os.path.exists(os.path.join(self.path, "image", name + ".tiff")):
                print("image:{} missing".format(name))
                assert 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        random.seed(time.time())
        x = self.data[x]
        img = cv2.imread(os.path.join(self.path, "image", x[0]))
        img = img.reshape(8, 256, 8, 256, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 256, 256, 3)
        idx = [i for i in range(64)]
        random.shuffle(idx)
        img = img[idx[:self.valid_blocks]]
        return img, x[2]


def build_data(path: str, batch_size, train: str, num_worker, valid_num=200, valid_block=32):
    return DataLoader(PANDA_dataset(path, train, valid_block, valid_num, valid_block), batch_size, shuffle=True,
                      num_workers=num_worker)


if __name__ == "__main__":
    d = PANDA_dataset("./", "train")
    print(random.randint(1, 10))
    print(d.__getitem__(4))
