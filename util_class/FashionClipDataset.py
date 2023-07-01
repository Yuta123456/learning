import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class FashionClipDataset(Dataset):
    def __init__(self, annotations_file, tensor_file):
        # TODO: tensor_fileの参照からマジックナンバー除去
        self.annotations = pd.read_csv(annotations_file)[0:100000]
        self.img_tensors = torch.load(tensor_file)
        # if (self.img_tensors.shape != (100000, 512)):
        #     print("not match size")
        #     exit()
        self.positive_pair = self.annotations.copy()
        self.positive_pair['new_column'] = 1
        self.negative_pair = self.annotations.copy()
        self.negative_pair.iloc[:, 0] = np.random.permutation(self.negative_pair)
        self.negative_pair['new_column'] = 0
        self.data = pd.concat([self.positive_pair, self.negative_pair], axis=0)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_tensor = self.img_tensors[idx % 100000]
        caption = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        return img_tensor, caption, label