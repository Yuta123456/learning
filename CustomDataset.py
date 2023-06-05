import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image

class EmbeddingDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)
        self.positive_pair = self.annotations.copy()
        self.positive_pair['new_column'] = 1
        self.negative_pair = self.annotations.copy()
        shuffle_caption = np.random.permutation(self.annotations.iloc[:, 1].values)
        self.negative_pair.iloc[:, 1] = shuffle_caption
        self.negative_pair['new_column'] = 0
        self.data = pd.concat([self.positive_pair, self.negative_pair], axis=0)
        print(self.data.shape)
        self.transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('L')
            image = Image.merge('RGB', [image] * 3)
        input_image = self.transform(image)
        # lstmに入れる形にする
        caption = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        return input_image, caption, label
    