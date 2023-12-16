import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image


class CompatibilityDataset(Dataset):
    def __init__(self, positive_file, negative_file, n=10000):
        # TODO: ? サンプリング
        self.positive_annotations = pd.read_csv(
            positive_file, header=None, encoding="utf-8"
        ).sample(n=n, random_state=42)

        self.negative_annotations = pd.read_csv(
            negative_file, header=None, encoding="utf-8"
        ).sample(n=n * 32, random_state=42)

        self.data = pd.concat(
            [self.positive_annotations, self.negative_annotations], axis=0
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data.iloc[idx, 0], self.data.iloc[idx, 2])

        image_1 = self.to_image(self.data.iloc[idx, 0])
        image_2 = self.to_image(self.data.iloc[idx, 2])

        # lstmに入れる形にする
        caption_1 = self.data.iloc[idx, 1]
        caption_2 = self.data.iloc[idx, 3]

        label = self.data.iloc[idx, 4]
        return image_1, caption_1, image_2, caption_2, label

    def to_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("L")
            image = Image.merge("RGB", [image] * 3)
        image = self.transform(image)
        return image
