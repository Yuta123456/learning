import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image

class EmbeddingDataset(Dataset):
    def __init__(self, annotations_file):
        # labels.csv => filepath, label の形式になっている
        # ここをfilepath, textにすればいい？
        self.annotations = pd.read_csv(annotations_file)
        self.transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('L')
            image = Image.merge('RGB', [image] * 3)
        input_image = self.transform(image)
        # lstmに入れる形にする
        caption = self.annotations.iloc[idx, 1]
        return input_image, caption
    