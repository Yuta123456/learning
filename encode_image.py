#sys.path.append("fashion-clip/")
from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import torch
from contextlib import redirect_stdout
from torch.utils.data import DataLoader

import os
from contextlib import redirect_stdout
def embedding_image(path):
    enb = fclip.encode_images(path, batch_size=32)
    enb = torch.from_numpy(enb)
    # image_embeddings = image_embeddings.to(device)
    return path, enb

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPUデバイスを取得
else:
    device = torch.device("cpu")  # CPUデバイスを取得

fclip = FashionCLIP('fashion-clip')

class FDataset(Dataset):
    def __init__(self, annotations_file, start, end):
        self.data = pd.read_csv(annotations_file)[start:end]   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        return img_path

start = 0
end = 100000
dataset = FDataset('./data/anotation_new.csv', start, end)
dataloader =  DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
image_embeddings = torch.Tensor([])
for i, (path) in enumerate(dataloader):
    # 予測と損失の計算
    _, image_embedding = embedding_image(path)
    image_embeddings = torch.cat((image_embeddings, image_embedding), dim=0)
    if i % 100 == 0:
        print(f"finish: {i * 100/len(dataloader)}%")

# image_embeddings.save('tensor.pt')
torch.save(image_embeddings, f'image_tensor/tensor_{start}-{end}.pt')