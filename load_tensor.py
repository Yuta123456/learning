import torch
import importlib
from FashionClipDataset import FashionClipDataset


# importlib.reload(FashionClipDataset)
importlib.invalidate_caches()
loaded_tensor = torch.load('image_tensor/tensor_0-100000.pt')

dataset = FashionClipDataset('./data/anotation_new.csv', 'image_tensor/tensor_0-100000.pt')
for i, value in enumerate(dataset):
    print(i,  value)
    if (i >= 2 ):
        break