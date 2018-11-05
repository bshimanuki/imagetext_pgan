import os

from torch.utils.data.dataset import Dataset

from make_image import convert

class TextDataset(Dataset):
    def __init__(self, path, size=None, transform=None):
        self.texts = open(path).read().strip().split('\n')
        self.transform = transform
        self.size = len(self.texts)
        if size is not None:
            self.size = min(self.size, size)

    def __getitem__(self, index):
        img = convert(self.texts[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.size
