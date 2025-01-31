import os
import tarfile
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNet50k(Dataset):
    def __init__(self, subset=False, seed=None):

        tarpath = "ImageNetVal/ILSVRC2012_img_val.tar"
        extractpath = "ImageNetVal/images"
        with tarfile.open(tarpath, 'r') as tar:
            if not os.path.exists(extractpath):
                tar.extractall(path=extractpath)

        self.data = []
        with open('ImageNetVal/file_to_label.txt') as f:
            for line in f:
                filename, label = line.split()
                self.data.append((filename, int(label)))

        if seed:
            random.seed(seed)

        if subset:
            self.data = random.sample(self.data, subset)

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, label = self.data[idx]
        label = torch.tensor(label)

        fpath = f'ImageNetVal/images/{filename}'
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        return img, label
        
if __name__ == "__main__":
    imgnet = ImageNet50k()
    print(f'images in dataset: {len(imgnet)}')
