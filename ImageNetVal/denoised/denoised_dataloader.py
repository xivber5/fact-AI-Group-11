import torch
from torch.utils.data import Dataset

class ImageNetDenoised(Dataset):
    def __init__(self, atk, path="ImageNetVal/denoised/"):
        if atk not in range(0,7):
            raise Exception
        
        self.atk = atk # this is an index
        self.path = path
        atk_radii = torch.linspace(0,16,7) / 255
        self.atk_rad = atk_radii[self.atk] # this is the actual attack radius

        self.image_dir = f"{self.path}{self.atk}/"
        self.targets = open(f"{self.path}targets.txt").read().splitlines()

    # i forgot to do this in denoise_imagenet.py, so it's here
    @staticmethod
    def f(image):
        image = image + torch.randn_like(image) * image
        image = torch.squeeze(image)
        image = torch.clamp(image, -1, 1)
        return image

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        target = torch.tensor(int(self.targets[idx]))
        image = torch.load(f"{self.path}{self.atk}/img{idx}.pt", 
                           map_location=torch.device("cpu"))
        
        return self.f(image), target
    

if __name__ == "__main__":

    data = ImageNetDenoised(6)
    img, target = data[4]
    print("num images:", len(data))
    print("image shape:", img.shape, "target:", target)