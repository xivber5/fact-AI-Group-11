import os
import torch
import torchattacks
from time import time
from torch.utils.data import DataLoader
from baselines.ViT.misc_functions import *
from ImageNetVal.dataloader import ImageNet50k #        
from baselines.ViT.ViT_new import vit_base_patch16_224
from diffusion_denoise import get_diffusion_models, denoise
from baselines_mod.ViT.explanation_gen_batched import Baselines, LRP #
from baselines_mod.ViT.ViT_LRP_batched import vit_base_patch16_224 as vit_LRP #
from baselines_mod.ViT.ViT_orig_LRP_batched import vit_base_patch16_224 as vit_orig_LRP #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_base_patch16_224(pretrained=True).to(device)
baselines = Baselines(model)

model_LRP = vit_LRP(pretrained=True).to(device)
model_LRP.eval()
lrp = LRP(model_LRP)

model_orig_LRP = vit_orig_LRP(pretrained=True).to(device)
model_orig_LRP.eval()
orig_lrp = LRP(model_orig_LRP)

diffusion, d_model = get_diffusion_models()
d_model.eval()

print('loaded models.')

def attack(model, images, labels, noise_level, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    # torch.backends.cudnn.deterministic = True
    atk = torchattacks.PGD(model, eps=noise_level, alpha=noise_level/5, steps=10)
    atk.set_normalization_used(mean, std)
    adv_images = atk(images, labels)
    return adv_images


data = ImageNet50k(subset=4000, seed=1)
dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=9)
atk_radii = torch.linspace(0,16,7) / 255

save_dir = 'ImageNetVal/denoised_copy/'

for i,atk in enumerate(atk_radii):
    atk = atk.item()
    os.makedirs(f'{save_dir}{i}/', exist_ok=True)


    for j, (image, target) in enumerate(dataloader):

        image = image.to(device)
        target = target.to(device)
        corrupted = attack(model, image, target, atk)
        denoised = denoise(corrupted, atk, diffusion, d_model)
        torch.save(denoised, f'{save_dir}{i}/img{j}.pt')

        if i == 0:
            with open(f'{save_dir}targets.txt', 'a') as f:
                f.write(f'{target.item()}\n')

        # i forgot tot add random noise scaled iwht noise level and clamping
        # this must be done when loading the denoised image tensors

        del corrupted, denoised
        torch.cuda.empty_cache()