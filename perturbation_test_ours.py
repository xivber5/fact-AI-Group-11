import os
import torch
import argparse
import numpy as np
import torchattacks
from time import time
import matplotlib.pyplot as plt
from samples.CLS2IDX import CLS2IDX
from torch.utils.data import DataLoader
from baselines.ViT.misc_functions import *
from ImageNetVal.dataloader import ImageNet50k # 
from baselines.ViT.ViT_new import vit_base_patch16_224
from perturbation_test import compute_saliency,mask_images,classify
from baselines_mod.ViT.explanation_gen_batched import Baselines,LRP #
from ImageNetVal.denoised.denoised_dataloader import ImageNetDenoised
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


def pertu_test_cycle(model, n_masks, positive_pertu=True, batch_size=16, model_throughput=30):
    N_RADII = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = torch.zeros(N_RADII, n_masks).to(device)
    clean_data = ImageNet50k(subset=4000, seed=1)
    clean_loader = DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=9)

    for R in range(N_RADII):
        correct = torch.zeros(n_masks).to(device)
        # attacks and denoising are done beforehand -> denoise_imagenet.py
        denoised_data = ImageNetDenoised(atk=R)
        denoised_loader = DataLoader(denoised_data, batch_size=batch_size, shuffle=False, num_workers=9)
        denoised_loader = iter(denoised_loader)

        for i, (image,target) in enumerate(clean_loader):
            denoised, target_ = next(denoised_loader)
            if not torch.all(target == target_):
                raise Exception('target mismatch in clean/denoise')
            
            if i % 100 == 0:
                print(f'[ours] ATK:{R+1}/{N_RADII}; BATCH:{i}/{len(clean_loader)}')
            if i == len(clean_loader)-1:
                print(f'[ours] ATK:{R+1}/{N_RADII}; BATCH:{i+1}/{len(clean_loader)}')

            image = image.to(device)
            denoised = denoised.to(device)
            target = target.to(device)

            # compute saliency and mask images
            saliency = compute_saliency(denoised, target, "ours")
            masked = mask_images(image, saliency, n_masks, postive=positive_pertu) # mask healthy images

            # masked has shape [16, 9, 3, 224, 224], but models expects [X, 3, 224, 224]
            current_batch_size, _, *dims = masked.shape
            masked = masked.view(-1, *dims)
            # reshape target vector to match this format
            target = target.unsqueeze(1).repeat(1, n_masks).view(-1)
            # additional tensor to keep track of pertur%
            pertur = torch.arange(n_masks).repeat(current_batch_size, 1).view(-1).to(device)

            # split in chunks for classifcation -> MODEL_THROUGHPUT
            n_imgs = target.shape[0]
            chunks = [model_throughput for _ in range(n_imgs // model_throughput)]
            if sum(chunks) < n_imgs:
                chunks.append(n_imgs % model_throughput)

            masked = torch.split(masked, chunks, dim=0)
            target = torch.split(target, chunks, dim=0)
            pertur = torch.split(pertur, chunks, dim=0)

            # predictions
            for M,T,P in zip(masked, target, pertur):
                logits = classify(model, M)
                for pert in range(n_masks):
                    # logits==T     correct predictions
                    # P==pert       only indices with e.g. 30% perturbation
                    correct[pert] += ((logits==T) & (P==pert)).sum().item()
            
            
            del image, target, saliency, masked, denoised, pertur, logits, M, T, P
            torch.cuda.empty_cache()

        print(correct / 4000, f'\n')
        results[R] = correct
        del denoised_data, denoised_loader 

    del clean_data, clean_loader   
    return results / 4000


def timer(func, *args):
    start = time()
    re = func(*args)
    end = time()
    return end-start, re


def save_results(results, n_mask, positive_pertu, time_elapsed):
    mth = 'ours'
    per = "pos" if positive_pertu else "neg"
    save_dir = f'perturbation_results/7R{n_mask}M_IMGNET4000/{per}/'

    os.makedirs(save_dir, exist_ok=True)
    torch.save(results, f'{save_dir}/{mth}.pt')

    with open(f'{save_dir}/{mth}_META.txt', 'w') as meta:
        meta.write(f"{'method:':16} {mth} (ours)\n")
        meta.write(f"{'dataset size:':16} 4000 images\n")
        meta.write(f"{'positive:':16} {positive_pertu}\n")
        meta.write(f"{'n_radii:':16} 7 = linspace(0, 16, 7) / 255\n")
        meta.write(f"{'n_masks:':16} {n_mask} = linspace(0.1, 0.9, {n_mask})\n")
        meta.write(f"{'time elapsed:':16} {time_elapsed:.2f} seconds (without attacking/denoising)\n")
        meta.write(f"{'tensor format:':16} [atk_rad, masks]\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pos', 'neg'], required=True)
    parser.add_argument('--masks', type=int, default=9)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--modeltp', type=int, default=32)

    args = parser.parse_args()
    MODEL = model
    METHOD = "ours"
    N_MASKS = args.masks
    POSITIVE_PERTU = True if args.mode == 'pos' else False
    BATCHSIZE = args.batchsize
    MODEL_THROUGHPUT = args.modeltp
    func_args = (MODEL, N_MASKS, POSITIVE_PERTU, BATCHSIZE, MODEL_THROUGHPUT)

    print(f'[{METHOD}: {POSITIVE_PERTU}] radii 7; masks {N_MASKS}')
    print(f'images 4000; batch size {BATCHSIZE}; model tp {MODEL_THROUGHPUT}')

    time_el, results = timer(pertu_test_cycle, *func_args)
    save_results(results, N_MASKS, POSITIVE_PERTU, time_el)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
    print('done')