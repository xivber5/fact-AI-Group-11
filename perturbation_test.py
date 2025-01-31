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


def compute_saliency(image, target, method, ablation=False):
    if method == "rollout": #rollout
        res = baselines.generate_rollout(image).reshape(image.shape[0], 1, 14, 14)
    elif method == "lrp": #lrp
        res = lrp.generate_LRP(image, start_layer=1, index=target).reshape(image.shape[0], 1, 14, 14)
    elif method == "transformer_attribution": #vta
        res = lrp.generate_LRP(image, start_layer=1, method="grad", index=target).reshape(image.shape[0], 1, 14, 14)
    elif method == "full_lrp":
        res = orig_lrp.generate_LRP(image, method="full", index=target).reshape(image.shape[0], 1, 224, 224)
    elif method == "lrp_last_layer":
        res = orig_lrp.generate_LRP(image, method="last_layer", is_ablation=ablation, index=target).reshape(image.shape[0], 1, 14, 14)
    elif method == "attn_last_layer": #raw attention
        res = lrp.generate_LRP(image, method="last_layer_attn", is_ablation=ablation, index=target).reshape(image.shape[0], 1, 14, 14)
    elif method == "attn_gradcam": #gradcam
        res = baselines.generate_cam_attn(image, index=target).reshape(image.shape[0], 1, 14, 14)
    elif method == "ours":
        res = lrp.generate_LRP(image, start_layer=1, method="transformer_attribution", index=target).reshape(image.shape[0], 1, 14, 14)

    if method not in ["full_lrp", "input_grads"]:
        res = torch.nn.functional.interpolate(res, scale_factor=16, mode="bilinear")

    # Normalize saliency map
    res = (res - res.min()) / (res.max() - res.min())

    return res


def mask_images(images, salience, n_masks, postive=True):
    batch_size, *dims = salience.shape
    numel = torch.tensor(dims).prod().item()
    flat = salience.view(batch_size, -1)
    #n_mask = (numel * torch.arange(0.1, 1.0, 0.1).to(salience.device)).int()
    pixels_to_mask = (numel * torch.linspace(0.1, 0.9, n_masks).to(salience.device)).int()

    mask_stack = []
    for p in pixels_to_mask:
        k_idx = torch.topk(flat, p.item(), dim=1, largest=postive).indices
        mask = torch.ones_like(flat, dtype=torch.bool)
        row_indices = torch.arange(flat.size(0)).unsqueeze(1)
        mask[row_indices, k_idx] = False
        mask_stack.append(mask.unsqueeze(1))
    
    mask_stack = torch.stack(mask_stack, dim=1)
    mask_stack = mask_stack.view(batch_size, mask_stack.shape[1], *dims)

    return images.unsqueeze(1) * mask_stack


def classify(model, images):
    with torch.no_grad():
        logits = model(images)
    logits = torch.softmax(logits, dim=1)
    logits = torch.argmax(logits, dim=1)
    return logits


def memcheck():
    print(f'{"total:":<15}{torch.cuda.get_device_properties(0).total_memory}')
    print(f'{"reserved:":<15}{torch.cuda.memory_reserved(0)}')
    print(f'{"alloc:":<15}{torch.cuda.memory_allocated(0)}\n')


def attack(model, images, labels, noise_level, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    # torch.backends.cudnn.deterministic = True
    atk = torchattacks.PGD(model, eps=noise_level, alpha=noise_level/5, steps=10)
    atk.set_normalization_used(mean, std)
    adv_images = atk(images, labels)
    return adv_images


def pertu_test_cycle(model, subset, method, n_radii, n_masks, positive_pertu=True, batch_size=16, model_throughput=30, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = torch.zeros(n_radii, n_masks).to(device)
    atk_radii = torch.linspace(0,16,n_radii) / 255
    # - 32

    data = ImageNet50k(subset=subset, seed=1)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=9)
    print('loaded data.')

    for R, atk_radius in enumerate(atk_radii):
        atk_radius = atk_radius.item()
        correct = torch.zeros(n_masks).to(device) # correct prediction for [10-90%] removed pixels

        for i, (image, target) in enumerate(dataloader):
            if i % 100 == 0:
                print(f'[{method}] ATK:{R+1}/{len(atk_radii)}; BATCH:{i}/{len(dataloader)}')
            if i == len(dataloader)-1:
                print(f'[{method}] ATK:{R+1}/{len(atk_radii)}; BATCH:{i+1}/{len(dataloader)}')
            
            image = image.to(device)
            target = target.to(device)

            # PGD attack on images
            corrupted = attack(model, image, target, atk_radius)
            #image = attack(model, image, target, atk_radius)

            # compute saliency and mask images
            saliency = compute_saliency(corrupted, target, method) # create saliency maps of (corrupted) image
            #saliency = compute_saliency(image, target, method)
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
                if debug:
                    memcheck()

            # free memory
            del image, target, saliency, masked, pertur, logits, M, T, P, corrupted
            torch.cuda.empty_cache()

        print(correct / subset, f'\n')
        results[R] = correct

    del data, dataloader   
    return results / subset


def timer(func, *args):
    start = time()
    re = func(*args)
    end = time()
    return end-start, re


def save_results(results, subset, method, n_rad, n_mask, positive_pertu, time_elapsed):
    mt_to_name = {"rollout":"rollout", "lrp":"lrp", "transformer_attribution":"vta", "attn_last_layer":"raw_attn", "attn_gradcam":"gradcam"}

    mth = mt_to_name[method]
    per = "pos" if positive_pertu else "neg"
    save_dir = f'perturbation_results/{n_rad}R{n_mask}M_IMGNET{subset}/{per}/'

    os.makedirs(save_dir, exist_ok=True)
    torch.save(results, f'{save_dir}/{mth}.pt')

    with open(f'{save_dir}/{mth}_META.txt', 'w') as meta:
        meta.write(f"{'method:':16} {mth} ({method})\n")
        meta.write(f"{'dataset size:':16} {subset} images\n")
        meta.write(f"{'positive:':16} {positive_pertu}\n")
        meta.write(f"{'n_radii:':16} {n_rad} = linspace(0, 16, {n_rad}) / 255\n")
        meta.write(f"{'n_masks:':16} {n_mask} = linspace(0.1, 0.9, {n_mask})\n")
        meta.write(f"{'time elapsed:':16} {time_elapsed:.2f} seconds\n")
        meta.write(f"{'tensor format:':16} [atk_rad, masks]\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["rollout", "lrp", "vta", "raw_attn", "gradcam"],required=True)
    parser.add_argument('--mode', choices=['pos', 'neg'], required=True)
    parser.add_argument('--subset', type=int, default=10000)
    parser.add_argument('--radii', type=int, default=9)
    parser.add_argument('--masks', type=int, default=9)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--modeltp', type=int, default=32)
    parser.add_argument('--debug', action='store_true')

    arg_to_mt = {"rollout":"rollout", "lrp":"lrp", "vta":"transformer_attribution", "raw_attn":"attn_last_layer", "gradcam":"attn_gradcam"}

    args = parser.parse_args()
    MODEL = model
    SUBSET = args.subset
    METHOD = arg_to_mt[args.method]
    N_RADII = args.radii
    N_MASKS = args.masks
    POSITIVE_PERTU = True if args.mode == 'pos' else False
    BATCHSIZE = args.batchsize
    MODEL_THROUGHPUT = args.modeltp
    DEBUG = args.debug
    func_args = (MODEL, SUBSET, METHOD, N_RADII, N_MASKS, POSITIVE_PERTU,  
                 BATCHSIZE, MODEL_THROUGHPUT, DEBUG)

    print(f'[{METHOD}: {POSITIVE_PERTU}] radii {N_RADII}; masks {N_MASKS}')
    print(f'images {SUBSET}; batch size {BATCHSIZE}; model tp {MODEL_THROUGHPUT}')

    time_el, results = timer(pertu_test_cycle, *func_args)
    save_results(results, SUBSET, METHOD, N_RADII, N_MASKS, POSITIVE_PERTU, time_el)

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
    print('done')