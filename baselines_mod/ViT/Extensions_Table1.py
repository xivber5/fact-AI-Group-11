import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
from tqdm import tqdm
from utilities.metrices import *

from utilities import render
from utilities.saver import Saver
from utilities.iou import IoU

from data.Imagenet import Imagenet_Segmentation

import cv2
import numpy as np

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from baselines_mod.ViT.ViT_pytorch_cam import vit_base_patch16_224 as vitmodel


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F



def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_new_explanation_method(model, input_img, method="ablationcam", targets=None):

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    target_layers = [model.blocks[-1].norm1]

    if method == 'ablationcam':
        cam = methods[method](model=model,
                target_layers=target_layers,
                reshape_transform=reshape_transform,
                ablation_layer=AblationLayerVit())
    else:
        cam = methods[method](model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform)

 
    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_img,
                    targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam

def attack(image, model, noise_level,label_index=None,mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]):
    import torchattacks
    torch.backends.cudnn.deterministic = True
    atk = torchattacks.PGD(model, eps=noise_level, alpha=noise_level/5, steps=10)
    atk.set_normalization_used(mean, std)
    labels = torch.FloatTensor([0]*1000)
    if label_index == None:
        # with torch.no_grad():
        logits = model(image)
        label_index = logits.argmax()
        # print(label_index)

    labels[label_index] = 1
    labels = labels.reshape(1, 1000)
    adv_images = atk(image, labels.float())
    return adv_images
    