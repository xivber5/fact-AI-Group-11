import argparse
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        device = next(self.model.parameters()).device
        output = self.model(input)
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, 1] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)
        
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        if method in ['grad', 'transformer_attribution']:
            #$ onehot tensor instead of single vector
            onehotTensor = F.one_hot(index, num_classes=output.size()[-1])
            #$ pass tensor to relprop
            relprop = self.model.relprop(onehotTensor.to(input.device), method=method, 
                                        is_ablation=is_ablation, start_layer=start_layer, **kwargs)
        else:
            onehotTensor = F.one_hot(index, num_classes=output.size()[-1])
            relprop = self.model.relprop(onehotTensor.to(input.device), method=method, 
                                         is_ablation=is_ablation, start_layer=start_layer, **kwargs)
        
        return relprop



class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        device = next(self.model.parameters()).device
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0,1] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        #$ grad and cam for entire batch
        gradB = self.model.blocks[-1].attn.get_attn_gradients()
        camB = self.model.blocks[-1].attn.get_attention_map()
        #$ results list
        batch_cam = []
        for b in range(camB.shape[0]):
            cam = camB[b, :, 0, 1:].reshape(-1, 14, 14)
            grad = gradB[b, :, 0, 1:].reshape(-1, 14, 14)
            grad = grad.mean(dim=[1, 2], keepdim=True)
            cam = (cam * grad).mean(0).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min()) #$ this normalizes twice
            #$ dim to stack along
            batch_cam.append(cam.unsqueeze(0))

        #$ return stacked tensor
        stacked_cam = torch.stack(batch_cam, dim=0)
        return stacked_cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]
