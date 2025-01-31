from guided_diffusion import dist_util
from guided_diffusion.script_util import NUM_CLASSES, create_model_and_diffusion, add_dict_to_argparser, args_to_dict
from torchvision import transforms
import argparse
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_to_256= transforms.Compose([
   transforms.Resize((256, 256)),])

trans_to_224= transforms.Compose([
   transforms.Resize((224, 224)),])


steps =  1000
start = 0.0001
end = 0.02
trial_num = 2
shape = (1, 3, 256, 256)


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=4,
        use_ddim=False,
        model_path="./guided_diffusion/models/256x256_diffusion_uncond.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args_diff = create_argparser().parse_args([])

d_model, diffusion = create_model_and_diffusion(
    **args_to_dict(args_diff, model_and_diffusion_defaults().keys())
)
d_model.load_state_dict(
    dist_util.load_state_dict(args_diff.model_path, map_location="cpu")
)
d_model.to(dist_util.dev())
if args_diff.use_fp16:
    d_model.convert_to_fp16()

d_model.eval()

def get_opt_t(delta, start, end, steps):
    return np.clip(int(np.around(1+(steps-1)/(end-start)*(1-1/(1+delta**2)-start))), 0, steps)

def add_noise(x, delta, opt_t, steps, start, end):
    return np.sqrt(1-beta(opt_t, steps, start, end))*(x + torch.randn_like(x) * delta)

def beta(t, steps, start, end):
    return (t-1)/(steps-1)*(end-start)+start

def denoise_inner(img, opt_t, steps, start, end, delta, diffusion, d_model, direct_pred=False):
    img_xt = add_noise(img, delta, opt_t, steps, start, end).unsqueeze(0).to(device)

    indices = list(range(opt_t))[::-1]
    img_iter = img_xt
    for i in indices:
        t = torch.tensor([i]*shape[0], device=device)
        with torch.no_grad():
            out = diffusion.p_sample(
                d_model,
                img_iter,
                t,
                clip_denoised=args_diff.clip_denoised,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs={},
            )
            img_iter = out['sample']
            if direct_pred:
                return out['pred_xstart']
            
    return img_iter



def get_diffusion_models():
    return diffusion, d_model

def denoise(image, noise_level, diffusion, d_model):
    opt_t = get_opt_t(noise_level, start, end, steps)
    # denoise(trans_to_256(image.cuda()).squeeze(0), opt_t, steps, start, end, noise_level)
    image_denoised = denoise_inner(
        trans_to_256(image.cuda()).squeeze(0), opt_t, steps, 
        start, end, noise_level,diffusion, d_model
        )
    
    return trans_to_224(image_denoised)