from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
import torch.nn.functional as F

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from model.cond_fn import MSEGuidance
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.metrics import calculate_psnr_pt, LPIPS
from pytorch_msssim import ssim, ms_ssim
from torchvision import transforms
import logging
from ldm.util import log_txt_as_img


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    cond_fn: Optional[MSEGuidance],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.

    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]).
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    img_t = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    img_t = einops.rearrange(img_t, "n h w c -> n c h w").contiguous()

    img_init, cond_snr, _ = model.preprocess_model(img_t)
    model.control_scales = [strength] * 13

    if cond_fn is not None:
        cond_fn.load_target(2 * img_init - 1)

    height, width = img_t.size(-2), img_t.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)

    input_img = []
    for b in range(img_t.shape[0]):
        input_img.append(model.transform_to_pil(img_init[b]))

    inputs = model.processor(images=input_img, return_tensors="pt", max_length=32).to(img_t.device, torch.float16)
    generated_ids = model.blip_model.generate(**inputs)
    generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=True)

    cond_text = model.get_learned_conditioning(generated_text)
    text_img = (log_txt_as_img((512, 512), generated_text, size=16) + 1) / 2

    x_T = None #torch.randn(shape, device=model.device, dtype=torch.float32)
    samples = sampler.sample(
        steps=steps, shape=shape, cond_img=img_init, cond_snr=cond_snr, cond_text=cond_text,
        positive_prompt="", negative_prompt="", x_T=x_T,
        cfg_scale=1.0, cond_fn=cond_fn, color_fix_type=color_fix_type
    )
    x_samples = samples.clamp(0, 1)

    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    img_init = (einops.rearrange(img_init, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]
    jscc_preds = [img_init[i] for i in range(n_samples)]
    
    if text_img is not None:
        text_img = (einops.rearrange(text_img, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    return preds, jscc_preds, text_img


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # TODO: add help info for these options
    parser.add_argument("--ckpt", required=True, type=str, help="full checkpoint path")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")
    parser.add_argument("--SNR", type=float, default=1)

    # latent image guidance
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--Lambda", type=float, default=0.0)
    parser.add_argument("--g_t_start", type=int, default=1001)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)

    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])

    return parser.parse_args()


def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                  "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                          "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                          "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    args.device = check_device(args.device)

    # Set up LPIPS metric
    lpips_metric = LPIPS(net="alex").to(args.device)
    
    # Load the DiffJSCC model
    model = ControlLDM.from_pretrained(args.ckpt)
    model.eval()
    model.to(args.device)
    
    # Set up the channel SNR
    model.preprocess_model.SNR_low = args.SNR
    model.preprocess_model.SNR_high = args.SNR

    assert os.path.isdir(args.input)
    convert_tensor = transforms.ToTensor()

    PSNR, MSSSIM, Lpips = [], [], []
    PSNR_jscc, MSSSIM_jscc, Lpips_jscc  = [], [], []

    for i, file_path in enumerate(list_image_files(args.input, follow_links=True)):
        
        # Read, resize, and pad the input image
        lq = Image.open(file_path).convert("RGB")
        lq_resized = auto_resize(lq, 512)
        x = pad(np.array(lq_resized), scale=64)

        lq_list, pred_list, pred_jscc_list = [], [], []

        for j in range(args.repeat_times):

            save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path = os.path.join(parent_path, f"{stem}_{j}.png")
            os.makedirs(parent_path, exist_ok=True)

            # initialize latent image guidance
            if args.use_guidance:
                cond_fn = MSEGuidance(
                    scale=args.Lambda, t_start=args.g_t_start, t_stop=args.g_t_stop,
                    space=args.g_space, repeat=args.g_repeat
                )
            else:
                cond_fn = None

            preds, jscc_preds, text_img = process(
                model, [x], steps=args.steps,
                strength=1,
                color_fix_type=args.color_fix_type,
                cond_fn=cond_fn,
            )

            pred, jscc_pred  = preds[0], jscc_preds[0]

            # remove padding
            pred = pred[:lq_resized.height, :lq_resized.width, :]
            jscc_pred = jscc_pred[:lq_resized.height, :lq_resized.width, :]
            
            # resize 
            pred_np = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
            jscc_pred_np = np.array(Image.fromarray(jscc_pred).resize(lq.size, Image.LANCZOS))
            lq_np = np.array(lq)

            pred_tensor = convert_tensor(pred_np).unsqueeze(0)
            jscc_pred_tensor = convert_tensor(jscc_pred_np).unsqueeze(0)
            lq_tensor = convert_tensor(lq_np).unsqueeze(0)

            lq_list.append(lq_tensor)
            pred_list.append(pred_tensor)
            pred_jscc_list.append(jscc_pred_tensor)

            if args.show_lq:
                images = [lq_np, pred_np] if args.disable_preprocess_model else [lq_np, jscc_pred_np, pred_np]
                Image.fromarray(np.concatenate(images, axis=1)).save(save_path)
            else:
                Image.fromarray(pred_np).save(save_path)
            
            if text_img is not None:
                folder = f'{args.output}_text'
                os.makedirs(folder, exist_ok=True)
                text_path = f'{folder}/{stem}_{j}.png'
                Image.fromarray(text_img[0]).save(text_path)
            

        lq_tensor = torch.cat(lq_list, 0)
        pred_tensor = torch.cat(pred_list, 0)
        jscc_pred_tensor = torch.cat(pred_jscc_list, 0)

        lpips_jscc = lpips_metric(jscc_pred_tensor.cuda(), lq_tensor.cuda(), normalize=True).mean()
        lpips = lpips_metric(pred_tensor.cuda(), lq_tensor.cuda(), normalize=True).mean()

        psnr_jscc = calculate_psnr_pt(jscc_pred_tensor, lq_tensor, crop_border=0).mean()
        psnr = calculate_psnr_pt(pred_tensor, lq_tensor, crop_border=0).mean()

        msssim_jscc = ms_ssim(jscc_pred_tensor, lq_tensor, data_range=1).mean()
        msssim = ms_ssim(pred_tensor, lq_tensor, data_range=1).mean()

        PSNR.append(psnr.item())
        MSSSIM.append(msssim.item())
        Lpips.append(lpips.item())

        PSNR_jscc.append(psnr_jscc.item())
        MSSSIM_jscc.append(msssim_jscc.item())
        Lpips_jscc.append(lpips_jscc.item())

        print(f"save to {save_path}")
        print(f'PSNR: {psnr_jscc:.4f}->{psnr:.4f}')
        print(f'MSSSIM: {msssim_jscc:.4f}->{msssim:.4f}')
        print(f'LPIPS: {lpips_jscc:.4f}->{lpips:.4f}')

    print(f'PSNR:  prev: {np.mean(PSNR_jscc):.4f}, after: {np.mean(PSNR):.4f}')
    print(f'MSSSIM:  prev: {np.mean(MSSSIM_jscc):.4f}, after: {np.mean(MSSSIM):.4f}')
    print(f'LPIPS:  prev: {np.mean(Lpips_jscc):.4f}, after: {np.mean(Lpips):.4f}')


if __name__ == "__main__":
    main()
