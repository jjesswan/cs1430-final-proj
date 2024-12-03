import zipfile
import os, sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ResShift.sampler import ResShiftSampler
from ResShift.basicsr.utils.download_util import load_file_from_url
from downsample import downsample_image_from_zip, image_extensions

_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    'deblur': 4,
}
_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
    'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
    'deblur': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_deblur_s4.pth',
}


def corrupt_in_batches_from_zip(source_zipped_image_path, corrupted_image_path, num_samples, scale=0.25, noise_prob=0, is_blur=False, blur_ksize=5, artifact_prob=0, artifact_quality=80):
    with zipfile.ZipFile(source_zipped_image_path, 'r') as zip_ref:
        for file_name in tqdm(zip_ref.namelist()[:num_samples]):
            if file_name.endswith(tuple(image_extensions)):
                downsample_image_from_zip(
                    zip_ref, 
                    file_name, 
                    output_dir=corrupted_image_path, 
                    scale=scale, 
                    noise_prob=noise_prob, 
                    is_blur=is_blur, 
                    blur_ksize=blur_ksize, 
                    artifact_prob=artifact_prob, 
                    artifact_quality=artifact_quality
                )


def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.task == 'realsr':
        if args.version in ['v1', 'v2']:
            configs = OmegaConf.load('./ResShift/configs/realsr_swinunet_realesrgan256.yaml')
        elif args.version == 'v3':
            configs = OmegaConf.load('./ResShift/configs/realsr_swinunet_realesrgan256_journal.yaml')
        else:
            raise ValueError(f"Unexpected version type: {args.version}")
        assert args.upscale == 4, 'We only support the 4x super-resolution now!'
        ckpt_url = _LINK[args.version]
        ckpt_path = ckpt_dir / f'resshift_{args.task}x{args.upscale}_s{_STEP[args.version]}_{args.version}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif args.task == 'bicsr':
        configs = OmegaConf.load('./ResShift/configs/bicx4_swinunet_lpips.yaml')
        assert args.upscale == 4, 'We only support the 4x super-resolution now!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}x{args.upscale}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif args.task == 'inpaint_imagenet':
        configs = OmegaConf.load('./ResShift/configs/inpaint_lama256_imagenet.yaml')
        assert args.upscale == 1, 'Please set scale equals 1 for image inpainting!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif args.task == 'inpaint_face':
        configs = OmegaConf.load('./ResShift/configs/inpaint_lama256_face.yaml')
        assert args.upscale == 1, 'Please set scale equals 1 for image inpainting!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan_face256']
        vqgan_path = ckpt_dir / f'celeba256_vq_f4_dim3_face.pth'
    elif args.task == 'faceir':
        configs = OmegaConf.load('./ResShift/configs/faceir_gfpgan512_lpips.yaml')
        assert args.upscale == 1, 'Please set scale equals 1 for face restoration!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan_face512']
        vqgan_path = ckpt_dir / f'ffhq512_vq_f8_dim8_face.pth'
    elif args.task == 'deblur':
        configs = OmegaConf.load('./ResShift/configs/deblur_gopro256.yaml')
        assert args.upscale == 1, 'Please set scale equals 1 for deblurring!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    # prepare the checkpoint
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.upscale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.reconstructed_image_path).exists():
        Path(args.reconstructed_image_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.upscale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.upscale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.upscale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.upscale)
    args.chop_size *= (4 // args.upscale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride



def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--source_zipped_image_path", type=str, default="", help="Input Zipped Image path.")
    parser.add_argument("-c", "--corrupted_image_path", type=str, default="", help="Corrupted Image path.")
    parser.add_argument("-o", "--reconstructed_image_path", type=str, default="", help="Output Reconstructed Image path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--upscale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="v1",
            choices=["v1", "v2", "v3"],
            help="Checkpoint version.",
            )
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="realsr",
            choices=['realsr', 'bicsr', 'inpaint_imagenet', 'inpaint_face', 'faceir', 'deblur'],
            help="Chopping forward.",
            )

    parser.add_argument(
            "--num_samples",
            type=int,
            default=50,
            help="Number of image samples being processed",
            )
    parser.add_argument(
            "--downscale",
            type=float,
            default=0.25,
            help="Downsample scale",
            )
    parser.add_argument(
            "--noise_prob",
            type=float,
            default=0,
            help="Probability of corrupting with Gaussian noises",
            )
    parser.add_argument(
            "--blur",
            action='store_true', 
            help='Enable corruption by blurring'
            )
    parser.add_argument(
            "--blur_ksize",
            type=int,
            default=25,
            help='Gaussian kernel size for blurring'
            )
    parser.add_argument(
            "--artifact_prob",
            type=float,
            default=0,
            help="Probability of corrupting with compression",
            )
    parser.add_argument(
            "--artifact_quality",
            type=int,
            default=10,
            help="Percent of image quality preserved after compression",
            )
    args = parser.parse_args()

    return args


def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    # first corrupt images
    print(f'[info] start corrupting images from {args.source_zipped_image_path}')
    corrupt_in_batches_from_zip(
        source_zipped_image_path=args.source_zipped_image_path, 
        corrupted_image_path=args.corrupted_image_path,
        num_samples=args.num_samples,
        scale=args.downscale,
        noise_prob=args.noise_prob,
        is_blur=args.blur,
        blur_ksize=args.blur_ksize,
        artifact_prob=args.artifact_prob,
        artifact_quality=args.artifact_quality
    )
    print(f'[info] corrupted images are saved in {args.corrupted_image_path}')

    resshift_sampler = ResShiftSampler(
            configs,
            sf=args.upscale,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )

    # setting mask path for inpainting
    if args.task.startswith('inpaint'):
        assert args.mask_path, 'Please input the mask path for inpainting!'
        mask_path = args.mask_path
    else:
        mask_path = None
    
    print(f'[info] start restoring corrupted images from {args.corrupted_image_path}')
    resshift_sampler.inference(
            args.corrupted_image_path,
            args.reconstructed_image_path,
            mask_path=mask_path,
            bs=args.bs,
            noise_repeat=False
            )
    print(f'[info] restored images are saved in {args.reconstructed_image_path}')

if __name__ == '__main__':
    main()

