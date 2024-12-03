import os
import numpy as np
import cv2
import argparse
import zipfile

from ResShift.utils.util_image import calculate_ssim, calculate_psnr, imread, bgr2rgb
import torch
import pyiqa


def calc_metrics(gt_zip_dir, sr_dir):
    clipiqa_metric = pyiqa.create_metric('clipiqa')
    musiq_metric = pyiqa.create_metric('musiq')
    PSNR = 0
    SSIM = 0
    CLIPIQA = 0
    MUSIQ = 0
    num_total = len(os.listdir(sr_dir))
    print(f'Calculating metrics for {num_total} images')
    
    with zipfile.ZipFile(gt_zip_dir, 'r') as zip_ref:
        for file_name in os.listdir(sr_dir):
            with zip_ref.open(os.path.join('images', file_name)) as file:
                image_bytes = file.read()
                gt_im = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
                gt_im = bgr2rgb(gt_im)
                gt_im_norm = gt_im.astype(np.float32) / 255.
                sr_im = cv2.imread(os.path.join(sr_dir, os.path.basename(file_name)), cv2.IMREAD_UNCHANGED)
                sr_im = bgr2rgb(sr_im)
                sr_im_norm = sr_im.astype(np.float32) / 255.
                PSNR += calculate_psnr(sr_im_norm, gt_im_norm)
                SSIM += calculate_ssim(sr_im_norm, gt_im_norm)

                
                gt_im_norm = torch.from_numpy(gt_im_norm).permute(2,0,1).unsqueeze(0).cuda()
                sr_im_norm = torch.from_numpy(sr_im_norm).permute(2,0,1).unsqueeze(0).cuda()
                CLIPIQA += clipiqa_metric(sr_im_norm).sum().item()
                MUSIQ += musiq_metric(sr_im_norm).sum().item()
    
    PSNR /= num_total
    SSIM /= num_total
    CLIPIQA /= num_total
    MUSIQ /= num_total

    return PSNR, SSIM, CLIPIQA, MUSIQ




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_zipped_image_path", type=str, default="", help="Source Zipped Image path.")
    parser.add_argument("-r", "--reconstructed_image_path", type=str, default="", help="Reconstructed Image path.")

    args = parser.parse_args()
    psnr, ssim, clipiqa, musiq = calc_metrics(gt_zip_dir=args.source_zipped_image_path, sr_dir=args.reconstructed_image_path)
    print(f'psnr: {psnr}, ssim: {ssim}, clipiqa: {clipiqa}, musiq: {musiq}')



