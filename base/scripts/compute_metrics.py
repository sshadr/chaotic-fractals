import os, sys


def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

import math
import torch
import numpy as np
from natsort import natsorted
from sklearn.metrics import f1_score as sklearn_f1_score
from cf.tools.string_tools import print_same_line
from cf.images.image_io import load_image, find_images
from cf.images.conversions import tensor_to_image, image_to_tensor
from lpips import LPIPS
from utils.losses import ssim
import click

class MSE:
    def __init__(self):
        self.name = "MSE"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2)**2)
        return mse
    
class IoU:
    def __init__(self):
        self.name = "IoU"

    @staticmethod
    def __call__(img1, img2):
        inter = np.logical_and(img1 > 0, img2 > 0)
        intersection = np.count_nonzero(inter)

        mask1 = img1 > 0
        mask2 = img2 > 0

        union_mask = mask1 | mask2
        union = np.count_nonzero(union_mask)
        iou = intersection / union
        return iou            

def f1_score(img1, img2):
    binary_image1 = img1 > 0
    binary_image2 = img2 > 0

    # Flatten the images
    binary_image1 = binary_image1.flatten()
    binary_image2 = binary_image2.flatten()

    # Use sklearn's f1_score with average='binary'
    f1 = sklearn_f1_score(binary_image1, binary_image2, average='binary')

    return f1

def calculate_metrics(dir1, dir2):
    files1 = natsorted(find_images(dir1))
    files2 = natsorted(find_images(dir2))

    assert len(files1) == len(files2), "Image sets have different sizes."
    img_count = len(files1)

    lpips_model = LPIPS(net='alex').cuda()

    mse_total = 0
    psnr_total = 0
    ssim_total = 0
    lpips_total = 0
    pearson_total = 0
    iou_total = 0
    f1_total = 0

    mse_calculator = MSE()
    iou_calculator = IoU()

    res = 1024
    crop = int(0.125*res) # remove padding
    
    for idx, (file1, file2) in enumerate(zip(files1, files2)):
        print_same_line(f"Processing image {idx+1}/{img_count}...")
        img1 = image_to_tensor(load_image(file1), to_cuda=True)
        img2 = image_to_tensor(load_image(file2), to_cuda=True)
        
        img1 = img1[:, :, crop:res-crop, crop:res-crop]
        img2 = img2[:, :, crop:res-crop, crop:res-crop]

        mse = mse_calculator(img1, img2)
        mse_total += mse
        psnr_total += 10 * math.log(1 / mse) / math.log(10)
       
        ssim_total += ssim(img1, img2) 
        
        lpips_total += lpips_model.forward(img1, img2).item()
        
        f1_total += f1_score(tensor_to_image(img1), tensor_to_image(img2))

        iou_total += iou_calculator(tensor_to_image(img1), tensor_to_image(img2))
        

    num_files = len(files1)
    mse_avg = mse_total / num_files
    psnr_avg = psnr_total / num_files
    ssim_avg = ssim_total / num_files
    lpips_avg = lpips_total / num_files
    pearson_avg = pearson_total / num_files
    iou_average = iou_total / num_files
    f1_average = f1_total/num_files

    print("Average MSE:", mse_avg)
    print("Average PSNR:", psnr_avg)
    print("Average SSIM:", ssim_avg)
    print("Average LPIPS:", lpips_avg)
    print("Average Pearson R:", pearson_avg)
    print("Average IoU", iou_average)
    print("Average F1", f1_average)

    # print(f'{mse_avg:.4f} & {psnr_avg:.4f} &  {ssim_avg:.4f} & {lpips_avg:.4f} & {f1_average:.4f} & {iou_average:.4f} & {pearson_avg:.4f}')
    print(f'{psnr_avg:.4f} &  {ssim_avg:.4f} & {lpips_avg:.4f} & {f1_average:.4f} & {iou_average:.4f}')
    
    results_file = os.path.join(dir2, "results.txt")
    with open(results_file, 'a') as f:
        # Write the formatted string into the file
        f.write(f'{psnr_avg:.4f} & {ssim_avg:.4f} & {lpips_avg:.4f} & {f1_average:.4f} & {iou_average:.4f}\n')

    print(f"Results saved to {results_file}")
   
@click.command()
@click.option('--root', help='root dir (assuming both folders are in the same folder)', required=True)
@click.option('--gt_folder', help='folder name of gt fractals', required=True)
@click.option('--opt_folder', help='folder name of optimized fractals', required=True)
def custom(root, gt_folder, opt_folder):
    dir1 = os.path.join(root, f"{gt_folder}")
    dir2 = os.path.join(root, opt_folder)
    calculate_metrics(dir1, dir2)
    # a results file will get saved in the "opt_folder" as results.txt

if __name__ == '__main__':
    custom()
    