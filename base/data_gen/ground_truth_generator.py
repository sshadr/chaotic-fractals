import os, sys

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

import numpy as np

from cf.images.image_io import save_image

from system.model_inference import ModelInfer, batch_point_gen
from system.renderer import SuperSampleRender

from utils.improved_pretraining import fractal_generator
from utils.optim_utils import safe_state, write_to_json

import click
import warnings

warnings.filterwarnings("ignore")

@click.command()
@click.option('--dir', help='Directory to save fractals', required=True)
@click.option('--num_f', help='resolution', required=True, type=int)
@click.option('--gt_name', help='Groundtruth image name (string)', required=True)
@click.option('--method', help='(imp)-Improved fractal training (string)', required=True)
def suite(dir, gt_name, num_f, method):
    if method == "imp":
        code = fractal_generator(num_f, gt_name) # gt_name is the seed for the fractal generator
        save_dir = os.path.join(dir, f"N_{num_f}")
    print("Saving in: ", save_dir)
    images_dir = os.path.join(save_dir, "all_images")
    os.makedirs(images_dir, exist_ok=True)

    codes_dir = os.path.join(save_dir, "all_codes")
    os.makedirs(codes_dir, exist_ok=True)

    points_generator = ModelInfer(optimized_code_path=code, lf=False, naive=True)
    
    total_points_target = int(1e8/2)
    base_res = 1024
    factor = 16
    color = np.ones(3, dtype=np.float32)
    bg_color = 0
    
    write_to_json(code, os.path.join(codes_dir, f"fdb_{gt_name}.json"))
    
    points = batch_point_gen(points_generator, total_points_target)
    fractal = SuperSampleRender(points, base_res, factor, color, bg_color)
    save_image(fractal, os.path.join(images_dir, f"fdb_{gt_name}.png"))

if __name__=='__main__':
    safe_state(1)
    suite()
    print("\n === TERMINATED ===")