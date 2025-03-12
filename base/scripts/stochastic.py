import json
import os, sys
import shutil
import warnings
import click
import numpy as np
import torch
from tqdm import trange

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from cf.images.image_io import save_image
from system.model_inference import ModelInfer, batch_point_gen
from system.renderer import SuperSampleRender
from utils.optim_utils import safe_state, make_points2d
from trainers.trainer_0order import optimize

warnings.filterwarnings("ignore")

@click.command
@click.option("--gt_name", help="Groundtruth image name", required=True)
@click.option("--optimizer", default="evolutionary", type=click.Choice(["evolutionary", "cuckoosearch"]))
@click.option("--renderer", default="binary", type=click.Choice(["binary", "gaussian"]))
@click.option("--fitness", default="pointcov", type=click.Choice(["pointcov", "hamming", "ourloss"]))
@click.option("--num_gens", help="number of generations", default=100, type=int)
@click.option("--gen_size", help="number of individuals in a generation", default=100, type=int)
@click.option("--ifs_size", help="number of IFS functions, or 0 to mirror the groundtruth", default=0, type=int)
def main(gt_name, optimizer, renderer, fitness, num_gens, gen_size, ifs_size):
    if ifs_size <= 0:
        with open(f"dataset/full_dataset/ifs_codes/{gt_name}.json") as f:
            ifs_size = len(json.load(f)["ifs_m"])
    out_dir = f"log/{optimizer}_{renderer}_{fitness}_{num_gens}gens_{gen_size}inds/N_{ifs_size}/{gt_name}/output"
    shutil.rmtree(out_dir, ignore_errors=True)

    best_model = optimize(gt_name, optimizer, renderer, fitness, num_gens, gen_size, ifs_size, out_dir)

    ifs_code_file = os.path.join(out_dir, "best_optimized_ifs_code.pth")
    torch.save(best_model.ifs_code, ifs_code_file)

    best_model = ModelInfer(ifs_code_file)
    
    total_points_target = int(1e8/2)
    points = batch_point_gen(best_model, total_points_target)
    final_image = SuperSampleRender(points, 1024, 16)
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    save_image(final_image, os.path.join(val_dir, "supersampled.png"))

if __name__ == "__main__":
    safe_state(1)
    main()
