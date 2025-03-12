import os, sys
import click
import time
import warnings

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from utils.optim_utils import safe_state, Parser
from trainers.trainer import FractalTrainer
from trainers.trainer_sa import FractalTrainerSA
from trainers.trainer_moment import FractalTrainerMoments
from cf.images.image_io import load_image

warnings.filterwarnings("ignore")

@click.command()
@click.option('--dir', help='Directory to save experiments', required=True)
@click.option('--config', help='Configuration file', required=True)
@click.option('--gt_name', help='Groundtruth image name (string)', required=True)
@click.option('--trainer_type', type=click.Choice(['cf', 'sa', 'moments']), required=True,
              help='Choose the type of trainer to use')
def suite(dir, config, gt_name, trainer_type):
    
    start = time.time()
    config = os.path.join("configs", f"{config}.json")
    
    args = Parser(config)
    seed = args.config.optimizer.seed
    safe_state(seed)

    num_matrices = args.config.optimizer.num_matrices
    args.config.logdirs.expt_name = f"{dir}/N_{num_matrices}/{gt_name}"

    gt_path = os.path.join("dataset", "full_dataset", "ifs", f"{gt_name}.png")
    gt_img = load_image(gt_path)
    
    if trainer_type == 'cf':
        trainer = FractalTrainer(args, gt_img)
    if trainer_type == 'sa':
        trainer = FractalTrainerSA(args, gt_img)
    if trainer_type == 'moments':
        trainer = FractalTrainerMoments(args, gt_img)

    trainer.train()
    end = time.time()
    print("------------------------------------")
    print("Time taken :", end-start)

if __name__ == "__main__":
    suite()
    print("\n === TERMINATED ===")