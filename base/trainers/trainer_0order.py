# Stochastic Algorithms: Evolutionary and Cuckoo search 

import math
import os
import random
from functools import cache, partial
import numpy as np
import torch
from easydict import EasyDict as edict
from cf.images.conversions import tensor_to_image, image_to_tensor
from cf.images.image_io import save_image, load_image
from lpips import LPIPS
from torch import nn
from tqdm import trange, tqdm

from system.model import Model
from system.renderer import GaussianSplatter, OglRenderer
from utils.losses import ssim, lipschitz_constraint_sigma, constrain_bias, constrain_condition_number


def optimize(gt_name, optimizer, renderer, fitness, num_gens, gen_size, ifs_size, out_dir):
    args = Args(edict({
        "seed" : 5,
        "num_points" : 250,
        "num_matrices" : ifs_size,
        "model_batches" : 4000,
        "parameterization" : "naive",
        "init" : "random",
        "randomize_probs" : 0,
        "randomize_sequence" : 0,
    }))
    
    logger = edict({
    "log_dir": os.path.join(out_dir, "tmp"),
    "init_path": os.path.join(out_dir, "tmp")
    })

    os.makedirs(logger.log_dir, exist_ok=True)
    os.makedirs(logger.init_path, exist_ok=True)

    match renderer:
        case "binary":
            renderer_type = OglRenderer
        case "gaussian":
            renderer_type = GaussianSplatter
    render_config = edict({
        "splat_scale": 1,
        "radius": 0.03,
        "supersample_factor": 5,
        "image_res": 1024,
        "background_color": 0
    })
    renderer = renderer_type(torch.tensor([[0.0, 0.0, 1.0]]), render_config)

    match fitness:
        case "pointcov":
            fitness_fn = point_coverage_fitness
        case "hamming":
            fitness_fn = hamming_fitness
        case "ourloss":
            fitness_fn = our_fitness
    gt_path = f"dataset/full_dataset/ifs/{gt_name}.png"
    gt_image = image_to_tensor(load_image(gt_path), normalize=False, to_cuda=True)
    fitness_fn = partial(fitness_fn, gt_image)

    gen = []
    for _ in trange(gen_size, desc="Setup"):
        model = Model(args, logger)
        if optimizer == "cuckoosearch":
            matrices, biases = model.get_all_params_naive()
            for matrix in matrices:
                nn.init.uniform_(matrix, -0.707, 0.707)
            for bias in biases:
                nn.init.uniform_(bias, -1, 1)
        gen.append(Individual(model, renderer, fitness_fn))
    save_best_individuals(gen, out_dir, 0)

    if optimizer == "evolutionary":
        for itr in trange(num_gens, desc="Generations"):
            for p in trange(gen_size, desc="Parents", leave=False):
                parent = gen[p]
                matrices, biases = parent.model.get_all_params_naive()
                sigma1 = min(0.1, math.sqrt((1 - parent.fitness) / 500))
                sigma2 = min(5.0, math.sqrt((1 - parent.fitness) * 5))
                with torch.no_grad():
                    for i in range(len(matrices)):
                        matrices[i] = (matrices[i] + (torch.randn_like(matrices[i]) * sigma1)).clamp(-0.707, 0.707)
                    for i in range(len(biases)):
                        biases[i] = (biases[i] + (torch.randn_like(biases[i]) * sigma2)).clamp(-1, 1)
                child_model = Model(args, logger)
                child_model.replace_all_params_naive(matrices, biases)
                gen.append(Individual(child_model, renderer, fitness_fn))
            scores = [sum(other.fitness <= ind.fitness for other in random.sample(gen, 5)) for ind in gen]
            gen = [gen[i] for i in np.argsort(scores)[-gen_size:]]
            save_best_individuals(gen, out_dir, itr + 1)
        return max(gen, key=lambda ind: ind.fitness).model
    # Cuckoo search reimplemented based on the Matlab code from the paper "Engineering Optimisation by Cuckoo Search".
    elif optimizer == "cuckoosearch":
        best = max(gen, key=lambda ind: ind.fitness)
        for itr in trange(num_gens, desc="Generations"):
            best_matrices, best_biases = best.model.get_all_params_naive()
            for n in trange(gen_size, desc="Egg Laying", leave=False):
                matrices, biases = gen[n].model.get_all_params_naive()
                with torch.no_grad():
                    for i in range(len(matrices)):
                        matrices[i] = cuckoo_step(matrices[i], best_matrices[i]).clamp(-0.707, 0.707)
                    for i in range(len(biases)):
                        biases[i] = cuckoo_step(biases[i], best_biases[i]).clamp(-1, 1)
                egg_model = Model(args, logger)
                egg_model.replace_all_params_naive(matrices, biases)
                egg = Individual(egg_model, renderer, fitness_fn)
                if gen[n].fitness < egg.fitness:
                    gen[n] = egg
            for n in trange(gen_size, desc="Nest Abandoning", leave=False):
                if random.random() < 0.25:
                    matrices, biases = gen[n].model.get_all_params_naive()
                    matrices1, biases1 = random.choice(gen).model.get_all_params_naive()
                    matrices2, biases2 = random.choice(gen).model.get_all_params_naive()
                    with torch.no_grad():
                        for i in range(len(matrices)):
                            matrices[i] = (matrices[i] + random.random() * (matrices1[i] - matrices2[i])).clamp(-0.707, 0.707)
                        for i in range(len(biases)):
                            biases[i] = (biases[i] + random.random() * (biases1[i] - biases2[i])).clamp(-1, 1)
                    new_model = Model(args, logger)
                    new_model.replace_all_params_naive(matrices, biases)
                    gen[n] = Individual(new_model, renderer, fitness_fn)
            save_best_individuals(gen, out_dir, itr + 1)
            best = max(best, *gen, key=lambda ind: ind.fitness)
            # print(gen[0].model.get_all_params_naive())
            # safe_state(1)
            # save_image(tensor_to_image(Individual(gen[0].model, renderer, fitness_fn).image, denormalize=False), os.path.join(out_dir, f"gen{itr + 1:03}", "best.png"))
            # safe_state(1)
            # save_image(tensor_to_image(Individual(gen[0].model, renderer, fitness_fn).image, denormalize=False), os.path.join(out_dir, f"gen{itr + 1:03}", "best2.png"))
            # safe_state(1)
            # save_image(tensor_to_image(Individual(gen[0].model, renderer, fitness_fn).image, denormalize=False), os.path.join(out_dir, f"gen{itr + 1:03}", "best3.png"))
            # save_image(tensor_to_image(Individual(gen[0].model, renderer, fitness_fn).image, denormalize=False), os.path.join(out_dir, f"gen{itr + 1:03}", "best4.png"))
            # save_image(tensor_to_image(Individual(gen[0].model, renderer, fitness_fn).image, denormalize=False), os.path.join(out_dir, f"gen{itr + 1:03}", "best5.png"))
        return best.model


def cuckoo_step(tensor, best_tensor):
    beta = 1.5
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    zeros = torch.zeros_like(tensor)
    step = torch.normal(zeros, sigma) / torch.normal(zeros, 1).abs() ** (1 / beta)
    return tensor + 0.01 * step * torch.normal(zeros) * (tensor - best_tensor)


def save_best_individuals(gen, out_dir, no):
    top = sorted(gen, key=lambda ind: ind.fitness, reverse=True)
    tqdm.write("Fitness: " + " ".join(f"{ind.fitness:7.4f}" for ind in top[:10]))
    gen_dir = os.path.join(out_dir, f"gen{no:03}")
    os.makedirs(gen_dir, exist_ok=True)
    for i, ind in enumerate(top[:3]):
        save_image(tensor_to_image(ind.image, denormalize=False), os.path.join(gen_dir, f"{i} {ind.fitness:.4f}.png"))


class Args:

    def __init__(self, optimizer_config):
        self.optimizer_config = optimizer_config

    def get_optimizer_config(self):
        return self.optimizer_config


class Individual:

    def __init__(self, model, renderer, fitness_fn):
        self.model = model
        with torch.no_grad():
            renderer.update_points(model())
            image = renderer.render()
        self.fitness = fitness_fn(image, model)
        self.image = image.cpu()


def point_coverage_fitness(image1, image2, model):
    image1 = image1[0, 0] >= 0.25
    image2 = image2[0, 0] >= 0.25
    return ((image1 & image2).sum() - (image1 ^ image2).sum()).item() / np.prod(image1.shape)


def hamming_fitness(image1, image2, model):
    image1 = image1[0, 0] >= 0.25
    image2 = image2[0, 0] >= 0.25
    return (image1 != image2).sum().item() / np.prod(image1.shape)


def our_fitness(image1, image2, model):
    singular_values = model.get_singular_values()
    loss = (2 * lpips_net()(image1, image2) +
            1 - ssim(image1, image2) +
            0.01 * lipschitz_constraint_sigma(singular_values) +
            0.01 * constrain_bias(model.get_processed_biases()) +
            0.1 * constrain_condition_number(singular_values))
    return torch.exp(-loss).item()


@cache
def lpips_net():
    return LPIPS().cuda()

