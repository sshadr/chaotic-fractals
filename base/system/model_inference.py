import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from utils.fractal_modules import fractal_code, sample_functions
from easydict import EasyDict as edict
from utils.optim_utils import uniform_sample1d, make_points2d, apply_mv_transform
from tqdm import tqdm

class ContractiveFunction(nn.Module):
    def __init__(self, weight, bias, opacity=None):
        super().__init__()        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")            
        else:
            self.device = torch.device("cpu")
            
        self.weight = weight.unsqueeze(0).to(self.device)
        self.bias = bias.unsqueeze(0).unsqueeze(-1).to(self.device)

        if opacity is None:
            self.opacity = torch.tensor(1, dtype=torch.float32).to(self.device)
        else:
            self.opacity = torch.clamp(opacity, 0.0, 1).to(self.device)            
        
    @property
    def prob(self):
        return torch.linalg.det(self.weight)

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

class ParallelIFS(nn.Module):
    def __init__(self, model_batches,  num_points):
        super().__init__()
        self.init_device()
        self.model_batches = model_batches
        self.num_points = num_points
        self.remove_points = 51 * self.model_batches

        self.tr_iter = 0
    
    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")  

    def forward(self, point, optimized_weights, optimized_biases, optimized_function_ops, code, sample_always=True):
        self.code = code
        # Batched matrix multiplication
        concatenated_pts = []
        if sample_always:
            with torch.no_grad():
                index = sample_functions(self.code, self.num_points, self.model_batches)
                
        for i in range(self.num_points):
            new_weight = optimized_weights[index[:, i]]
            new_bias = optimized_biases[index[:, i]]

            op = optimized_function_ops[index[:, i]]

            point = torch.matmul(new_weight, point) + new_bias

            x1 = torch.cat([point, op.view(self.model_batches, 1, 1)], dim=1)
            concatenated_pts.append(x1)

        concatenated_pts = torch.cat(concatenated_pts, dim=0).squeeze(2)
        points = concatenated_pts[self.remove_points:, :]

        return points

class ModelInfer(nn.Module):
    def __init__(self, optimized_code_path, lf=False, naive=False):
        super().__init__()
        """
        Inference Model - Simplified for faster testing.
        Used in the fractal visualizer.
        args: A Parser class. Contains all configs from the json file. We use only the optimizer and renderer configs here
        """
        self.init_device()
        self.code_path = optimized_code_path
        self.initialize_params()
        self.lf = lf

        with torch.no_grad():
            # naive is when the fractal code is stored as a dictionary. 
            # If the fractal code is optimized, it is most likely saved as a torch checkpoint.
            if not naive:
                self.ifs_code = torch.load(f"{self.code_path}", weights_only=False)
            else:
                self.ifs_code = edict(self.code_path)

            self.function_list = []
            if naive:
                for num_f in range(len(self.ifs_code.ifs_m)):
                    weight = torch.tensor(self.ifs_code.ifs_m[num_f], dtype = torch.float32)
                    bias = torch.tensor(self.ifs_code.ifs_t[num_f], dtype = torch.float32)

                    c = ContractiveFunction(weight = weight, bias = bias)
                    self.function_list.append(c)

            if not lf and not naive:
                for num_f in range(len(self.ifs_code.contractive_functions)):
                    lipschitz_layer = self.ifs_code.contractive_functions[num_f].ll

                    c = ContractiveFunction(weight=lipschitz_layer.weights(), bias=lipschitz_layer.bias, opacity=lipschitz_layer.opacity)
                    self.function_list.append(c)
                
                print(len(self.function_list))

            if lf:
                lf_ifs_code = edict(self.ifs_code)
                num_transforms = lf_ifs_code.w.shape[0]             

                all_w, _ = make_matrices_from_svdformat(lf_ifs_code.w)
                all_b = lf_ifs_code.b

                all_w = all_w.transpose(1, 2)
                
                for num_f in range(num_transforms):
                    c = ContractiveFunction(weight=all_w[num_f], bias=all_b[num_f])
                    self.function_list.append(c)            

            self.ifs_code = fractal_code(self.function_list, self.optimized_probs(), self.seed)
        
        self.parallel_ifs = ParallelIFS(model_batches=self.model_batches, num_points=self.num_points)
        
        self.init_points_shape = (self.model_batches, 2)
        self.init_point = torch.tensor([0, 0], dtype=torch.float32)
        self.register_buffer("batched_points", self.init_point.repeat(self.model_batches, 1).unsqueeze(-1))
        self.batched_points = self.batched_points.to(self.device) # it doesnt matter what the initial points are

        batched_points = uniform_sample1d(min=-5, max = 5, dim=self.init_points_shape).to(self.device)
        self.batched_points = batched_points.unsqueeze(-1)
        
        self.last_points = None
        self.R_90 = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32).to(self.device)

    def forward(self, scale=1.0):
        # sample a new batch of random points every iteration
        # ---------------------
        if self.last_points is not None:
            self.batched_points = self.last_points
            
        # ---------------------
    
        points = self.parallel_ifs.forward(self.batched_points, self.optimized_weights(), self.optimized_biases(), self.optimized_function_opacities(), self.ifs_code, sample_always=True) # takes batched points (class attribute)
        ops = points[:, 2, None]
        points = points[:, :2]
        
        if self.last_points is None:
            tpoints, self.full_scale, self.full_translation = apply_mv_transform(points, return_transform=True)
        else:
            # once original scaling and translations are stored, use them for all subsequent scales
            translated_points = points - self.full_translation
            tpoints = translated_points * self.full_scale 

        if self.lf:
            tpoints = torch.matmul(tpoints.squeeze(-1), self.R_90)

        top_points = torch.cat([tpoints, ops], dim=1)

        # store the last point(s) for future iteration incase of batched generation
        self.last_points = points[-self.model_batches:, :2].unsqueeze(-1)

        return top_points
    
    def initialize_params(self):
        self.num_points = 250
        self.seed = 1 # random
        self.model_batches = 25000 #200000

    def get_processed_biases(self):
        functions = self.function_list
        pbiases = []
        for _, layer in enumerate(functions):
            w = layer.bias
            pbiases.append(w)
        
        return torch.cat(pbiases, dim=0)
    
    def get_processed_weights(self):
        functions = self.function_list
        pweights = []
        for _, layer in enumerate(functions):
            w = layer.weight
            pweights.append(w)
        
        return pweights
    
    def optimized_weights(self):
        w = self.get_processed_weights()
        return torch.cat(w, dim=0)
    
    def optimized_biases(self):
        return self.get_processed_biases()
    
    def optimized_function_opacities(self):
        functions = self.function_list
        ops = []
        for i, layer in enumerate(functions):
            ops.append(layer.opacity)
        
        ops_tensor = torch.stack(ops)
        
        return ops_tensor
    
    def optimized_probs(self, scale=1.0):
        functions = self.function_list
        probs = []
        for i, layer in enumerate(functions):
            probs.append(layer.prob)
        
        probs = torch.abs(torch.tensor(probs))
        probs += 0.01

        normalized_probs = probs/probs.sum()
        return normalized_probs
    
    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")


# =======================================
# Batched point generation 
def batch_point_gen(model, total_points_target, points_path=None):
    '''
    points_path (optional) -- save path for generated points
    '''
    total_points = torch.empty((0, 2)).cuda() # not the most efficient way. Necessary for dynamically allocating generations to the memory.

    total_points_target_in_millions = total_points_target / 1e6
    with tqdm(total=total_points_target_in_millions, desc="Generated Points", unit=" M points") as pbar:
        while total_points.shape[0] < total_points_target:
            # Generate a batch of points
            points = model.forward()
            points = make_points2d(points)

            # Update the progress bar with the number of new points
            pbar.update(int(points.shape[0])/1e6)

            total_points = torch.cat((total_points, points), dim=0)
            if total_points.shape[0] >= total_points_target:
                total_points = total_points[:total_points_target]
                break
    
    print("Generated num points:", total_points.shape)
    return total_points

    # # save points to disk
    # print(f"Saving point cloud to {points_path} .......")
    # torch.save(total_points, points_path)
    # print("Saved point cloud")


#  --------------------------------------------
# from learning fractals codebase
def make_rotation_matrix(theta):
    r_mat = torch.concat([torch.cos(theta), -torch.sin(theta),
                          torch.sin(theta),  torch.cos(theta)]).view(2, 2)
    return r_mat


def make_diagnal_matrix(sigma_1, sigma_2):
    zero =  torch.zeros(1).to(sigma_1.device)
    d_mat = torch.concat([sigma_1,  zero,
                          zero, sigma_2]).view(2, 2)
    return d_mat

def make_matrices_from_svdformat(ifs_w_weight, force_all=True):
    all_w = []
    all_sgv = []
    for i in range(ifs_w_weight.shape[0]):
        theta_1, theta_2, sigma_1, sigma_2, d1, d2 = ifs_w_weight[i]
        r_mat1 = make_rotation_matrix(theta_1.unsqueeze(0))
        r_mat2 = make_rotation_matrix(theta_2.unsqueeze(0))
        if i == 0 or force_all:
            sig_mat = make_diagnal_matrix(
                    torch.sigmoid(sigma_1.unsqueeze(0)),
                    torch.sigmoid(sigma_2.unsqueeze(0)))
        else:
            sig_mat = make_diagnal_matrix(
                    F.softplus(sigma_1.unsqueeze(0)),
                    F.softplus(sigma_2.unsqueeze(0)))
        all_sgv.append(torch.diag(sig_mat).unsqueeze(0))
        d1 = d1.unsqueeze(0)
        d2 = d2.unsqueeze(0)
        d_mat = make_diagnal_matrix(
                d1.sign() - d1.detach() + d1,
                d2.sign() - d2.detach() + d2)
        w = torch.matmul(torch.matmul(torch.matmul(r_mat1, sig_mat), r_mat2), d_mat).T
        all_w.append(w.unsqueeze(0))
    return torch.cat(all_w, dim=0), torch.cat(all_sgv, dim=0)

# END --------------------------------------------
