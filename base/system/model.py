import os, sys
import torch
import torch.nn as nn
from utils.fractal_modules import fractal_code
from easydict import EasyDict as edict
from utils.optim_utils import apply_mv_transform, uniform_sample1d, write_to_json
from system.blocks import *

## Optimization Model 
class Model(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        """
        args: A Parser class. Contains all configs from the json file. We use only the optimizer and renderer configs here
        """
        self.init_device()

        self.args = args
        self.logger = logger
        self.initialize_params()

        self.init_ifs_code()
        
        self.parallel_ifs = ParallelIFS(model_batches=self.model_batches, num_points=self.num_points, randomize_sequence=self.randomize_sequence)
        
        self.init_points_shape = (self.model_batches, 2)
        self.init_point = torch.tensor([0, 0], dtype=torch.float32)
        self.register_buffer("batched_points", self.init_point.repeat(self.model_batches, 1).unsqueeze(-1))
        self.batched_points = self.batched_points.to(self.device)

        # initialize colors
        # (optional) while returned as a learnable param, it is not optimized.
        self.foreground_color = 1 * torch.ones((1, 3), dtype=torch.float32).to(self.device)        
        self.foreground_color.requires_grad = True
        self.color_params = [self.foreground_color]
    
    def forward(self):
        with torch.no_grad():
            # sample a new batch of random points every iteration
            # ---------------------
            batched_points = uniform_sample1d(min=-5, max = 5, dim=self.init_points_shape).to(self.device)            
            self.batched_points = batched_points.unsqueeze(-1)

            self.ifs_code = fractal_code(self.function_list, self.optimized_probs(), self.seed)            
            # ---------------------
        
        points = self.parallel_ifs.forward(self.batched_points, self.optimized_weights(), self.optimized_biases(), self.optimized_function_opacities(), self.ifs_code, sample_always=True) # takes batched points (class attribute)
        ops = points[:, 2, None]
        points = points[:, :2]
        tpoints = apply_mv_transform(points)
        top_points = torch.cat([tpoints, ops], dim=1)
        
        return top_points
    
    def initialize_params(self):
        self.optimizer_config = self.args.get_optimizer_config()

        self.N = self.optimizer_config.num_matrices
        self.num_points = self.optimizer_config.num_points
        self.seed = self.optimizer_config.seed
        self.model_batches = self.optimizer_config.model_batches
        self.parameterization = self.optimizer_config.parameterization
        self.init_scheme = self.optimizer_config.init
        
        self.randomize_probs = bool(self.optimizer_config.randomize_probs)
        self.randomize_sequence = bool(self.optimizer_config.randomize_sequence)
        
        if self.randomize_probs:
            print("================= Randomizing Probs ================= ")
                    
    def init_ifs_code(self):
        # Initialize functions
        self.function_list = [ContractiveFunction(parameterization=self.parameterization, init_scheme=self.init_scheme) for _ in range(self.N)]
       
        self.ifs_code = fractal_code(self.function_list, self.optimized_probs(), self.seed)        
        data = self.create_stats_dict("init")
        write_to_json(data, f"{self.logger.init_path}/init_weights_stats.json")        
    
    def save_current_state(self, save_opacity=False):
        self.current_model_params = [param.clone() for param in self.parameters()]  # Store original parameters
        if self.parameterization == 'naive':
            self.current_matrices, self.current_lbiases = self.get_all_params_naive()
        if self.parameterization == 'qr' and not save_opacity:
            self.current_singular_values, self.current_au_values, self.current_av_values, self.current_lbiases = self.get_all_params_qr(return_opacity=False)
        if self.parameterization == 'qr' and save_opacity:            
            self.current_singular_values, self.current_au_values, self.current_av_values, self.current_lbiases, self.current_opacities = self.get_all_params_qr(return_opacity=True)
        if self.parameterization == 'mat_exp':
            self.current_singular_values, self.current_uv_values, self.current_lbiases = self.get_all_params_mat_exp()
        if self.parameterization == 'angle':
            self.current_singular_values, self.current_angles, self.current_lbiases = self.get_all_params_angle_param()
    
    def get_learnable_singular_values(self):
        functions = self.function_list
        singular_values = []
        for _, layer in enumerate(functions):
            sv = layer.ll.psigmas             
            singular_values.extend([sv])
        
        return singular_values
    
    def get_singular_values(self):
        functions = self.function_list
        singular_values = []
        for _, layer in enumerate(functions):
            sv = layer.ll.sigmas             
            singular_values.extend([sv])
        
        return singular_values
    
    def get_opacity_values(self):
        functions = self.function_list
        ops = []
        for _, layer in enumerate(functions):
            op = layer.ll.opacity          
            ops.extend([op])

        opacity_tensor = torch.stack(ops)
        return opacity_tensor
    
    def get_all_params_angle_param(self):
        functions = self.function_list
        singular_values = []
        angles = []
        learnable_biases = []
        for _, layer in enumerate(functions):
            sv = layer.ll.psigmas 
            lbias = layer.ll.lbias

            singular_values.extend([sv])
            learnable_biases.extend([lbias])
            
            angle = layer.ll.angles
            angles.extend([angle])
            
        return singular_values, angles, learnable_biases
    
    def get_all_params_mat_exp(self):
        functions = self.function_list
        singular_values = []
        uv_values = []
        learnable_biases = []
        for _, layer in enumerate(functions):
            sv = layer.ll.psigmas 
            lbias = layer.ll.lbias

            singular_values.extend([sv])
            learnable_biases.extend([lbias])
            
            uv = layer.ll.uv_params
            uv_values.extend([uv])
            
        return singular_values, uv_values, learnable_biases
    
    def get_all_params_qr(self, return_opacity=False):
        functions = self.function_list
        singular_values = []
        au_values = []
        av_values = []
        learnable_biases = []
        opacity = []
        for _, layer in enumerate(functions):
            sv = layer.ll.psigmas 
            lbias = layer.ll.lbias

            singular_values.extend([sv])
            learnable_biases.extend([lbias])
            
            au = layer.ll.au
            av = layer.ll.av
            
            au_values.extend([au])
            av_values.extend([av])

            if return_opacity:
                op = layer.ll.opacity
                opacity.extend([op])
        
        if return_opacity:
            return singular_values, au_values, av_values, learnable_biases, opacity
        
        return singular_values, au_values, av_values, learnable_biases
        
    def get_all_params_naive(self):
        functions = self.function_list
        matrices = []
        learnable_biases = []
        for _, layer in enumerate(functions):
            mat = layer.ll.matrix 
            lbias = layer.ll.lbias

            matrices.extend([mat])
            learnable_biases.extend([lbias])
        
        
        return matrices, learnable_biases
    
    # ---------------------------------
    def randomize_all_params_qr(self, rand_opacity=False):
        '''
        used for perturbing singular values in simulated annealing
        singular_values: mostly should be self.current_singular_values 
        noise: gaussian noise's std. we apply random gaussian noise with mean=0, std=noise to the singular values to perturb

        '''
        functions = self.function_list
        for i, layer in enumerate(functions):            
            layer.ll.psigmas =  2 * torch.rand_like(layer.ll.psigmas) - 1
            layer.ll.au =  2 * torch.rand_like(layer.ll.au) - 1
            layer.ll.av =  2 * torch.rand_like(layer.ll.av) - 1
            layer.ll.lbias = 2 * torch.rand_like(layer.ll.lbias) - 1
            
            layer.ll.psigmas.requires_grad = True
            layer.ll.au.requires_grad = True
            layer.ll.av.requires_grad = True
            layer.ll.lbias.requires_grad = True

            if rand_opacity:
                layer.ll.opacity = torch.rand_like(layer.ll.opacity) # [0, 1]
                layer.ll.opacity.requires_grad = True


    def randomize_all_params_naive(self):
        '''
        used for perturbing singular values in simulated annealing
        singular_values: mostly should be self.current_singular_values 
        noise: gaussian noise's std. we apply random gaussian noise with mean=0, std=noise to the singular values to perturb

        '''
        functions = self.function_list
        for i, layer in enumerate(functions):
            layer.ll.matrix =  2 * torch.rand_like(layer.ll.matrix) - 1
            layer.ll.lbias = 2 * torch.rand_like(layer.ll.lbias) - 1

            layer.ll.matrix.requires_grad = True
            layer.ll.lbias.requires_grad = True

    # ---------------------------------
    def add_noise_to_grad_params(self, noise=0.1):
        params1, params2, params3, _ = self.get_learnable_params()
        
        all_params = params1 + params2 #+ params3

        for param in all_params:
            grad_noise = torch.randn_like(param) * noise
            param.grad += grad_noise            

    def replace_all_params_qr(self, singular_values, au_values, av_values, learnable_biases, current_function_opacity=None, noise=0):
        '''
        used for perturbing singular values in simulated annealing
        singular_values: mostly should be self.current_singular_values 
        noise: gaussian noise's std. we apply random gaussian noise with mean=0, std=noise to the singular values to perturb

        '''
        functions = self.function_list
        for i, layer in enumerate(functions):
            if current_function_opacity is not None:
                singular_tensor, au_tensor, av_tensor, bias_tensor, op_tensor = singular_values[i], au_values[i], av_values[i], learnable_biases[i], current_function_opacity[i]
            else:
                singular_tensor, au_tensor, av_tensor, bias_tensor = singular_values[i], au_values[i], av_values[i], learnable_biases[i]
            
            layer.ll.psigmas = singular_tensor + torch.randn_like(singular_tensor) * noise
            layer.ll.au = au_tensor + torch.randn_like(au_tensor) * noise 
            layer.ll.av = av_tensor + torch.randn_like(av_tensor) * noise
            layer.ll.lbias = bias_tensor + torch.randn_like(bias_tensor) * noise
            
            layer.ll.psigmas.requires_grad = True
            layer.ll.au.requires_grad = True
            layer.ll.av.requires_grad = True
            layer.ll.lbias.requires_grad = True

            if current_function_opacity is not None:
                layer.ll.opacity = torch.ones_like(op_tensor) * 0.5 # reset to 0.5 and let gradient descent take over no simulated annealing
                layer.ll.opacity.requires_grad = True
    
    def replace_all_params_naive(self, matrices, learnable_biases, noise=0):
        '''
        used for perturbing singular values in simulated annealing
        singular_values: mostly should be self.current_singular_values 
        noise: gaussian noise's std. we apply random gaussian noise with mean=0, std=noise to the singular values to perturb

        '''
        functions = self.function_list
        for i, layer in enumerate(functions):
            matrix_tensor, bias_tensor = matrices[i], learnable_biases[i]
            layer.ll.matrix = matrix_tensor + torch.randn_like(matrix_tensor) * noise
            layer.ll.lbias = bias_tensor + torch.randn_like(bias_tensor) * noise

            layer.ll.matrix.requires_grad = True
            layer.ll.lbias.requires_grad = True

    def replace_all_params_mat_exp(self, singular_values, uv_values, learnable_biases, noise=0):
        '''
        used for perturbing singular values in simulated annealing
        singular_values: mostly should be self.current_singular_values 
        noise: gaussian noise's std. we apply random gaussian noise with mean=0, std=noise to the singular values to perturb

        '''
        functions = self.function_list
        for i, layer in enumerate(functions):
            singular_tensor, uv_tensor, bias_tensor = singular_values[i], uv_values[i], learnable_biases[i]
            layer.ll.psigmas = singular_tensor + torch.randn_like(singular_tensor) * noise
            layer.ll.uv_params = uv_tensor + torch.randn_like(uv_tensor) * noise
            layer.ll.lbias = bias_tensor + torch.randn_like(bias_tensor) * noise

            layer.ll.psigmas.requires_grad = True
            layer.ll.uv_params.requires_grad = True
            layer.ll.lbias.requires_grad = True

    def replace_all_params_angle_param(self, singular_values, angles, learnable_biases, noise=0):
        '''
        used for perturbing singular values in simulated annealing
        singular_values: mostly should be self.current_singular_values 
        noise: gaussian noise's std. we apply random gaussian noise with mean=0, std=noise to the singular values to perturb

        '''
        functions = self.function_list
        for i, layer in enumerate(functions):
            singular_tensor, angles_tensor, bias_tensor = singular_values[i], angles[i], learnable_biases[i]
            layer.ll.psigmas = singular_tensor + torch.randn_like(singular_tensor) * noise
            layer.ll.angles = angles_tensor + torch.randn_like(angles_tensor) * noise
            layer.ll.lbias = bias_tensor + torch.randn_like(bias_tensor) * noise

            layer.ll.psigmas.requires_grad = True
            layer.ll.angles.requires_grad = True
            layer.ll.lbias.requires_grad = True

    # ---------------------------------

    def get_biases(self):
        functions = self.function_list
        biases = []
        for _, layer in enumerate(functions):
            biases.extend(layer.get_learnable_biases())
        
        return torch.cat(biases, dim=0)
    
    def get_processed_biases(self):
        functions = self.function_list
        pbiases = []
        for _, layer in enumerate(functions):
            w = layer.get_processed_biases()
            pbiases.append(w)
        
        return torch.cat(pbiases, dim=0)
    
    def get_processed_weights(self):
        functions = self.function_list
        pweights = []
        for _, layer in enumerate(functions):
            w = layer.get_processed_weights()
            pweights.append(w)
            
        return pweights

    def get_learnable_params(self):
        functions = self.function_list
        color_params = self.color_params
        params1 = []
        params2 = []
        params3 = []
        for _, layer in enumerate(functions):
            params1.extend(layer.get_learnable_weight_params())
            params2.extend(layer.get_learnable_biases())
            params3.extend(layer.get_learnable_opacities())

        return params1, params2, params3, color_params
    
    # @property
    def optimized_weights(self):
        w = self.get_processed_weights()
        return torch.cat(w, dim=0)
    
    # @property
    def optimized_biases(self):
        return self.get_processed_biases()
    
    def optimized_function_opacities(self):
        functions = self.function_list
        ops = []
        for i, layer in enumerate(functions):
            ops.append(layer.ll.function_opacity)
        
        ops_tensor = torch.stack(ops)

        return ops_tensor
    
    # @property
    def optimized_probs(self):
        functions = self.function_list
        probs = []
        for i, layer in enumerate(functions):
            probs.append(layer.prob())
        
        probs = torch.abs(torch.tensor(probs))
        
        if self.randomize_probs:
            probs = torch.rand_like(probs)
        # else use the determinant

        normalized_probs = probs/(probs.sum())        
        return normalized_probs
    
    def postprocess_code(self, last_ifs=None):
        """
        undos pre-process_code on IFScode
        """
        m = []
        t = []
        i=0

        if last_ifs is None:
            last_ifs = self.ifs_code

        return last_ifs
    

    def create_stats_dict(self, name, ifs=None):
        '''
        name : can be "init" or "opt" (for optimized results)
        '''

        if ifs is not None:
            layer_list = ifs.contractive_functions
        else:
            layer_list = self.function_list


        data = edict()
        
        matrices = []
        vectors = []
        max_singular_value_list = []
        dets = []
        probs = []
        condition_numbers = []
        function_opacities = []

        for i in range(len(layer_list)):            
            matrix = layer_list[i].get_processed_weights()[0]

            determinant = torch.linalg.det(matrix)
            dets.append(determinant.item())
            matrices.append(matrix.tolist())
            vectors.append(layer_list[i].ll.bias.tolist())

            _, singular_values, _ = torch.linalg.svd(matrix)

            # Get the largest singular value (spectral norm)
            condition_number = singular_values.max()/singular_values.min()
            sn = torch.max(singular_values)
            # max_singular_value_list.append(sn.item())
            max_singular_value_list.extend(singular_values.tolist())
            condition_numbers.append(condition_number.item())

            # eigen_real = torch.real(torch.linalg.eigvals(matrix))
            # eigen_real = eigen_real
            # eigen_vals.append(eigen_real.tolist())

            probs.append(layer_list[i].prob())

            function_opacities.append(layer_list[i].ll.function_opacity.tolist())

        probs = torch.abs(torch.tensor(probs))        
        normalized_probs = probs/probs.sum()

        data[f"{name}_matrices"] = matrices
        data[f"{name}_vectors"] = vectors
        
        data.singular_values = max_singular_value_list
        data.determinants = dets
        data.probs = normalized_probs.tolist()
        data.condition_numbers = condition_numbers
        data.opacities = function_opacities

        return data
    
    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
