import os, sys
import random
import torch
import torch.nn as nn
from utils.fractal_modules import sample_functions
from utils.optim_utils import uniform_sample1d, create_skew_symmetric_matrix, random_orthogonal_matrix

class LipschtizLinear(nn.Module):
    """

    """
    def __init__(self, in_features, out_features, parameterization, init_scheme):
        super().__init__()
        self.init_device()
        self.in_features = in_features
        self.out_features = out_features

        self.skew_width = self.in_features * (self.in_features - 1) // 2 # this is used for sending the corresponding params from self.angles for builiding the right matrix

        self.use_matrix_exp = False
        self.use_qr = False
        self.use_angle_param = False
        self.use_naive = False

        if parameterization == 'naive':
            self.use_naive = True
        if parameterization == 'mat_exp':
            self.use_matrix_exp = True
        if parameterization == 'qr':
            self.use_qr = True
        if parameterization == 'angle':
            self.use_angle_param = True

        if self.use_naive:
            self.matrix = torch.eye(self.in_features, device=self.device)
            self.matrix.requires_grad = True

        if self.use_matrix_exp:
            self.uv_params = torch.zeros(2*self.skew_width).to(self.device)
            self.uv_params.requires_grad = True
        
        if self.use_qr:
            
            self.au = random_orthogonal_matrix(self.device)
            self.av = random_orthogonal_matrix(self.device)

            self.au.requires_grad = True
            self.av.requires_grad = True
        
        if self.use_angle_param:
            self.angles = uniform_sample1d(0, 360, 2) * torch.pi/180
            self.angles.requires_grad = True
        
        if init_scheme == "isotropic":
            self.psigmas = torch.ones(self.in_features, device=self.device)*1
            self.lbias = torch.zeros(self.in_features).to(self.device)

        if init_scheme == "random":
            self.psigmas = torch.rand(self.in_features, device=self.device)
            self.lbias = (torch.rand(self.in_features, device=self.device) * 2) - 1            
            
        self.psigmas.requires_grad = True
        self.lbias.requires_grad = True

        self.opacity = (torch.tensor(1.0, dtype=torch.float32)).to(self.device)
        self.opacity.requires_grad = False

    @property
    def bias(self):
        return torch.tanh(self.lbias)
    
    @property
    def function_opacity(self):
        op = torch.clamp(self.opacity, 0.0, 1.0)
        return op
        # return torch.sigmoid(self.opacity)
    
    @property
    def sigmas(self):
        if self.use_naive:
            _, sigmas, _ = torch.svd(self.matrix)
            return sigmas
        return torch.sigmoid(self.psigmas)
    
    def weights(self):
        if self.use_naive:
            return self.matrix
        
        if self.use_matrix_exp:
            Au = create_skew_symmetric_matrix(params= self.uv_params[0:self.skew_width], dim=self.in_features)
            Av = create_skew_symmetric_matrix(params= self.uv_params[self.skew_width:2*self.skew_width], dim=self.in_features)

            U = torch.linalg.matrix_exp(Au).to(self.device)
            V = torch.linalg.matrix_exp(Av).to(self.device)
            V_transposed = torch.transpose(V, 1, 0)
        
        elif self.use_qr:
            # QR parameterization -------------------------

            U, _ = torch.linalg.qr(self.au)
            V_transposed, _ = torch.linalg.qr(self.av)

            # QR parameterization -------------------------

        elif self.use_angle_param:
            # angle_param
            theta, phi = self.angles[0], self.angles[1]
            U = torch.stack([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)]).view(2, 2).to(self.device)
            V_transposed = torch.stack([torch.cos(phi), torch.sin(phi), -torch.sin(phi), torch.cos(phi)]).view(2, 2).to(self.device)
        

        S = torch.diag(self.sigmas)

        w = (U @ S @ V_transposed)
        
        return w
    
    
    def get_parameters(self):
        if self.use_naive:
            return {
                "weight": [self.matrix],
                "bias": [self.lbias],
                "opacity": [self.opacity]
            }
        if self.use_matrix_exp:
            return {
            "weight": [self.psigmas, self.uv_params],
            "bias": [self.lbias],
            "opacity": [self.opacity]
         }
        if self.use_qr:
            return {
                "weight": [self.psigmas, self.au, self.av],
                "bias": [self.lbias],
                "opacity": [self.opacity]
            }
        if self.use_angle_param:
            return {
                "weight": [self.psigmas, self.angles],
                "bias": [self.lbias],
                "opacity": [self.opacity]
            }
        
    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")


class ContractiveFunction(nn.Module):
    """
    Storage class for a contractive function. 
    It is initialized in the Model class and the model is constructed 
    using the weights of the transformation layers inside this class
    """
    def __init__(self, parameterization, init_scheme, ndims=None):
        super().__init__()
        if ndims is None:
            self.ndims = 2
        else:
            self.ndims = ndims
            
        self.ll = LipschtizLinear(2, 2, parameterization=parameterization, init_scheme=init_scheme)
        self.device = self.ll.device

    # @property
    def prob(self):
        self._prob = 1 # equal probability (normalized later w.r.t num functions) 
        # if determinant as prob
        # w = self.ll.weights()
        # self._prob = torch.linalg.det(w)
        return self._prob

    def get_learnable_weight_params(self):
        params = []
        params.extend(self.ll.get_parameters()["weight"])
        return params
    
    def get_learnable_biases(self):
        params = []
        params.extend(self.ll.get_parameters()["bias"])
        return params
    
    def get_learnable_opacities(self):
        params = []
        params.extend(self.ll.get_parameters()["opacity"])
        return params
                
    def get_processed_biases(self):
        ls = (self.ll.bias.unsqueeze(0).unsqueeze(-1))        
        return ls
    
    def get_processed_weights(self):
        ls = (self.ll.weights().unsqueeze(0))
        return ls

class ParallelIFS(nn.Module):
    def __init__(self, model_batches,  num_points, randomize_sequence):
        super().__init__()
        self.init_device()
        self.model_batches = model_batches
        self.ori_num_points = num_points
        self.remove_points = 10 * self.model_batches

        self.tr_iter = 0

        self.randomize_sequence = randomize_sequence
        if self.randomize_sequence:
            print("================= Randomizing sequence ================= ")
        
    
    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")  

    def forward(self, point, optimized_weights, optimized_biases, optimized_function_ops, code, sample_always=True):
        self.code = code
        # Batched matrix multiplication

        if self.randomize_sequence:
            self.num_points = random.randint(self.ori_num_points-100, self.ori_num_points+100)
            remove = random.randint(10-3, 10+3)
            self.remove_points = remove * self.model_batches # warm-up phase
        else:
            self.num_points = self.ori_num_points
            # warm up is fixed to 10*model_batches as well

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

        # points is of shape [self.model_batches, 3]...first 2 are the actual coordinates and the third is the function opacity
        
        return points