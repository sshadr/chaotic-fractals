import json, csv
import os, sys
from easydict import EasyDict as edict
from cf.tools.train_tools import TrainingLog
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

class Parser():
    # class to simplify json file usage in the codebase
    def __init__(self, config_file, gt_config=None):
        self.file = config_file        
        self.config = load_json(self.file)

        if gt_config is not None:
            self.gt_data = gt_config
            self.config.forward = load_json(self.gt_data)
    
    def get_render_config(self):
        return self.config.renderer
    
    def get_optimizer_config(self):
        return self.config.optimizer
    
    def get_forward_config(self):
        return self.config.forward
    
    def get_logging_config(self):
        return self.config.logdirs


def prepare_output_and_logger(args):
    '''
    args: would be the Parser class. This would contain all args from the json file easily accessible through the class instance

    '''
    d = edict() # dictionary to return path info to trainer
    logging = args.get_logging_config()
    log_dir = os.path.join(os.path.join(getDir(), "log"), logging.expt_name)
    os.makedirs(log_dir, exist_ok=True)

    num_batches = args.config.optimizer.model_batches # number of batches of the model (create directories to store intermediate outputs)
    save_intermediate_batches = logging.save_intermediates
    subfolders = logging.subfolders
    
    # ---------- Creating base folders
    for _, folder in enumerate (logging.folders):
        d[f"{folder}_path"] = os.path.join(log_dir, folder)
        os.makedirs(d[f"{folder}_path"], exist_ok=True)

    # ---------

    # ---------- Sub-folders for saving model outputs (final output)
    batch = "final"
    for subfolder in subfolders:
        key = f"batch_{batch}_{subfolder}_path"
        d[key] = os.path.join(d["batch_path"], f"batch_{batch}", f"{subfolder}")
        os.makedirs(d[key], exist_ok=True)

    # ----------
    if save_intermediate_batches and num_batches > 1: 
        print(" ------ Saving intermediate batch outputs -------")     
        root_path = d["batch_path"]
        os.makedirs(root_path, exist_ok=True)
        for batch in range(num_batches):
            for subfolder in subfolders:
                key = f"batch_{batch}_{subfolder}_path"
                d[key] = os.path.join(root_path, f"batch_{batch}", f"{subfolder}")
                os.makedirs(d[key], exist_ok=True)

    shutil.copyfile(args.file, log_dir + '/config.json')
    d.log_dir = log_dir
    tb = TrainingLog(log_dir, add_unique_str=False)
    d.tb_writer = tb

    return d

def upscale_patch(img, new_view):
    '''
    img - full scale image
    new_view - view matrix looking at the patch. Corrections are made to make sure the renderer's view and upscaling process results in the same patch
    '''
    corrected_view_matrix = new_view.clone()

    corrected_view_matrix[0,0] = 1/new_view[0,0] 
    corrected_view_matrix[1,1] = 1/new_view[1,1] 
    corrected_view_matrix[:2, 2] = new_view[:2, 2] * 1/new_view[1,1]
    
    corrected_view_matrix = corrected_view_matrix[:2,:].unsqueeze(0).cuda()
    
    warp_grid = F.affine_grid(corrected_view_matrix, img.shape, align_corners=True)
    warped_tensor_bilinear = F.grid_sample(img, warp_grid, mode='bilinear', align_corners=True)    
    bilinearly_upsampled = warped_tensor_bilinear

    return bilinearly_upsampled

def sample_patch(scale=1):
    """
    takes in a scale and returns the view_matrix which is made up of R, T (camera extrinsics).
    """
    def uniform_2d(scale):
        """
        This is uniform sampling and the min and max of the box are calculated according to the scale
        the z-axis is always set to 0, since we dont want to translate z
        """
        pos_dist_to_border = 1 - scale # negative of this is the min and that is what we add in the sampling line
        box_range = 2 * pos_dist_to_border

        x = torch.rand(1) * (box_range) - pos_dist_to_border
        y = torch.rand(1) * (box_range) - pos_dist_to_border
        
        sampled_point = torch.cat([x, y, torch.zeros_like(x)])
        return sampled_point
    
    view_matrix = torch.eye(3)
    R = torch.tensor([[1/scale, 0, 0],
                                [0, 1/scale, 0],
                                [0, 0, 1]])
    
    T = uniform_2d(scale)

    view_matrix = R
    view_matrix[:2, 2] = T[:2]

    return view_matrix

def apply_mv_transform(points, return_transform = False):

    min_vals, _ = torch.min(points, dim=0)
    max_vals, _ = torch.max(points, dim=0)
    
    # 1.5 creates a padding of 25%
    sx = 1.5 / (max_vals[0] - min_vals[0])
    sy = 1.5 / (max_vals[1] - min_vals[1])

    s = torch.min(sx, sy)
    translation = (max_vals + min_vals)/2

    translated_points = points - translation
    tpoints = translated_points * s

    # print('Transformed points range:', tpoints.min().item(), tpoints.max().item())
    if return_transform:
        return tpoints, s, translation
    return tpoints

def make_points2d(points):
    return points[:, :2]

def make_points3d(points):
    '''
    concatenates extra dimension with all 1s
    '''
    return torch.cat([points, torch.ones_like(points[:, 0, None])], dim=1)

def uniform_sample1d(min, max, dim=1):
    x = torch.rand(dim) * (max-min) + min

    return x

def normalize_tensor(x, out_min=0.0, out_max=1.0):
    # return x
    in_min = torch.min(x)
    in_max = torch.max(x)
    return (out_max - out_min) / (in_max - in_min) * (x - in_min) + out_min

def random_orthogonal_matrix(device):
    """
    Function to create a random 2x2 orthogonal matrix (rotation or reflection)
    """
    theta = torch.rand(1) * 2 * torch.pi  # Random angle from [0, 2*pi)

    # Randomly choose between determinant +1 (rotation) or -1 (reflection)
    det_choice = torch.randint(0, 2, (1,))  # 0 or 1

    if det_choice == 0:
        # Rotation matrix (det = +1)
        orthogonal_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ], device=device)
    else:
        # Reflection matrix (det = -1)
        orthogonal_matrix = torch.tensor([
            [torch.cos(theta), torch.sin(theta)],
            [torch.sin(theta), -torch.cos(theta)]
        ], device=device)
    
    return orthogonal_matrix

def create_skew_symmetric_matrix(params, dim):
    device = params.device
    matrix = torch.zeros((dim, dim), device=device)

    # Fill the upper triangular part with params
    matrix[torch.triu(torch.ones(dim, dim), 1) == 1] = params.flatten()
    
    # Transpose the matrix to get elements below the main diagonal
    matrix = matrix - matrix.t()
    
    return matrix


def mean_filter(size):
    return torch.nn.AvgPool2d(size, stride=size)

def prepare_gauss_filter(kernel_size, sigma):
    def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Generate a 2D Gaussian kernel."""
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        x = x.view(1, -1)
        y = x.t()
        kernel = torch.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
        kernel_2d = kernel / kernel.sum()
        return kernel_2d

    gaussian_kernel_2d = gaussian_kernel(kernel_size, sigma).cuda()
    return gaussian_kernel_2d   

def apply_gaussian_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply a Gaussian filter to an image tensor."""
        # Add batch and channel dimensions to the kernel
        kernel = kernel.view(1, 1, kernel.size(0), kernel.size(1))
        kernel = kernel.repeat(image.size(1), 1, 1, 1)  # Repeat the kernel for each channel

        # Apply the Gaussian filter using conv2d
        filtered_image = F.conv2d(image, kernel, padding=kernel.size(2) // 2, groups=image.size(1))
        return filtered_image

def gauss_filter(x, gaussian_kernel_2d):
    filtered_image = apply_gaussian_filter(x, gaussian_kernel_2d)
    return filtered_image
  
def create_name(frame_count):
    '''
    used to pad 0's before the file name
    '''
    str_name = str(frame_count).zfill(6)

    return str_name

def load_ifs_code(code_path):
    ifs_code = torch.load(code_path)
    return ifs_code

def safe_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))

def plot_loss_curve(loss_values, loss_name, save_path, log_plot=True):
    def plot(loss_values, loss_name, save_path, log_scale=False):
        plt.figure()
        plt.plot(np.arange(len(loss_values)), loss_values, 'r-')
        if log_scale:
            loss_name = f"{loss_name}_log"
            plt.yscale('log') 
        plt.title(f'{loss_name}')
        plt.xlabel('Iteration')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"{loss_name}.png"))

    plot(loss_values, loss_name, save_path, False)
    if log_plot:
        plot(loss_values, loss_name, save_path, True)

def load_json(file_path):
    data = edict(json.load(open(file_path)))
    return data

def write_to_json(data_dict, file_name):
    with open(f"{file_name}", "w") as f:
        json.dump(data_dict, f, indent=2)

def write_to_csv(data_dict, file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)

        # Write the keys as the header row
        writer.writerow(data_dict.keys())

        # Write the values as the data rows
        writer.writerow(data_dict.values())