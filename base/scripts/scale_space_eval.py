# Script to generate images for evaluation in multiple scales

import os, sys
from tqdm import tqdm
import click
import numpy as np
import torch
from cf.openGL.context import OpenGLContext
from cf.openGL.texture import Texture2D
from cf.openGL.operations.resampling import SupersamplingOP
from cf.images.image_io import load_image, save_image

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from system.model_inference import ModelInfer, batch_point_gen
from utils.optim_utils import make_points2d, load_json
from cf.openGL.operations.point_rasterizer import PointRasterizationOP, OpenGLPointCloud


# Takes in the ifs code paths, save_dir
def sample_view(max_scale):
    scale = max_scale

    R = torch.tensor([[scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1]])

    def uniform_2d(scale):
        """
        This is uniform sampling and the min and max of the box are calculated according to the scale
        """
        pos_dist_to_border = 1 - 1/scale
        box_range = 2 * pos_dist_to_border

        x = torch.rand(1) * (box_range) - pos_dist_to_border
        y = torch.rand(1) * (box_range) - pos_dist_to_border
        
        sampled_point = torch.cat([x, -y, torch.zeros_like(x)])
        return sampled_point


    T = uniform_2d(scale)
    
    view_matrix = torch.eye(3)
    view_matrix[:2, :2] = R[:2, :2]
    view_matrix[:2, 2] = T[:2]

    return view_matrix.numpy()
    
class ScaleRender():
    def __init__(self, idx, method, save_root, gt_log_dir, ours_log_dir, lf32_log_dir, lf256_log_dir, moments_log_dir, evol_pcov_log_dir, cuckoo_log_dir, nr_log_dir, img_res=1024):

        self.save_root = save_root
        self.idx = idx

        if method == "all":
            self.methods = ['ours', 'lf_32', 'lf_256', 'moments', 'evol_pcov', 'cuckoo', 'nr']
        else:
            self.methods = [f'{method}']

        # Settings for ModelInfer for corresponding methods
        self.log_dirs = {
            'nr': (nr_log_dir, False, False),
            'cuckoo': (cuckoo_log_dir, False, False),
            'evol_pcov': (evol_pcov_log_dir, False, True),
            'moments': (moments_log_dir, False, True),
            'ours': (ours_log_dir, False, False),
            'lf_32': (lf32_log_dir, True, False),
            'lf_256': (lf256_log_dir, True, False),
            'gt': (gt_log_dir, False, True)
        }

        check = os.path.join(self.save_root, "views")
        os.makedirs(check, exist_ok=True)
        file_path = os.path.join(check, f"fdb_{idx}.txt")

        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Loading the matrices.")
            # Load the matrices from the file
            self.transformations = np.loadtxt(file_path)
        else:
            views = []
            views.append(np.eye(3, dtype=np.float32).flatten()) # full_scale

            scale_bins = np.array([1, 2, 4, 8])
            num_samples_per_bin = 2 

            selected_scales = []

            # Loop through each pair of adjacent bins and sample continuous values
            for i in range(len(scale_bins) - 1):
                # Generate continuous samples between the current and next bin
                samples = np.random.uniform(scale_bins[i], scale_bins[i + 1], size=num_samples_per_bin)
                selected_scales.extend(samples)

            for i in range(len(selected_scales)):
                view_matrix = sample_view(selected_scales[i])
                views.append(view_matrix.flatten())
            
            views.sort(key=lambda x: x[0])
            
            # Save the matrices to a file        
            with open(file_path, 'w') as f:
                for matrix in views:
                    np.savetxt(f, [matrix], delimiter=" ", fmt='%.18e')
        
            self.transformations = views

        self.img_res = (img_res, img_res)
        self.total_points_per_scale = int(1e8/2)
        self.supersample_factor = 8

        self.bg_color = 0
        self.point_color = np.array([1, 1, 1], dtype=np.float32)

        # counter for saving
        self.view_counter = 0
        
    def compute_box_verts(self, view_matrix=None):        
        # invert view matrix
        scale = view_matrix[0, 0]

        tx = view_matrix[0, 2] # Translation x
        ty = view_matrix[1, 2] # Translation y

        curr_pos = np.array([tx, -ty]) / (scale)
        
        pos_dist_to_border = 1/scale # negative of this is the min and that is what we add in the sampling line
        
        top_right = np.array([curr_pos[0] + pos_dist_to_border, (curr_pos[1] + pos_dist_to_border)*1])
        top_left = np.array([curr_pos[0] - pos_dist_to_border, (curr_pos[1] + pos_dist_to_border)*1])
        bottom_right = np.array([curr_pos[0] + pos_dist_to_border, (curr_pos[1] - pos_dist_to_border)*1])
        bottom_left = np.array([curr_pos[0] - pos_dist_to_border, (curr_pos[1] - pos_dist_to_border)*1])
        
        return np.array([bottom_left, bottom_right, top_right, top_left])


    def add_points_for_scale(self, view_matrix):
        """
        Generate points for a given scale by applying a view matrix and filtering valid points.
        """

        scale = view_matrix[0, 0]
        box_verts = self.compute_box_verts(view_matrix)

        x_min = np.min(box_verts[:, 0])
        x_max = np.max(box_verts[:, 0])
        y_min = np.min(box_verts[:, 1])
        y_max = np.max(box_verts[:, 1])

        # quick init for concat. if points are not in view, they get out
        total_points = self.all_scales[0:1]
        # ==================
        # ==================

        total_points_target_in_millions = self.total_points_per_scale / 1e6
        
        with tqdm(total=total_points_target_in_millions, desc="Generating Points", unit=" M points") as pbar:
            pbar.update(total_points.shape[0] / 1e6)
            
            while total_points.shape[0] < self.total_points_per_scale:
                # Generate a batch of points
                points = self.model.forward()
                points = make_points2d(points)

                # Filter points within bounds
                valid_points_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)                
                valid_points = points[valid_points_mask]

                # --------------------
                # Update the progress bar with the number of new points
                pbar.update(int(valid_points.shape[0])/1e6)

                # Concatenate valid points to the total points
                total_points = torch.cat((total_points, valid_points), dim=0)
                # points generated using IFS is unique every iteration
                
                if total_points.shape[0] >= self.total_points_per_scale:
                    total_points = total_points[:self.total_points_per_scale, :]
                    break
        
        print("Scale, Generated num points:", scale, total_points.shape)
        
        self.all_scales = total_points

        upload_points = True
        return upload_points
    
    def render_scales(self):
        context = OpenGLContext()
        context.bind()
        
        rast_res = (self.img_res[0] * self.supersample_factor, self.img_res[1] * self.supersample_factor)
        self.context_point_op = PointRasterizationOP(rast_res)
        self.context_supersample_op = SupersamplingOP(self.img_res)

        # print("Total points transferred to buffer:", self.all_scales.shape[0])
        context_pc = OpenGLPointCloud(self.all_scales) 

        for i in range(len(self.transformations)):                
            matrix = self.transformations[i].reshape(3, 3).astype(np.float32)
            upload_points = self.add_points_for_scale(matrix)

            if upload_points:
                context_pc.clean_buffers() # cleanup --delete previous buffers, etc
                print("Total points transferred to buffer:", self.all_scales.shape[0])
                context_pc = OpenGLPointCloud(self.all_scales)
                
            if context_pc.vao is None:
                context_fractal_supersampled = Texture2D(np.zeros((self.img_res[0], self.img_res[1], 4)))
            else:
                context_fractal = self.context_point_op.render(context_pc, self.point_color, self.bg_color, matrix)
                context_fractal_supersampled = self.context_supersample_op.render(context_fractal, self.supersample_factor)            

            # save rendered image
            save_image(context_fractal_supersampled.download_image(), f"{ self.render_save_path}/fdb_{self.idx}_{self.view_counter}.png")
            self.view_counter += 1
                
        print("Finished generating images for all views")
        context.unbind()
    
    def render_methods(self):
        # for should iterate through methods and run render_scales for each method
        for method in self.methods:
            self.view_counter = 0  # Reset the view counter for each method

            # Load the model depending on the method
            if method == "ours":
                name = self.log_dirs[f'{method}'][0].split("/log/")[1].split("/")[0]
                self.render_save_path = os.path.join(self.save_root, f"{method}_{name}")
                os.makedirs(self.render_save_path, exist_ok=True)

                code_path = os.path.join(self.log_dirs['ours'][0], f"fdb_{self.idx}", "output", "best_optimized_ifs_code.pth")
                self.model = ModelInfer(code_path, lf=False, naive=False)
            
            if method == "lf_32":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                lf_ifs_code = os.path.join(self.log_dirs['lf_32'][0], f"fdb_{self.idx}", "iter3000_opti_ifs_code.pth")
                self.model = ModelInfer(optimized_code_path=lf_ifs_code, lf=True, naive=False)

            if method == "lf_256":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                lf_ifs_code = os.path.join(self.log_dirs['lf_256'][0], f"fdb_{self.idx}", "iter3000_opti_ifs_code.pth")
                self.model = ModelInfer(optimized_code_path=lf_ifs_code, lf=True, naive=False)
            
            if method == "gt":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                gt_code_path = os.path.join(self.log_dirs['gt'][0], f"fdb_{self.idx}.json")
                gt_code = load_json(gt_code_path)
                self.model = ModelInfer(gt_code, lf=False, naive=True)

            if method == "moments":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                code_path = os.path.join(self.log_dirs['moments'][0], f"fdb_{self.idx}", "output", "best_optimized_ifs_code.pth")
                self.model = ModelInfer(code_path, lf=False, naive=False)

            if method == "evol_pcov":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                code_path = os.path.join(self.log_dirs['evol_pcov'][0], f"fdb_{self.idx}.json")
                evol_code = load_json(code_path)
                self.model = ModelInfer(evol_code, lf=False, naive=True)
            
            if method == "cuckoo":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                code_path = os.path.join(self.log_dirs['cuckoo'][0], f"fdb_{self.idx}.json")
                code = load_json(code_path)
                self.model = ModelInfer(code, lf=False, naive=True)
        
            if method == "nr":
                self.render_save_path = os.path.join(self.save_root, f"{method}")
                os.makedirs(self.render_save_path, exist_ok=True)

                code_path = os.path.join(self.log_dirs['nr'][0], f"fdb_{self.idx}.json")
                code = load_json(code_path)
                self.model = ModelInfer(code, lf=False, naive=True)

            # Call render_scales to render the views using the loaded model
            self.all_scales = batch_point_gen(self.model, int(1e8/2), None)            
            self.render_scales() # has a loop over views

            del self.all_scales

        

@click.command()
@click.option('--idx', help='image index', required=True)
@click.option('--save_root', help='path to save', required=True)
@click.option('--gt_log_dir', help='path to ifs code optimized', required=True)
@click.option('--ours_log_dir', help='path to ifs code optimized', required=True)
@click.option('--lf32_log_dir', help='path to ifs code optimized', required=True)
@click.option('--lf256_log_dir', help='path to ifs code optimized', required=True)
@click.option('--moments_log_dir', help='path to ifs code optimized', required=True)
@click.option('--evol_pcov_log_dir', help='path to ifs code optimized', required=True)
@click.option('--cuckoo_log_dir', help='path to ifs code optimized', required=True)
@click.option('--nr_log_dir', help='path to ifs code optimized', required=True)
@click.option('--method', 
              help='Which method to run (string)', 
              required=True, 
              type=click.Choice(['ours', 'lf_32', 'lf_256', 'moments', 'evol_pcov', 'cuckoo', 'nr', 'all'], case_sensitive=False))

def main(idx, method, save_root, gt_log_dir, ours_log_dir, lf32_log_dir, lf256_log_dir, moments_log_dir, evol_pcov_log_dir, cuckoo_log_dir, nr_log_dir):

    cool = ScaleRender(idx, method, save_root, gt_log_dir, ours_log_dir, lf32_log_dir, lf256_log_dir, moments_log_dir, evol_pcov_log_dir, cuckoo_log_dir, nr_log_dir)    
    cool.render_methods()
    
    # rendered images save in save_root/method

if __name__ == "__main__":
    main()
    print("\n === TERMINATED ===")