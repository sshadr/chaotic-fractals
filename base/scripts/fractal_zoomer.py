# script to generate fractal at multiple scales given the camera path as input

import os, sys
from tqdm import tqdm
import click
import time

import numpy as np
import torch
from cf.openGL.context import OpenGLContext
from cf.openGL.texture import Texture2D
from cf.openGL.operations.display import ImageDisplayOP
from cf.openGL.operations.text import TextOP
from cf.openGL.operations.resampling import SupersamplingOP
from cf.images.image_io import save_image, create_video
from cf.openGL.operations.point_rasterizer import PointRasterizationOP, OpenGLPointCloud
from cf.openGL.operations.shapes import SimplePointRasterizationOP, LineRasterizationOP

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from system.model_inference import ModelInfer, batch_point_gen
from utils.optim_utils import edict, make_points2d, load_json

fern3 = edict({
    "ifs_m" : [
                [[0.85, 0.04], [-0.04, 0.85]],
                [[0.20, -0.26], [0.23, 0.22]],
                [[-0.15, 0.28], [0.26, 0.24]]
            ],
    "ifs_t" : [[0.0, 0.4], [0.0, 0.4], [0.0, 0.11]],
    "ifs_p" : [0.5, 0.2, 0.2]
}
)

fern = edict({
    "ifs_m" : [
                [[0.0, 0.0], [0.0, 0.16]],
                [[0.85, 0.04], [-0.04, 0.85]],
                [[0.20, -0.26], [0.23, 0.22]],
                [[-0.15, 0.28], [0.26, 0.24]]
            ],
    "ifs_t" : [[0.0, 0.0], [0.0, 0.4], [0.0, 0.4], [0.0, 0.11]],
    "ifs_p" : [0.1, 0.5, 0.2, 0.2]
}
)

sierpinski = edict({
    "ifs_m" : [
                [[0.5, 0], [0.0, 0.5]],
                [[0.5, 0], [0.0, 0.5]],
                [[0.5, 0], [0.0, 0.5]],                
            ],
    "ifs_t" : [[0.0, 0.0], [0.5, 0.0], [0.25, 0.433]],
    "ifs_p" : [0.5, 0.2, 0.2]
}
)

class OffRender():
    def __init__(self, method, trajectory_file, ours_log_dir, lf32_log_dir, lf256_log_dir, moments_log_dir, evol_pcov_log_dir, gt_log_dir, cuckoo_log_dir, nr_log_dir, resume_idx, img_res=1024):

        self.transformations = np.loadtxt(trajectory_file, delimiter=' ', dtype=np.float64)

        # resume 
        self.resume_idx = 0
        if resume_idx > 0:
            self.resume_idx = int(resume_idx)
            print("Resuming from index:", int(resume_idx))
            self.transformations = np.vstack((self.transformations[0], self.transformations[int(resume_idx):, :]))
        
        self.render_save_path = os.path.dirname(trajectory_file)
        self.render_save_path = os.path.join(self.render_save_path, f"{method}")
        os.makedirs(self.render_save_path, exist_ok=True)

        self.bbox_save_path = os.path.join(self.render_save_path, "bbox")
        os.makedirs(self.bbox_save_path, exist_ok=True)

        if method == "ours":
            code_path = os.path.join(ours_log_dir, f"best_optimized_ifs_code.pth")
            self.model = ModelInfer(code_path, lf=False, naive=False)
        
        if method == "lf_32":
            lf_ifs_code = lf32_log_dir
            self.model = ModelInfer(optimized_code_path=lf_ifs_code, lf=True, naive=False)
            
        if method == "lf_256":
            lf_ifs_code = lf256_log_dir
            self.model = ModelInfer(optimized_code_path=lf_ifs_code, lf=True, naive=False)
        
        if method == "gt":
            gt_code = load_json(gt_log_dir)
            self.model = ModelInfer(gt_code, lf=False, naive=True)

        if method == "moments":
            code_path = os.path.join(moments_log_dir, "best_optimized_ifs_code.pth")
            self.model = ModelInfer(code_path, lf=False, naive=False)

        if method == "evol_pcov":
            code_path = load_json(evol_pcov_log_dir)
            self.model = ModelInfer(code_path, lf=False, naive=True)
        
        if method == "cuckoo":
            code_path = load_json(cuckoo_log_dir)
            self.model = ModelInfer(code_path, lf=False, naive=True)
        
        if method == "nr":        
            code_path = load_json(nr_log_dir)
            self.model = ModelInfer(code_path, lf=False, naive=True)
                

        self.img_res = (img_res, img_res)
        self.all_scales = batch_point_gen(self.model, int(1e8/2), None)
        
        self.total_points_per_scale = int(1e8/2)
        self.supersample_factor = 8

        self.bg_color = 0
        self.point_color = np.array([1, 1, 1], dtype=np.float32)
        
        line_res = (self.img_res[0], self.img_res[1])
        self.line_render = LineRasterizationOP(line_res, rendertarget_count=1)
        self.bbox_point_render = SimplePointRasterizationOP(line_res, rendertarget_count=1)
        self.overlay_line_op = ImageDisplayOP(2*(self.img_res[1],), rendertarget_count=1)
        

#---------------------------------- 
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

        # filter existing points first
        total_points = self.all_scales


        valid_points_mask = (total_points[:, 0] >= x_min) & (total_points[:, 0] <= x_max) & \
                            (total_points[:, 1] >= y_min) & (total_points[:, 1] <= y_max)

        total_points = total_points[valid_points_mask]
        # ==================
        # ==================

        if total_points.shape[0] == 0:
            # skip this by saying no points will land in this window
            self.all_scales = total_points
            upload_points = True
            return upload_points
        
        # Check how many more points are needed
        points_needed = self.total_points_per_scale - total_points.shape[0]
        
        if points_needed <= 0:
            print("Already have enough points.")
            upload_points = False
            return upload_points

        total_points_target_in_millions = self.total_points_per_scale / 1e6
        time_limit = 20 * 60  # convert minutes to seconds
        start_time = time.time()  # Record the start time
        # if the following loop doesnt finish in 20 mins, break
        
        with tqdm(total=total_points_target_in_millions, desc="Generating Points", unit=" M points") as pbar:
            pbar.update(total_points.shape[0] / 1e6)
            
            while total_points.shape[0] < self.total_points_per_scale:
                # Check the elapsed time
                elapsed_time = time.time() - start_time
                print("Elapsed Time: ", elapsed_time/60)
                if elapsed_time > time_limit:
                    print(f"Time limit of {time_limit/60} minutes exceeded.")
                    break

                # Generate a batch of points
                points = self.model.forward()
                points = make_points2d(points)

                # Filter points within bounds
                
                valid_points_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)                
                valid_points = points[valid_points_mask]

                # --------------------                
                points_needed = self.total_points_per_scale - total_points.shape[0]
                if points_needed > 0:
                    if valid_points.shape[0] > points_needed:
                        valid_points = valid_points[:points_needed, :]  # Clip to the number of needed points                
                    total_points = torch.cat((total_points, valid_points), dim=0)
                
                # Update the progress bar with the number of new points
                pbar.update(int(valid_points.shape[0])/1e6)
                
                del points
                del valid_points
                torch.cuda.empty_cache()  # Free up unused memory

                # points generated using IFS is unique every iteration
                
                if total_points.shape[0] >= self.total_points_per_scale:
                    break
        
        print("Scale, Generated num points:", scale, total_points.shape)
        
        self.all_scales = total_points

        upload_points = True
        return upload_points

    def draw_bbox(self, input_tex, matrix):

        # bbox computation:
        box_verts = self.compute_box_verts(matrix)

        center = np.mean(box_verts, axis=0)
        center = center.flatten().astype(np.uint)
        
        line_verts = np.concatenate([
                    box_verts[0], box_verts[1],  # bottom_left to bottom_right
                    box_verts[1], box_verts[2],  # bottom_right to top_right
                    box_verts[2], box_verts[3],  # top_right to top_left
                    box_verts[3], box_verts[0]   # top_left to bottom_left
                ], axis=0)
        
        line_verts = OpenGLPointCloud(line_verts)

        # ==========================
        
        verts = OpenGLPointCloud(box_verts)
        radius = 3
        
        lines = self.line_render.render(line_verts, viewMatrix=np.eye(3), line_thickness = 1)
        # points = self.bbox_point_render.render(verts, radius=radius, viewMatrix=np.eye(3, dtype=np.float32))
        overlay_lines = self.overlay_line_op.render(tex=input_tex, overlay_tex=lines, overlay_pos=center, alpha_composite_overlay=False)
        
        line_verts.clean_buffers()
        verts.clean_buffers()

        return overlay_lines


    def render_scales(self):
        print("Num generations:", len(self.transformations))
        scales = [self.transformations[i].reshape(3, 3)[0, 0] for i in range(len(self.transformations))]

        # --------------------
        # If intermediate scales need to be rendered - use the followinig block to pick random scales
        # Define the specific ranges you want to pick from (31-35, 64-68)
        ranges = [(6, 9), (31, 35)]
        # Initialize the selected_scales list
        selected_scales = []
        
        # Loop through each range and pick two random scales from the corresponding range
        for lower_bound, upper_bound in ranges:
            # Filter scales that fall within the current range
            range_scales = [s for s in scales if lower_bound <= s <= upper_bound]
            selected_scales.extend(np.random.choice(range_scales, 1, replace=False))
            
        # range picked for real images [1x, 7x, 31x, 104x(the last:scales[-1])]- Fig.6
        selected_scales.extend([1, scales[-1]])
        # --------------------
        # --------------------
        # Just the first and the last scale 
        selected_scales = [1, scales[-1]] # Fig.5 
        selected_scales.sort()
        
        # --------------------
        # selected_transformations is used to compute bounding box of the current view on the previous captured view
        selected_transformations = []

        # Loop through each scale in selected_scales and find the corresponding transformation
        for selected_scale in selected_scales:
            if selected_scale in scales:
                # Find the index of the selected scale in the scales array
                index = scales.index(selected_scale)
                # Append the corresponding transformation to selected_transformations
                selected_transformations.append(self.transformations[index])

        # Convert selected_transformations into a numpy array to save it in the desired format
        selected_transformations = np.array(selected_transformations)
        # Save the transformations to box.txt in the same format as the original file
        np.savetxt(os.path.join(self.bbox_save_path, 'box.txt'), selected_transformations, delimiter=' ', fmt='%.18e')

        print("Selected scales from each bin:", selected_scales)
        # --------------------
        # --------------------
        # Start Render
        context = OpenGLContext()
        context.bind()
        
        supersample_factor = self.supersample_factor
        rast_res = (self.img_res[0] * supersample_factor, self.img_res[1] * supersample_factor)
        self.context_point_op = PointRasterizationOP(rast_res)
        self.context_supersample_op = SupersamplingOP(self.img_res)

        context_pc = OpenGLPointCloud(self.all_scales) 

        context_text_op = TextOP((300, 50), font_size=22)
        context_text_overlay_op = ImageDisplayOP(2*(self.img_res[1],), rendertarget_count=1)

        # aux for bbox 
        prev_captured_image = None
        prev_captured_scale = None

        for i in range(len(self.transformations)):
            i = i + self.resume_idx              
            matrix = self.transformations[i].reshape(3, 3).astype(np.float32)
            scale = matrix[0, 0]

            upload_points = self.add_points_for_scale(matrix)

            if upload_points:
                context_pc.clean_buffers() # cleanup --delete previous buffers, etc
                del context_pc                
                print("Total points transferred to buffer:", self.all_scales.shape[0])
                context_pc = OpenGLPointCloud(self.all_scales)
                
            if context_pc.vao is None:
                context_fractal_supersampled = Texture2D(np.zeros((self.img_res[0], self.img_res[1], 4)))
            else:
                context_fractal = self.context_point_op.render(context_pc, self.point_color, self.bg_color, matrix)
                context_fractal_supersampled = self.context_supersample_op.render(context_fractal, self.supersample_factor)

            if scale == 1:
                prev_captured_image = Texture2D(context_fractal_supersampled.download_image())
                prev_captured_scale = scale
                prev_matrix = np.eye(3, dtype=np.float32)

                # 
                prev_matrix_inv = np.linalg.inv(prev_matrix)
                scale_mat = self.transformations[-1].reshape(3, 3).astype(np.float32)
                new_matrix = prev_matrix_inv @ scale_mat
                overlayed_image = self.draw_bbox(prev_captured_image, new_matrix)
                save_image(overlayed_image.download_image(), f"{ self.bbox_save_path}/scale_{prev_captured_scale:.2f}.png")

            if scale in selected_scales:
                # get bbox and draw on prev_captured_image
                prev_matrix_inv = np.linalg.inv(prev_matrix)
                new_matrix = prev_matrix_inv @ matrix
                overlayed_image = self.draw_bbox(prev_captured_image, new_matrix)
                save_image(overlayed_image.download_image(), f"{ self.bbox_save_path}/scale_{prev_captured_scale:.2f}.png")
                save_image(prev_captured_image.download_image(), f"{ self.bbox_save_path}/scale_{prev_captured_scale:.2f}_nobbox.png")

                prev_captured_image = Texture2D(context_fractal_supersampled.download_image())
                prev_captured_scale = scale
                prev_matrix = matrix
                
            if scale == scales[-1]:
                save_image(context_fractal_supersampled.download_image(), f"{ self.bbox_save_path}/scale_{scale}.png")

                
            info_text_tex = context_text_op.render(
                f"Scale: {scale:.2f}",
                position=(10, 12),
                color =[1.0, 0.647, 0.0],
                background_color=(0, 0, 0, 0.5)
            )        
            display_tex = context_text_overlay_op.render(
                context_fractal_supersampled,
                overlay_tex=info_text_tex, 
                overlay_pos=(context_text_overlay_op.rendertargets[0].color.resolution[0] - context_text_op.rendertargets[0].color.resolution[0], 0)
            )

            # save rendered image
            save_image(display_tex.download_image(), f"{ self.render_save_path}/{i:05}.png")
                
        print("Finished generating images along trajectory")
        context.unbind()
        
        create_video(self.render_save_path, self.render_save_path)
        

@click.command()
@click.option('--gt_log_dir', help='path to ifs code optimized', required=True)
@click.option('--ours_log_dir', help='path to ifs code optimized', required=True)
@click.option('--lf32_log_dir', help='path to ifs code optimized', required=True)
@click.option('--lf256_log_dir', help='path to ifs code optimized', required=True)
@click.option('--moments_log_dir', help='path to ifs code optimized', required=True)
@click.option('--evol_pcov_log_dir', help='path to ifs code optimized', required=True)
@click.option('--cuckoo_log_dir', help='path to ifs code optimized', required=True)
@click.option('--nr_log_dir', help='path to ifs code optimized', required=True)
@click.option('--trajectory_file', help='Groundtruth image name (string)', required=True)
@click.option('--method', 
              help='Which method to run (string)', 
              required=True, 
              type=click.Choice(['ours', 'lf_32', 'lf_256', 'moments', 'evol_pcov', 'cuckoo', 'nr', 'gt'], case_sensitive=False))

@click.option('--resume_idx', help='which method to run(string)', type=int, required=True)
def main(method, trajectory_file, ours_log_dir, lf32_log_dir, lf256_log_dir, moments_log_dir, evol_pcov_log_dir, gt_log_dir, cuckoo_log_dir, nr_log_dir, resume_idx):
    
    cool = OffRender(method, trajectory_file, ours_log_dir, lf32_log_dir, lf256_log_dir, moments_log_dir, evol_pcov_log_dir, gt_log_dir, cuckoo_log_dir, nr_log_dir, resume_idx)
    cool.render_scales()
    # creates a video after it finishes. where: in the same directory as the trajectory file

if __name__ == "__main__":
    main()
    print("\n === TERMINATED ===")