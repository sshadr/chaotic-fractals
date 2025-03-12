import glfw
import os, sys
import math
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
import logging
from cf.openGL.window import Transform2DWindow
from cf.openGL.texture import Texture2D
from cf.openGL.operations.display import Texture2DCollageOP, ImageDisplayOP
from cf.openGL.operations.text import TextOP
from cf.openGL.operations.resampling import SupersamplingOP
from cf.images.image_io import save_image
from cf.mathematics.utils import next_power_of_2, prev_power_of_2
from cf.tools.string_tools import print_same_line
from cf.mathematics.matrix import isotropic_scaling_matrix_2D, translation_matrix_2D
from cf.openGL.operations.shapes import LineRasterizationOP
from cf.openGL.operations.point_rasterizer import PointRasterizationOP, OpenGLPointCloud

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from system.model_inference import ModelInfer, batch_point_gen
from utils.optim_utils import edict, load_json

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

class FastFractalOpsWindow(Transform2DWindow):

    def __init__(self, code_path, points_path=None, img_res=1024):
        # img_res = 512
        display_res = (img_res*2, img_res)
        super().__init__(display_res, title="FastFractal Ops Window", logging_level=logging.INFO, drag_speed=2.)
                    

        
        _, ext = os.path.splitext(code_path)
        if ext == ".json":
            self.model = ModelInfer(code_path, lf=False, naive=True)
        if ext == ".pth":
            self.model = ModelInfer(code_path, lf=False, naive=False)

        self.img_res = (img_res, img_res)
        self.supersample_factor = 4

        self.bg_color = 0
        
        # self.point_color = np.array([0.2, 0.5, 0.4], dtype=np.float32)
        self.point_color = np.array([1, 1, 1], dtype=np.float32)
        
        self.downsample_factor = 3
        self.overlay_res = (int(self.img_res[0]/self.downsample_factor), int(self.img_res[1]/self.downsample_factor))
        self.overlay_pos = (self.display_res[0] - self.overlay_res[0], self.display_res[1] - self.overlay_res[1])

        self.view_matrix = np.eye(3).astype(np.float32)

        self.tex1 = Texture2D()

        self.total_points = torch.empty((0, 2)).cuda()
        self.total_points_per_scale = int(1e8/2)

        self.total_points_target = int(1e8/2)

        self.fixed_fractal = None
        # auxillary variables
        self.save_dir = None
        self.landmarks = []
        self.trajectories = []
        self.recorded_scales = []
        self.recorded_positions = []

        self.all_points = None
        if points_path is None:
            self.points_path = None
            self.regenerate_points()
            self.all_scales = self.all_scales.to(torch.float32)
        else:
            self.points_path = points_path

            self.all_scales = torch.load(self.points_path)

        print("========== Loaded Points ==========")
        self.init_shaders()
    

    def regenerate_points(self):
        if self.all_points is not None:
            self.all_points.clean_buffers()
        self.all_scales = batch_point_gen(self.model, self.total_points_target, self.points_path)
        

    def init_shaders(self):
        super().init_shaders()
        self.text_op = TextOP((300, 50), font_size=17)
        self.text_overlay_op = ImageDisplayOP(2*(self.display_res[1],), rendertarget_count=1)

        self.texture_collage_op = Texture2DCollageOP(self.display_res, rendertarget_count=2)

        print("Total points transferred to buffer:", self.all_scales.shape[0])
        self.all_points = OpenGLPointCloud(self.all_scales) 

        rast_res = (self.img_res[0] * self.supersample_factor, self.img_res[1] * self.supersample_factor)
        self.point_op = PointRasterizationOP(rast_res)
        self.supersample_op = SupersamplingOP(self.img_res)

        self.overlay_op = ImageDisplayOP(2*(self.display_res[1],), rendertarget_count=1)
        self.overlay_line_op = ImageDisplayOP(2*(self.overlay_res[1],), rendertarget_count=1)
        self.line_render = LineRasterizationOP(self.overlay_res, rendertarget_count=1)
        

    #----------------------------------
    def clean_points_buffer(self):   
        # Clean up the buffers holding the previous points
        self.all_points.clean_buffers()
        self.all_scales = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.init_shaders()

    def update_view_matrix(self, view_matrix):
        if isinstance(view_matrix, torch.Tensor):
            self.view_matrix = view_matrix.detach().cpu().numpy()

    def render(self):

        self.fractal = self.point_op.render(self.all_points, self.point_color, self.bg_color, self.transform) # takes in numpy matrix                   
        # self.fractal_supersampled = self.supersample_op.render(self.fractal, self.supersample_factor)

        if self.fixed_fractal is None:
            self.fixed_fractal = self.fractal
            # self.fixed_fractal = self.fractal_supersampled
        
        info_text_tex = self.text_op.render(
            f"Scale: {self.scale:.2f}",
            position=(10, 12),
            background_color=(0, 0, 0, 0.5)
        )        
        display_tex = self.text_overlay_op.render(
            self.fractal,
            overlay_tex=info_text_tex, 
            overlay_pos=(self.text_overlay_op.rendertargets[0].color.resolution[0] - self.text_op.rendertargets[0].color.resolution[0], 0)
        )

        # ---------------------------------------------
        self.line_vertices = OpenGLPointCloud(self.compute_box_verts())
        lines = self.line_render.render(self.line_vertices, viewMatrix=np.eye(3), line_thickness = 1)

        overlay_lines = self.overlay_line_op.render(tex=self.fixed_fractal, overlay_tex=lines, overlay_pos=self.current_position.astype(np.uint))
              
        scale_visualize = self.texture_collage_op.render((display_tex, overlay_lines), rendertarget_id=1)
        
        return_tex = (display_tex, scale_visualize)

        return return_tex
    
        # ---------------------------------------------
    
    def invert_y_axis(self, pos):
        pos[1] *= -1

        return pos
    
    def mouse_move(self, move_position):
        if self.left_mouse_down:
            self.drag_speed = (1/self.scale) * 2  # 2 to account for [-1, 1]
            offset = (self.click_position - move_position) * self.drag_speed 
            self.current_position = self.base_position + offset / self.display_res 

    def mouse_scroll(self, sign):
        self.scale *= 1 + sign * self.zoom_speed

    @property
    def transform(self):
        if not self.is_matrixfile_loaded:
            translation = translation_matrix_2D(self.current_position)
            scaling = isotropic_scaling_matrix_2D(self.scale)
            return scaling @ translation
        else:
            flat_transform = np.float32(self.transformations[self.frame_count])
            tmatrix = flat_transform.reshape(3,3)
            self.scale = tmatrix[0, 0]
            self.current_position = tmatrix[0:2, 2]

            return tmatrix
    
    def construct_trajectories(self):
        assert len(self.landmarks) >= 2, "Need atleast 2 positions to construct trajectories"
        print("Num landmarks", self.landmarks)
        num_samples = 500
        for i in range(len(self.landmarks)-1):
            start_scale = self.recorded_scales[i]
            end_scale = self.recorded_scales[i+1]

            start_pos = self.recorded_positions[i]
            end_pos = self.recorded_positions[i+1] * end_scale

            # interpolate between scales and positions
            # tmp_start = start_scale
            # tmp_end = end_scale

            # interpolated_scales = np.linspace(tmp_start, tmp_end, 1000)
            
            # log scaling ---------------------
            log_start_scale = np.log2(start_scale)
            log_end_scale = np.log2(end_scale)
        
            # Generate 1000 points in log space
            interpolated_log_scales = np.linspace(log_start_scale, log_end_scale, num_samples, dtype=np.float64)
            # interpolated_scales = np.linspace(start_scale, end_scale, 1000, dtype=np.float64)
            
            # Convert back to linear space
            interpolated_scales = (2**interpolated_log_scales).astype(np.float64)
            # log scaling ---------------------

            interpolated_scales = np.repeat(interpolated_scales[:, np.newaxis], 2, axis=1)
            offsets = (end_pos - start_pos) * ((interpolated_scales - interpolated_scales[0]) / (interpolated_scales[-1] - interpolated_scales[0]))
            ss = np.repeat(start_pos[:, np.newaxis], num_samples, axis=1).transpose(1, 0)
            interpolated_positions = ss + offsets

            # construct transformation matrices
            for i in range(len(interpolated_scales)):
                cpos = interpolated_positions[i]
                translation = translation_matrix_2D(cpos.astype(np.float32))
                scaling = isotropic_scaling_matrix_2D(interpolated_scales[i].astype(np.float32))
                matrix = translation @ scaling
                self.trajectories.append(matrix.flatten())

    def key_press(self, key):
        super().key_press(key)
        if self.register_key((glfw.MOD_SHIFT, glfw.KEY_Q), key, "Save end landmarks to file."):

            # construct the 2 matrices:
            for i in range(len(self.landmarks)-1):
                start_scale = self.recorded_scales[i]
                end_scale = self.recorded_scales[i+1]

                start_pos = self.recorded_positions[i]
                end_pos = self.recorded_positions[i+1] * end_scale
                
                # log scaling ---------------------
                log_start_scale = np.log2(start_scale)
                log_end_scale = np.log2(end_scale)
            
                # Generate 1000 points in log space
                interpolated_log_scales = np.linspace(log_start_scale, log_end_scale, 2, dtype=np.float64)
                # interpolated_scales = np.linspace(start_scale, end_scale, 1000, dtype=np.float64)
                
                # Convert back to linear space
                interpolated_scales = (2**interpolated_log_scales).astype(np.float64)
                # interpolated_scales = np.logspace(log_start_scale, log_end_scale, 1000, base=2, dtype=np.float64)
                # log scaling ---------------------

                interpolated_scales = np.repeat(interpolated_scales[:, np.newaxis], 2, axis=1)
                offsets = (end_pos - start_pos) * ((interpolated_scales - interpolated_scales[0]) / (interpolated_scales[-1] - interpolated_scales[0]))
                ss = np.repeat(start_pos[:, np.newaxis], 2, axis=1).transpose(1, 0)
                interpolated_positions = ss + offsets

                # construct transformation matrices
                for i in range(len(interpolated_scales)):
                    cpos = interpolated_positions[i]
                    translation = translation_matrix_2D(cpos.astype(np.float32))
                    scaling = isotropic_scaling_matrix_2D(interpolated_scales[i].astype(np.float32))
                    matrix = translation @ scaling
                    self.trajectories.append(matrix.flatten())

            # ==============================================

            root = tk.Tk()
            root.withdraw()
            file_name = filedialog.asksaveasfilename(
                defaultextension = '.txt',
                filetypes = (('Text files', '*.txt'), ('All files', '*.*'))
                ) 
            if file_name:
                for idx, matrix in enumerate(self.trajectories):
                    print_same_line(f"Saving matrix {idx+1}/{len(self.trajectories)} to disk.")
                    with open(file_name, "a") as f:
                        np.savetxt(f, [matrix], delimiter=" ", fmt='%.18e')  # Exponential format for very high precision

                print("")


        if self.register_key((glfw.MOD_SHIFT, glfw.KEY_P), key, "Preview and Save trajectories."):
            if len(self.trajectories) == 0:
                self.construct_trajectories()
                self.trajectories = np.array(self.trajectories)

                # save trajectories if you want to:
                root = tk.Tk()
                root.withdraw()
                file_name = filedialog.asksaveasfilename(
                    defaultextension = '.txt',
                    filetypes = (('Text files', '*.txt'), ('All files', '*.*'))
                    ) 
                if file_name:
                    for idx, matrix in enumerate(self.trajectories):
                        print_same_line(f"Saving matrix {idx+1}/{len(self.trajectories)} to disk.")
                        with open(file_name, "a") as f:
                            # np.savetxt(f, [matrix], delimiter=" ", fmt='%1.3f')
                            np.savetxt(f, [matrix], delimiter=" ", fmt='%.18e')  # Exponential format for very high precision

                    print("")

            self.preview_trajectories = True
            preview_num_samples = 100
            spaced_indices = np.round(np.linspace(0, len(self.trajectories) - 1, preview_num_samples)).astype(int)

            self.transformations = self.trajectories[spaced_indices]
            
            
            self.is_recording_frame = not self.is_recording_frame
            self.is_matrixfile_loaded = not self.is_matrixfile_loaded
            self.preview_trajectories = False
        
        if self.register_key(glfw.KEY_B, key, "Save current trajectory."):
            self.preview_trajectories = True
            self.transformations = self.trajectories
            
            self.is_recording_frame = not self.is_recording_frame
            self.is_matrixfile_loaded = not self.is_matrixfile_loaded
            self.preview_trajectories = False
        
        if self.register_key(glfw.KEY_G, key, "Record landmarks."):
            self.landmarks.append([self.scale, self.current_position])
            self.recorded_scales.append(self.scale)
            self.recorded_positions.append(self.current_position)

            self.record_min_scale = math.floor(min(self.recorded_scales))
            self.record_max_scale = math.ceil(max(self.recorded_scales))

            print("Landmarks", self.landmarks)

        if self.register_key((glfw.MOD_SHIFT, glfw.KEY_G), key, "Reset Recorded landmarks."):
            self.landmarks = []
            self.trajectories = []
            self.recorded_scales = []
            self.recorded_positions = []
            print("Clearing Landmarks", self.landmarks)
        
        if self.register_key(glfw.KEY_PERIOD, key, "Display next power of 2."):
            self.scale = next_power_of_2(self.scale)
        
        if self.register_key(glfw.KEY_COMMA, key, "Display previous power of 2."):
            self.scale = prev_power_of_2(self.scale)
        
        if self.register_key(glfw.KEY_S, key, "Save render to file."):
            if self.save_dir is None:
                self.save_dir = filedialog.askdirectory()
                print(self.save_dir)
            save_image(self.render_tex.download_image(), f"{self.save_dir}/scale_{self.scale:.2f}.png")

    #----------------------------------

    def compute_box_verts(self, scale=None, curr_pos=None):
        # y-axis is multiplied by -1 to account for the warping
        if scale is None:
            scale = self.scale
        if curr_pos is None:
            curr_pos = self.current_position

        pos_dist_to_border = 1/scale # negative of this is the min and that is what we add in the sampling line
        
        top_right = np.array([curr_pos[0] + pos_dist_to_border, (curr_pos[1] + pos_dist_to_border)*-1])
        top_left = np.array([curr_pos[0] - pos_dist_to_border, (curr_pos[1] + pos_dist_to_border)*-1])
        bottom_right = np.array([curr_pos[0] + pos_dist_to_border, (curr_pos[1] - pos_dist_to_border)*-1])
        bottom_left = np.array([curr_pos[0] - pos_dist_to_border, (curr_pos[1] - pos_dist_to_border)*-1])
        
        # return np.concatenate([bottom_left, bottom_right, top_right, top_left], axis=0)
    
        return np.concatenate([bottom_left, bottom_right, bottom_right, top_right, top_right, top_left, top_left, bottom_left], axis=0)
    
    #----------------------------------
    

if __name__=='__main__':
    log_dir = r'I:\work\ours_compiled'
    fractal_id = 0

    code_path = os.path.join(log_dir, f"fdb_{fractal_id}.pth")

    viewer = FastFractalOpsWindow(code_path, points_path=None)
    viewer.run()
