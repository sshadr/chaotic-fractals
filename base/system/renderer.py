import os, sys
import torch
import numpy as np
from cf.openGL.context import OpenGLContext
from cf.images.image_io import save_image
from cf.images.conversions import tensor_to_image, image_to_tensor
from cf.openGL.operations.resampling import SupersamplingOP

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from cf.openGL.operations.point_rasterizer import PointRasterizationOP, OpenGLPointCloud
from utils.optim_utils import make_points2d, make_points3d
from diff_isogaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.point_splatting import GaussianModel, OrthographicCam


class GaussianSplatter():
    def __init__(self, points, configs):
        super().__init__()

        # Setup
        self.init_device()

        self.coord_correction_matrix = torch.eye(3)
        self.coord_correction_matrix[1, 1] = -1
        self.coord_correction_matrix = self.coord_correction_matrix.to(self.device)
        self.splat_scale = configs.splat_scale

        self.kernel_size = configs.supersample_factor
        self.image_res = int(configs.image_res * self.kernel_size)
        self.downsampler = torch.nn.AvgPool2d(self.kernel_size, stride=self.kernel_size).to(self.device)

        # camera
        self.R = np.eye(3)
        self.T = np.array([0, 0, 0], dtype=np.float32)

        self.view_matrix = torch.eye(4).to(self.device)
        self.proj_matrix = torch.eye(4).to(self.device)

        if configs.background_color == 1:
            self.bg_color = torch.ones((3), dtype=torch.float32).to(self.device)
        else:
            self.bg_color = torch.zeros((3), dtype=torch.float32).to(self.device)

        self.scaling_modifier = self.splat_scale

        self.init_camera()
        

        self.white = torch.ones((1, 3), dtype=torch.float32)

        # Gaussians
        self.gaussians = GaussianModel()

        self.num_points = points.shape[0]
        
        self.colors_precomp = self.white.expand(self.num_points, -1).to(self.device)
        self.opacity = torch.ones((self.num_points, 1)).to(self.device)
        self.shs = None
        self.scales = (torch.tensor([1.0, 1.0, 1.0] * self.num_points, dtype=torch.float32)).to(self.device)

        self.rotations = torch.tensor([1.0, 0.0, 0.0, 0.0] * self.num_points, dtype=torch.float32).to(self.device)
        self.cov3D_precomp = None

        self.update_points(points.to(self.device))
        self.init_render_utils()
        
    def render(self, view_matrix=None):
        if view_matrix is not None:
            scale = view_matrix[:2, :2].clone().to(self.device)
            translation = view_matrix[:2, 2].clone().to(self.device)

            warped_points = torch.matmul((self.points[:, :2]), scale) - translation
            self.gaussians._xyz = make_points3d(warped_points)

        img, radii = self.rasterizer(
                                    means3D = self.gaussians._xyz,
                                    means2D = self.screenspace_points,
                                    shs = self.shs,
                                    colors_precomp = self.colors_precomp,
                                    opacities = self.opacity,
                                    scales = self.scales,
                                    rotations = self.rotations,
                                    cov3D_precomp = self.cov3D_precomp
                                    )
        # self.image is of shape (3, 512, 512)        
        # unsqueeze to make it a tensor with (batch, channels, width, height)
        img = img.unsqueeze(0)
        self.image = (self.downsampler(img))

        return self.image

    def save_render(self, path):
        save_img = self.image.clone()
        save_image(tensor_to_image(save_img), path)
    
    def update_img_res(self, res):
        # also change the relative size of the points
        prev_res = self.image_res
        self.image_res = res
        new_splat_scale = self.splat_scale * (prev_res / self.image_res)
        self.update_splat_scale(new_splat_scale)
        self.init_render_utils()
    
    def update_splat_scale(self, new_scale):
        self.splat_scale = new_scale

    def unbind(self):
        "Another dummy function"
        self.points = None

    def update_points(self, points, point_color=None):
        def correct_axis_for_points(points):
            return torch.matmul(points, self.coord_correction_matrix)
        
        if points.shape[0] != self.num_points:
            self.num_points = points.shape[0]
            self.shs = None
            self.scales = (torch.tensor([1.0, 1.0, 1.0] * self.num_points, dtype=torch.float32)).to(self.device)
            self.rotations = torch.tensor([1.0, 0.0, 0.0, 0.0] * self.num_points, dtype=torch.float32).to(self.device)
            self.cov3D_precomp = None

        if point_color is None:
            self.colors_precomp = self.white.expand(self.num_points, -1).to(self.device)
        else:
            self.colors_precomp = point_color.expand(self.num_points, -1)

        self.opacity = (points[:, 2]).unsqueeze(1) # 3rd dim
        points = make_points3d(points[:, :2]) # first 2 dims

        self.points = correct_axis_for_points(points)
        self.gaussians._xyz = self.points.to(self.device)
        self.screenspace_points = torch.zeros_like(self.gaussians.get_xyz, dtype=self.gaussians.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            self.screenspace_points.retain_grad()
        except:
            pass

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
    
    def move_camera(self, view_matrix):        
        def extend_matrix_to_4D(matrix):
            newmat = torch.eye(4)
            newmat[:2, :2] = matrix[:2, :2]
            newmat[3, :2] = matrix[:2, 2] * -1

            return newmat
        
        vm = extend_matrix_to_4D(view_matrix.clone()).to(self.device)
        
        self.camera = OrthographicCam(self.image_res, self.image_res,
                            vm
                            )
        self.init_render_utils()

    def init_camera(self):
        # Initialize a camera.
        self.camera = OrthographicCam(self.image_res, self.image_res,
                            self.view_matrix
                            )
        
    def init_render_utils(self):
        self.raster_settings = GaussianRasterizationSettings(
                                image_height=int(self.image_res),
                                image_width=int(self.image_res),
                                tanfovx=1.0,
                                tanfovy=1.0,
                                bg=self.bg_color,
                                scale_modifier=self.scaling_modifier,
                                viewmatrix=self.camera.world_view_transform,
                                projmatrix=self.proj_matrix,
                                sh_degree=0,
                                campos=self.camera.camera_center,
                                prefiltered=False,
                                debug=False # True -> writes to a .dump file if errors in diff rasterizer occur
                            )

        self.rasterizer = GaussianRasterizer(raster_settings=self.raster_settings)

def SuperSampleRender(points, base_res, factor, color=None, bg_color=0):
    img_res = (base_res * factor, base_res * factor)    
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()        
    if color is None:
        color = np.ones(3, dtype=np.float32)

    ogl_context = OpenGLContext()
    ogl_context.bind()

    
    all_points = OpenGLPointCloud(make_points2d(points))
    point_op = PointRasterizationOP(img_res)

    supersampleop = SupersamplingOP((base_res, base_res))

    fractal_super = point_op.render(all_points, color, bg_color, viewMatrix=np.eye(3, dtype=np.float32))
    
    fractal = supersampleop.render(fractal_super, factor)

    fractal_img = fractal.download_image()
    
    ogl_context.unbind()

    return fractal_img

class OglRenderer():
    def __init__(self, points, configs):
        super().__init__()
        """
        configs: are specific to the renderer (from the json file)
        points: 
        """
        self.context = OpenGLContext()
        self.context.bind()

        self.image_res = (configs.image_res, configs.image_res)
        self.radius = configs.radius
        
        self.point_color = np.array([1, 1, 1], dtype=np.float32) # white
        self.bg_color = 0 # black

        self.points = None
        self.update_points(points)
        self.update_img_res(self.image_res)
    
    def unbind(self):
        self.context.unbind()

    def update_points(self, points, color=None):   
        "color is dummy to maintain structure with the other class"

        assert points.shape[1] == 3
        if self.points is not None:
            self.points.clean_buffers()
        self.points = OpenGLPointCloud(make_points2d(points))

    def update_img_res(self, res):
        self.point_op = PointRasterizationOP(res)

    def save_render(self, path):
        save_img = self.image.clone()
        save_image(tensor_to_image(save_img), path)
    

    def render(self, view_matrix=None):
        '''
        takes in the scale and renders a scaled view...
        the scale should change the view-matrix corresponding to the scale
        Note: Warning: view_matrix should always be of type float32
        '''
        if isinstance(view_matrix, torch.Tensor):
            view_matrix = view_matrix.numpy().astype(np.float32)
        if view_matrix is None:
            view_matrix = np.eye(3).astype(np.float32)
        
        self.image_texture = self.point_op.render(self.points, self.point_color, self.bg_color, view_matrix)
        self.image = np.ascontiguousarray(self.image_texture.download_image())
        self.image = image_to_tensor(self.image, to_cuda=True)

        self.image = self.image[:, 0:3, :, :] # (1, 3, w, h)
        return self.image
     