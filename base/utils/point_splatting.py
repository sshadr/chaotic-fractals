import torch

class GaussianModel:
    '''
    The only parameter we use is the positions (_xyz)
    '''
    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)

    @property
    def get_xyz(self):
        return self._xyz
    

class OrthographicCam:
    "Simple camera"
    def __init__(self, width, height, world_view_transform):
        self.image_width = width
        self.image_height = height
        self.world_view_transform = world_view_transform

        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]