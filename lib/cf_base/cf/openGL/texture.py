from cf.openGL import *
import math
import numpy as np
from abc import ABC, abstractmethod

# an abstract class for an OpenGL texture
class Texture(ABC):

    @abstractmethod
    def __init__(self):
        self.handle = glGenTextures(1)
        self._resolution = (0, 0)
        self._channels = 0
        self._gl_format = None
        self.target = None
        self.need_allocation = True

    # def __del__(self):
    #     glDeleteTextures(1, [self.handle])

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if self._resolution != value:
            self.need_allocation = True
        self._resolution = value

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        if self._channels != value:
            self.need_allocation = True
        self._channels = value

    @property
    def gl_format(self):
        return self._gl_format

    @gl_format.setter
    def gl_format(self, value):
        if self._gl_format != value:
            self.need_allocation = True
        self._gl_format = value

    @property
    def max_mip_levels(self):
        return int(math.log(max(self.resolution[0], self.resolution[1]), 2)) + 1

    # bind the texture
    def bind(self):
        glBindTexture(self.target, self.handle)

    # unbind the texture
    def unbind(self):
        glBindTexture(self.target, 0)

    # set sampling parameters
    def set_params(self, min_filter=GL_NEAREST, mag_filter=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER):
        self.bind()
        glTexParameteri(self.target, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(self.target, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameterfv(self.target, GL_TEXTURE_BORDER_COLOR, [0, 0, 0, 1])
        glTexParameteri(self.target, GL_TEXTURE_WRAP_S, wrap)
        glTexParameteri(self.target, GL_TEXTURE_WRAP_T, wrap)

    # generate a linear MIP map
    def build_MIP(self):
        self.bind()
        glGenerateMipmap(self.target)

    # infer format and internal format from channel count
    @staticmethod
    def gl_format_from_channel_count(c):
        assert c > 0 and c < 5, "Channel count can only be in [1-4]"
        formats = [
            (GL_R32F, GL_RED),
            (GL_RG32F, GL_RG),
            (GL_RGB32F, GL_RGB),
            (GL_RGBA32F, GL_RGBA)
        ]
        return formats[c-1]


#=====================================================================

# an OpenGL 2D texture
class Texture2D(Texture):

    def __init__(self, image=None, flip_h=True):
        super().__init__()
        self.target = GL_TEXTURE_2D
        if image is not None:
            self.upload_image(image, flip_h=flip_h)

    # allocate GPU memory for the texture
    def allocate_memory(self, resolution, channels=4, depth_texture=False, force_allocation=False):
        self.resolution = resolution
        if depth_texture:
            self.channels = 1
            self.gl_format = GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT
        else:
            self.channels = channels
            self.gl_format = self.gl_format_from_channel_count(channels)
        if self.need_allocation or force_allocation:
            self.bind()
            glTexImage2D(self.target, 0, self.gl_format[0], resolution[0], resolution[1], 0, self.gl_format[1], GL_FLOAT, None)
            self.need_allocation = False

    # upload an image into the texture
    def upload_image(self, image, flip_h=True):
        if image.ndim == 2:
            image = image[..., None]
        w, h, self.channels = image.shape
        self.resolution = (w, h)
        if flip_h:
            image = np.flip(image, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.gl_format = self.gl_format_from_channel_count(self.channels)
        self.bind()
        glTexImage2D(self.target, 0, self.gl_format[0], self.resolution[1], self.resolution[0], 0, self.gl_format[1], GL_FLOAT, image)
        self.need_allocation = False

    # download the texture as an image
    def download_image(self, flip_h=True):
        self.bind()

        # PyOpenGL does not support downloading GL_RG textures
        # therefore: make it GL_RGB temporarily
        temp_channels = 3 if self.channels == 2 else self.channels

        _, gl_format = self.gl_format_from_channel_count(temp_channels)
        image = glGetTexImage(self.target, 0, gl_format, GL_FLOAT)
        self.unbind()
        if self.channels == 1:
            image = image[..., None]
        w, h, c = image.shape
        image = np.reshape(image, (h, w, c))

        # if temp channel was necessary, throw it away
        if not temp_channels == self.channels:
            image = image[..., 0:2]

        if flip_h:
            image = np.flip(image, 0)
        return image

#=====================================================================

# an OpenGL 2D texture array
class Texture2DArray(Texture):

    def __init__(self, image_list=None, flip_h=True):
        super().__init__()
        self._layers = 0
        self.target = GL_TEXTURE_2D_ARRAY
        self.need_allocation = True
        if image_list is not None:
            self.upload_image_list(image_list, flip_h=flip_h)
    
    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        if self._layers != value:
            self.need_allocation = True
        self._layers = value

    # allocate GPU memory for the texture array
    def allocate_memory(self, resolution, layers, channels=4, depth_texture=False, force_allocation=False):
        self.resolution = resolution
        self.layers = layers
        self.channels = channels
        if depth_texture:
            self.gl_format = GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT
        else:
            self.gl_format = self.gl_format_from_channel_count(channels)
        if self.need_allocation or force_allocation:
            self.bind()
            glTexImage3D(self.target, 0, self.gl_format[0], resolution[0], resolution[1], layers, 0, self.gl_format[1], GL_FLOAT, None)
            self.need_allocation = False
    
    # upload an image list into the texture array
    def upload_image_list(self, image_list, flip_h=True):
        self.layers = len(image_list)
        assert self.layers > 0, "Image list is empty."
        w, h, self.channels = image_list[0].shape
        self.resolution = (w, h)
        self.gl_format = self.gl_format_from_channel_count(self.channels)
        for img in image_list[1:]:
            assert img.shape == image_list[0].shape, "Image dimensions don't match."
        self.bind()
        if self.need_allocation:
            self.allocate_memory(self.resolution, self.layers, self.channels)
        up_data = np.array(image_list)
        if flip_h:
            up_data = np.flip(up_data, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage3D(self.target, 0, self.gl_format[0], self.resolution[1], self.resolution[0], self.layers, 0, self.gl_format[1], GL_FLOAT, up_data)

    # download the texture array as an image list
    def download_image(self, flip_h=True, ogl=False):
        self.bind()

        # Allocate a buffer to store the image data for all layers
        image_data = np.empty((self.layers, self.resolution[1], self.resolution[0], self.channels), dtype=np.float32)

        # Download the image data for all layers
        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, self.gl_format[1], GL_FLOAT, image_data) #.ctypes.data)
        self.unbind()
        if ogl:
            # This condition is here coz of some bug or mistake in the implementation(I cannot figure out how to smooth this out)
            # Its not nice as the user has to set this flag but we have no soln yet...
            # any debugging would involve rewriting cuda kernels but it is also the case that the data transferred is 100 percent correct....
            # so I think the mistake or limitation is within this function or the way I allocate the layout during the write in cuda but any other change in the cuda side gives a illegal memory access error.
            # In summary: the data that is being transferred using the copy_to_texture function is correct and we dont want to change that...

            layers, h, w, c = image_data.shape
            image_data = np.reshape(image_data, (layers, w, h, c))
        if flip_h:
            image_data = np.flip(image_data, 1)

        return image_data
    
    # copy list of textures into the texture array
    def insert_textures(self, textures, offset=0):
            assert len(textures) + offset <= self.layers, "Too many textures to copy into texture array with given offset."
            self.set_params()
            for idx, tex in enumerate(textures):
                assert all(a <= b for a, b in zip(tex.resolution, self.resolution)), f"Image dimensions not compatible."
                tex.set_params()
                glCopyImageSubData(
                    tex.handle, tex.target, 0, 0, 0, 0,
                    self.handle, self.target, 0, 0, 0, idx+offset,
                    tex.resolution[0], tex.resolution[1], 1)
