import os, sys
import numpy as np
import torch
from cf.openGL.context import *
from cf.openGL.operation import OpenGLOP

class PointRasterizationOP(OpenGLOP):

    def __init__(self, resolution, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "point_rast_simple"),
            None,
            os.path.join(this_dir, "shaders", "point_rast_simple"), 
            rendertarget_count, 
            resolution, True)
        
        self.init_uniform("viewMatrix")
        self.init_uniform("res")
        self.init_uniform("opt_color")

    
    def render(self, points, opt_color, bg_color, viewMatrix, rendertarget_id=0):        
        assert rendertarget_id < self.rendertarget_count
        output_res = self.rendertargets[rendertarget_id].color.resolution

        glViewport(0, 0, *output_res)
        
        if opt_color is None:
            opt_color = np.ones(3, dtype=np.float32).flatten()
        else:
            opt_color = opt_color.astype(np.float32).flatten()
        
        opt_color = opt_color.tolist()
        
        if viewMatrix.dtype != np.float32:
            viewMatrix = viewMatrix.astype(np.float32)
            viewMatrix = viewMatrix.flatten()
        
        glBindVertexArray(points.vao)

        # glClearColor(0., 0., 0., 1.0)
        glClearColor(bg_color, bg_color, bg_color, 1.0)
        glUseProgram(self.shader)

        glProgramUniform2i(self.shader, self.uniforms["res"], *output_res)
        glProgramUniform3f(self.shader, self.uniforms["opt_color"], *opt_color)

        glDisable(GL_POINT_SMOOTH)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)

        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glDrawBuffers(self.rendertarget_count, [GL_COLOR_ATTACHMENT0 + i for i in range(self.rendertarget_count)])
        
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)   
        
        glProgramUniformMatrix3fv(self.shader, self.uniforms["viewMatrix"], 1, True, *viewMatrix)
        
        glClear(GL_COLOR_BUFFER_BIT) #| GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_POINTS, 0, points.num_points)

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)

        glUseProgram(0) 
        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_POINT_SPRITE)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0)


        return self.rendertargets[rendertarget_id].color


class OpenGLPointCloud:
    ## a minimal class for rendering 2D points
    def __init__(self, points, data_path=None):
        try:
            num_entries = 2 # 2D positions of points
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            
            points = points.astype(np.float32)
            data = points.flatten()
            self.num_points = points.shape[0]

            if self.num_points == 0:
                self.vao = None
                self.point_buffer = None
                return

            #---------------------------------
            # upload data to the GPU

            self.vao = glGenVertexArrays(1)
            self.point_buffer = glGenBuffers(1) # buffer to hold vertices

            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.point_buffer)
            glBufferData(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW)
        
            elem_size = data[0].nbytes
            stride = elem_size * num_entries

            # position
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        except Exception as e:
            print(f"Error uploading data to the GPU: {e}")
            raise

    def clean_buffers(self):
        if self.vao is None:
            return
        
        glDeleteVertexArrays(1, [self.vao]) # 1 refers to the number of buffers
        glDeleteBuffers(1, [self.point_buffer])
      