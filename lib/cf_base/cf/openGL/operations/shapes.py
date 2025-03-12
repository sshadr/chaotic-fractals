from cf.openGL.context import *
from cf.openGL.operation import OpenGLOP
import os
import numpy as np
 
class SimplePointRasterizationOP(OpenGLOP):

    def __init__(self, resolution, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"), 
            None,
            os.path.join(this_dir, "shaders", "red"), 
            rendertarget_count, 
            resolution, True)
        
        self.init_uniform("viewMatrix")
        self.init_uniform("res")
        self.init_uniform("radius")

    
    def render(self, points, radius, viewMatrix, rendertarget_id=0):        
        assert rendertarget_id < self.rendertarget_count
        output_res = self.rendertargets[rendertarget_id].color.resolution

        glViewport(0, 0, *output_res)
        
        if viewMatrix.dtype != np.float32:
            viewMatrix = viewMatrix.astype(np.float32)

        glBindVertexArray(points.vao)

        # glClearColor(0., 0., 0., 1.0)
        glUseProgram(self.shader)

        glProgramUniform2i(self.shader, self.uniforms["res"], *output_res)
        glProgramUniform1f(self.shader, self.uniforms["radius"], radius)

        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        glPointSize(radius)

        glDrawBuffers(self.rendertarget_count, [GL_COLOR_ATTACHMENT0 + i for i in range(self.rendertarget_count)])
        
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)   
        
        glProgramUniformMatrix3fv(self.shader, self.uniforms["viewMatrix"], 1, True, *viewMatrix)
        glViewport(0, 0, *self.rendertargets[rendertarget_id].color.resolution)
        
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
    

class LineRasterizationOP(OpenGLOP):

    def __init__(self, resolution, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"), 
            None,
            os.path.join(this_dir, "shaders", "red"), 
            rendertarget_count, 
            resolution, True)
        self.init_uniform("viewMatrix")

    
    def render(self, points, viewMatrix, line_thickness, rendertarget_id=0):        
        assert rendertarget_id < self.rendertarget_count
        if viewMatrix.dtype != np.float32:
            viewMatrix = viewMatrix.astype(np.float32)

        glBindVertexArray(points.vao)
        glUseProgram(self.shader)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_MULTISAMPLE)
        glLineWidth(line_thickness)

        glDrawBuffers(self.rendertarget_count, [GL_COLOR_ATTACHMENT0 + i for i in range(self.rendertarget_count)])
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)   
        
        glProgramUniformMatrix3fv(self.shader, self.uniforms["viewMatrix"], 1, True, *viewMatrix)
        glViewport(0, 0, *self.rendertargets[rendertarget_id].color.resolution)
        glActiveTexture(GL_TEXTURE3)
        glClear(GL_COLOR_BUFFER_BIT) #| GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_LINES, 0, points.num_points)
        
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)

        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0)

        return self.rendertargets[rendertarget_id].color