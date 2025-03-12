import os
from OpenGL.GL import *
from cf.openGL.render import render_screen_quad
from cf.openGL.operation import OpenGLOP
    
# reduce texture by averaging over pixels
class SupersamplingOP(OpenGLOP):

    def __init__(self, out_resolution, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"), 
            None,
            os.path.join(this_dir, "shaders", "supersampling"), 
            rendertarget_count, 
            out_resolution)
        self.init_uniform("factor")

    def render(self, texture, factor, rendertarget_id=0):
        assert rendertarget_id < self.rendertarget_count
        glViewport(0, 0, *self.rendertargets[rendertarget_id].color.resolution)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        glProgramUniform1i(self.shader, self.uniforms["factor"], factor)

        glActiveTexture(GL_TEXTURE0)
        texture.bind()
        texture.set_params()
        
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)
        render_screen_quad()

        glUseProgram(0)

        return self.rendertargets[rendertarget_id].color
