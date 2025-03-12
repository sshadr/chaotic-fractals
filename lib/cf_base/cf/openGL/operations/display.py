import os
from OpenGL.GL import *
from cf.openGL.render import render_screen_quad
from cf.openGL.texture import Texture2D
from cf.openGL.operation import OpenGLOP

# Render a textured screen quad. Allows slicing of array textures and MIP levels and rendering directly to the screen
class ImageDisplayOP(OpenGLOP):

    def __init__(self, res, rendertarget_count=0):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"),
            None,
            os.path.join(this_dir, "shaders", "textured_quad"), 
            rendertarget_count,
            rendertarget_resolution=res)
        self.init_uniform("outputRes")
        self.init_uniform("showArray")
        self.init_uniform("level")
        self.init_uniform("layer")        
        self.init_uniform("showOverlay")
        self.init_uniform("overlayPosition")
        self.init_uniform("alpha_composite_overlay")
        self.res = res

    def render(self, tex, to_screen=False, overlay_tex=None, overlay_pos=(0, 0), level=0, layer=0, rendertarget_id=0, alpha_composite_overlay=True):
    
        if not to_screen:
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
                self.rendertargets[rendertarget_id].color.handle, 0)

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, self.res[0], self.res[1])
        glUseProgram(self.shader)

        if type(tex) == Texture2D:
            glActiveTexture(GL_TEXTURE0)
            glProgramUniform1i(self.shader, self.uniforms["showArray"], False)
        else:
            glActiveTexture(GL_TEXTURE1)
            glProgramUniform1i(self.shader, self.uniforms["showArray"], True)
        
        tex.bind()
        min_filter = GL_NEAREST if level == 0 else GL_NEAREST_MIPMAP_NEAREST
        tex.set_params(min_filter=min_filter)

        glProgramUniform2i(self.shader, self.uniforms["outputRes"], *self.res)
        glProgramUniform1i(self.shader, self.uniforms["level"], level)
        glProgramUniform1i(self.shader, self.uniforms["layer"], layer)
        
        if overlay_tex is not None:
            glProgramUniform1i(self.shader, self.uniforms["showOverlay"], True)
            glProgramUniform1i(self.shader, self.uniforms["alpha_composite_overlay"], alpha_composite_overlay)
            glActiveTexture(GL_TEXTURE2)
            overlay_tex.bind()
            overlay_tex.set_params()
            glProgramUniform2i(self.shader, self.uniforms["overlayPosition"], *overlay_pos)
        else:
            glProgramUniform1i(self.shader, self.uniforms["showOverlay"], False)

        glClear(GL_COLOR_BUFFER_BIT)
        render_screen_quad()
        
        tex.unbind()
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)

        if not to_screen:
            return self.rendertargets[rendertarget_id].color
       

#=====================================================================

# Arranges multiple Texture2Ds into a row (or column).
# This is particularly useful if the textures have different resolutions and thus cannot be put 
# into a Texture2DArray for visualization using the Texture2DArrayGridOP.
class Texture2DCollageOP(OpenGLOP):

    def __init__(self, res, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"),
            None,
            os.path.join(this_dir, "shaders", "texture_collage"), 
            rendertarget_count,
            rendertarget_resolution=res)
        self.init_uniform("res")
        self.init_uniform("texCount")
        self.init_uniform("vertical")

    def render(self, textures, vertical=False, rendertarget_id=0):
        assert rendertarget_id < self.rendertarget_count
        assert len(textures) < 9, "Current implementation supports only 8 textures."

        output_res = self.rendertargets[rendertarget_id].color.resolution

        glViewport(0, 0, *output_res)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        
        for idx, tex in enumerate(textures):
            glActiveTexture(GL_TEXTURE0 + idx)
            tex.bind()
            tex.set_params()

        glProgramUniform2i(self.shader, self.uniforms["res"], *output_res)
        glProgramUniform1i(self.shader, self.uniforms["texCount"], len(textures))
        glProgramUniform1i(self.shader, self.uniforms["vertical"], vertical)
        
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)
        render_screen_quad()

        glUseProgram(0)

        return self.rendertargets[rendertarget_id].color