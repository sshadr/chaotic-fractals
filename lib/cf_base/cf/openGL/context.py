from cf.openGL import *
from cf.openGL.window import OpenGLWindow
from cf.openGL.framebuffer import Framebuffer

class OpenGLContext:
    def __init__(self):
        if OGL_FRAMEWORK == OpenGLFramework.glfw:
            self.window = OpenGLWindow()
            self.fbo = self.window.fbo
        elif OGL_FRAMEWORK == OpenGLFramework.egl:
            self.fbo = Framebuffer()

    def bind(self):
        self.fbo.bind()
    
    def unbind(self):
        self.fbo.unbind()