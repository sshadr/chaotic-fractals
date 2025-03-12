from cf.openGL import *

# an OpenGL framebuffer
class Framebuffer:

    def __init__(self):
        self.handle = glGenFramebuffers(1)

    #def __del__(self):
        #glDeleteFramebuffers(1, [self.handle])

    def bind(self, target=GL_FRAMEBUFFER):
        glBindFramebuffer(target, self.handle)

    def unbind(self, target=GL_FRAMEBUFFER):
        glBindFramebuffer(target, 0)
