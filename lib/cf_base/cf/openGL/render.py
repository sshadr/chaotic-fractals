from cf.openGL import *

# create a screen-filling quad
def render_screen_quad():
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)
    glTexCoord2f(0, 1)
    glVertex2f(1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glTexCoord2f(1, 0)
    glVertex2f(-1, 1)
    glEnd()


# create quads from a vertex list
def render_quads(vertices):
    glBegin(GL_QUADS)
    for v in vertices:
        glVertex3f(*v)
    glEnd()


