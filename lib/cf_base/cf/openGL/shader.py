from cf.openGL import *
from OpenGL.GL.shaders import compileProgram, compileShader

def insert_defines(src, defines):
    '''
    defines is a list: First element contains a dict, Second element contains the chosen key in dict
    '''
    if defines[0] is None or defines[1] not in defines[0]:
        return src

    dict = defines[0][defines[1]]
    # Iterate over the defines and format them as #define directives
    define_directives = ""
    
    for variable, value in dict.items():
        define_directives += f"#define {variable} {value}\n"
    
    lines = src.split('\n')
    lines.insert(1, define_directives)
    src = '\n'.join(lines)
    return src

# create a shader program, consisting of a vertex, fragment, and (optionally) geometry shader and a dictionary of pre-processor directives (compile time changes)
def create_shader(path_vertex, path_geometry, path_fragment, defines=None):
    
    vertex_src = insert_defines(open(path_vertex + ".vert", "r").read(), defines=[defines, "vert"])
    fragment_src = insert_defines(open(path_fragment + ".frag", "r").read(), defines=[defines, "frag"])
    
    arg_list = [
        compileShader(vertex_src, GL_VERTEX_SHADER), 
        compileShader(fragment_src, GL_FRAGMENT_SHADER)]
    if path_geometry:
        geometry_src = insert_defines(open(path_geometry + ".geom", "r").read(), defines=[defines, "geom"])
        arg_list.append(compileShader(geometry_src, GL_GEOMETRY_SHADER))
    return compileProgram(*arg_list)
