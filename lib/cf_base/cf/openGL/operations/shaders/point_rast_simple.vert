#version 420 core

layout(location = 0) in vec2 position;

uniform mat3 modelMatrix = mat3(1);
uniform mat3 viewMatrix;

void discardPoint()
{
    gl_Position = vec4(-1000, 0, 0, 1);
}

mat3 construct_matrix(mat3 viewMatrix){
    // building correct transformation matrix- scaling and translation directions
    // Note: matrix is transposed in the input side
    mat3 new_view = viewMatrix;
    
    new_view[2][0] =  -1  * viewMatrix[2][0];
    new_view[2][1] = viewMatrix[2][1];
    return new_view;
}

void main()
{
    mat3 new_view = construct_matrix(viewMatrix);     
    vec3 pos = new_view * vec3(position, 1);

    gl_Position = vec4(pos.xy, 0, 1);

    // Adjust the point size to stay 1 pixel in screen space
    gl_PointSize = 1.f;
}