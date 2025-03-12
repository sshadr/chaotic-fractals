#version 420

layout(location=0) out vec4 out_color;
uniform vec3 opt_color;
in vec2 uv;

void main()
{
    out_color = vec4(opt_color, 1);
}
