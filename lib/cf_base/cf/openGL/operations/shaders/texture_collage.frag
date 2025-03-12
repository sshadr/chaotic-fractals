#version 420

layout (binding = 0) uniform sampler2D inputTexture0;
layout (binding = 1) uniform sampler2D inputTexture1;
layout (binding = 2) uniform sampler2D inputTexture2;
layout (binding = 3) uniform sampler2D inputTexture3;
layout (binding = 4) uniform sampler2D inputTexture4;
layout (binding = 5) uniform sampler2D inputTexture5;
layout (binding = 6) uniform sampler2D inputTexture6;
layout (binding = 7) uniform sampler2D inputTexture7;

uniform ivec2 res;
uniform int texCount;
uniform bool vertical = false;

in vec2 uv;

out vec4 out_color;


// this is somewhat ugly...
vec4 sampleTexture(int textureIdx, vec2 coord)
{
	switch(textureIdx)
	{
		case 0: return texture(inputTexture0, coord, 0);
		case 1: return texture(inputTexture1, coord, 0);
		case 2: return texture(inputTexture2, coord, 0);
		case 3: return texture(inputTexture3, coord, 0);
		case 4: return texture(inputTexture4, coord, 0);
		case 5: return texture(inputTexture5, coord, 0);
		case 6: return texture(inputTexture6, coord, 0);
		case 7: return texture(inputTexture7, coord, 0);
	}
}


void main()
{		
	vec2 position = ivec2(gl_FragCoord.xy) / vec2(res);
	vec2 scaling = (vertical) ? vec2(1, texCount) : vec2(texCount, 1);
	vec2 scaledUV = uv * scaling;

	for (int i=0; i<texCount; i++)
	{
		float posSlice = (vertical) ? position.y : position.x;
		if (posSlice < (i+1.) / texCount)
		{
			vec2 offset = (vertical) ? vec2(0, i) : vec2(i, 0);
			out_color = sampleTexture(i, scaledUV - offset);
			return;	
		}
	}
}