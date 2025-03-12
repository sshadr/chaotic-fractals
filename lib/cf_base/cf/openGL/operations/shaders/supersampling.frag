#version 420

layout(location=0) out vec4 out_color;

layout(binding=0) uniform sampler2D inputTexture2D;

uniform int factor;

//===================================================================================

void main()
{  					
	ivec2 baseCoord = ivec2(gl_FragCoord.xy) * factor;
	out_color = vec4(0);
	for (int x=0; x<factor; x++)
	{
		for (int y=0; y<factor; y++)
		{
			ivec2 sampleCoord = baseCoord + ivec2(x, y);
			vec4 fetch = texelFetch(inputTexture2D, sampleCoord, 0);
			
			// ignore pixels that are not fully opaque
			if (fetch.a != 1)
			{
				out_color = vec4(0);
				return;
			} 
			
			out_color += fetch;
		}	
	}

	out_color /= (out_color.a != 0) ? out_color.a : 1;
	
	// un-comment this line to revert back to single sampling
	// out_color = texelFetch(inputTexture2D, baseCoord, 0);
}
